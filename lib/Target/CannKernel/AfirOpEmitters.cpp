//===- AfirOpEmitters.cpp - Custom emitters for missing PyAsc ops ---------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// PyAsc's Translation.cpp does not yet have emitters for:
//   - ascendc::BroadcastL2Op
//   - ascendc::ReduceSum2DL2Op (and other ReduceND L2 ops)
//
// The ops exist in the AscendC dialect (TableGen-defined) but were omitted
// from PrintableOpTypes. This file provides correct emitters so that
// CannTranslation can handle them without patching PyAsc.
//
// Correct CANN adv_api signatures (from tikcfw/include/adv_api/):
//
//   Broadcast<T, dim, axis>(dst, src, dstShape[dim], srcShape[dim])
//     -- shape arrays are uint32_t[dim], NOT reinterpret_cast<uint64_t>
//
//   ReduceSum<T, AscendC::AR>(dst, src, sharedTmpBuf, srcShape[2], innerPad)
//     -- pattern types (AR, RA, ...) defined in adv_api/reduce/reduce_common.h
//     -- requires #include "adv_api/reduce/reduce.h" in the emitted .cpp
//
//===----------------------------------------------------------------------===//

#include "Target/CannKernel/AfirOpEmitters.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Target/Asc/CodeEmitter.h"
#include "ascir/Target/Asc/Common.h"

#include "mlir/Support/LogicalResult.h"

using namespace mlir;

// ─── BroadcastL2Op ──────────────────────────────────────────────────────────
//
// MLIR:
//   ascendc.broadcast_l2 %dst, %src, %ds0, %ds1, %ss0, %ss1
//       {constRank = 2 : i32, operandSegmentSizes = ...}
//       : local_tensor, local_tensor, i32, i32, i32, i32
//
// Emits:
//   {
//     uint32_t afir_dstShape[2] = {(uint32_t)ds0, (uint32_t)ds1};
//     uint32_t afir_srcShape[2] = {(uint32_t)ss0, (uint32_t)ss1};
//     AscendC::Broadcast<half, 2, 0>(dst, src, afir_dstShape, afir_srcShape);
//   }
//
// axis=0 means "broadcast along first axis", which matches our AR (rows→cols)
// use-case.  constRank drives the array size.
//
static LogicalResult emitBroadcastL2Op(CodeEmitter &emitter,
                                        ascendc::BroadcastL2Op op) {
  auto &os = emitter.ostream();
  uint32_t rank = op.getConstRank();

  // Build unique names for the shape arrays to avoid collisions in the same
  // scope (multiple broadcast ops can appear in one function body).
  std::string dstArrName =
      "afir_dstShape_" + std::to_string(reinterpret_cast<uintptr_t>(op->getLoc().getAsOpaquePointer()));
  std::string srcArrName =
      "afir_srcShape_" + std::to_string(reinterpret_cast<uintptr_t>(op->getLoc().getAsOpaquePointer()));

  // Emit a scoped block so the local arrays don't leak.
  os << "{\n";
  os.indent();

  os << "uint32_t " << dstArrName << "[" << rank << "] = {";
  auto dstShapes = op.getDstShape();
  for (unsigned i = 0; i < rank; ++i) {
    if (i) os << ", ";
    os << "(uint32_t)" << emitter.getOrCreateName(dstShapes[i]);
  }
  os << "};\n";

  os << "uint32_t " << srcArrName << "[" << rank << "] = {";
  auto srcShapes = op.getSrcShape();
  for (unsigned i = 0; i < rank; ++i) {
    if (i) os << ", ";
    os << "(uint32_t)" << emitter.getOrCreateName(srcShapes[i]);
  }
  os << "};\n";

  // axis=0: broadcast the first (A) dimension.  This matches AR layout
  // (source has shape [1, N], destination [M, N]).
  // Use the actual element type of the dst tensor (not hardcoded 'half').
  auto dstElemType =
      cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
  os << ascNamespace << "::Broadcast<";
  if (failed(emitter.emitType(op.getLoc(), dstElemType)))
    return failure();
  os << ", " << rank << ", 0>("
     << emitter.getOrCreateName(op.getDst()) << ", "
     << emitter.getOrCreateName(op.getSrc()) << ", "
     << dstArrName << ", " << srcArrName << ")";

  os.unindent();
  os << "\n}";
  return success();
}

// ─── ReduceSum2DL2Op ────────────────────────────────────────────────────────
//
// MLIR:
//   ascendc.reduce_sum_2d_l2 %dst, %src {layout = 0 : i32}
//       : local_tensor, local_tensor
//
// The MLIR op has no explicit shape operands; shape info lives in TBuf sizes.
// The adv_api ReduceSum<T, pattern> requires srcShape[dim] at runtime.
//
// Because the kernel body already has the individual dimension variables
// (e.g. v31 = tb_m_rows, v18 = dim_n), we need to reconstruct the shape.
// However at the IR level the op only carries dst and src; we cannot easily
// extract the shape from the op alone.
//
// Strategy: emit ReduceSum<T, AR>(dst, src, sharedTmpBuf, srcShape, innerPad)
// using a PopStackBuffer for the tmp.  We declare a local uint32_t[2] for
// srcShape.  The values are taken from the TBuf sizes via GetSize() helpers
// that are emitted inline.
//
// Simpler alternative that avoids shape reasoning: use the L1 WholeReduceSum
// in a loop, or emit a verbatim note and rely on the user to patch.
//
// For correctness we emit the adv_api ReduceSum.  The two dimensions are:
//   srcShape[0] = dst.GetSize() / sizeof(T)   (A dimension = number of rows)
//   srcShape[1] = src.GetSize() / dst.GetSize()  (R dimension = row length)
// But GetSize returns byte count, so:
//   rows = dst.GetSize() / sizeof(half)
//   cols = src.GetSize() / dst.GetSize()
//
// Emits:
//   {
//     uint32_t afir_shape[2] = {
//         (uint32_t)(dst.GetSize() / sizeof(half)),
//         (uint32_t)(src.GetSize() / dst.GetSize()) };
//     AscendC::LocalTensor<uint8_t> afir_tmp;
//     AscendC::PopStackBuffer<uint8_t, AscendC::TPosition::LCM>(afir_tmp);
//     AscendC::ReduceSum<half, AscendC::AR>(dst, src, afir_tmp, afir_shape, false);
//   }
//
static LogicalResult emitReduceSum2DL2Op(CodeEmitter &emitter,
                                          ascendc::ReduceSum2DL2Op op) {
  auto &os = emitter.ostream();

  const char *patternStr =
      (op.getLayout() == ascendc::ReduceLayout::AR) ? "AR" : "RA";

  std::string shapeArr =
      "afir_shape_" + std::to_string(reinterpret_cast<uintptr_t>(op->getLoc().getAsOpaquePointer()));
  std::string tmpTensor =
      "afir_tmp_" + std::to_string(reinterpret_cast<uintptr_t>(op->getLoc().getAsOpaquePointer()));

  std::string dstName = emitter.getOrCreateName(op.getDst()).str();
  std::string srcName = emitter.getOrCreateName(op.getSrc()).str();

  os << "{\n";
  os.indent();

  os << "uint32_t " << shapeArr << "[2] = {\n";
  os.indent();
  os << "(uint32_t)(" << dstName << ".GetSize() / sizeof(half)),\n";
  os << "(uint32_t)(" << srcName << ".GetSize() / " << dstName
     << ".GetSize())};\n";
  os.unindent();

  os << ascNamespace << "::LocalTensor<uint8_t> " << tmpTensor << ";\n";
  os << ascNamespace << "::PopStackBuffer<uint8_t, "
     << ascNamespace << "::TPosition::LCM>(" << tmpTensor << ");\n";

  os << ascNamespace << "::ReduceSum<half, "
     << ascNamespace << "::" << patternStr << ">("
     << dstName << ", " << srcName << ", "
     << tmpTensor << ", " << shapeArr << ", false)";

  os.unindent();
  os << "\n}";
  return success();
}

// ─── Public entry point ─────────────────────────────────────────────────────

mlir::LogicalResult mlir::afir::tryEmitAfirOp(CodeEmitter &emitter,
                                               mlir::Operation &op) {
  if (auto broadcastOp = dyn_cast<ascendc::BroadcastL2Op>(&op))
    return emitBroadcastL2Op(emitter, broadcastOp);
  if (auto reduceOp = dyn_cast<ascendc::ReduceSum2DL2Op>(&op))
    return emitReduceSum2DL2Op(emitter, reduceOp);
  return failure(); // not handled here; fall back to PyAsc
}
