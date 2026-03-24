//===- CannTranslation.cpp - CANN kernel C++ translation --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/CannKernel/CannTranslation.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Target/Asc/CodeEmitter.h"
#include "ascir/Target/Asc/Common.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {

/// Emit the TilingData struct declaration from a PyStructType.
static LogicalResult emitTilingStructDecl(CodeEmitter &emitter, Location loc,
                                          emitasc::PyStructType pyType) {
  auto &os = emitter.ostream();
  StringRef structName = pyType.getNameAttr().getValue();
  os << "struct " << structName << " {\n";
  os.indent();

  auto typesAttr = pyType.getTypesAttr();
  auto namesAttr = pyType.getNamesAttr();
  auto types = typesAttr.getValue();
  auto names = namesAttr.getValue();

  if (types.size() != names.size())
    return emitError(loc, "PyStructType types/names size mismatch");

  for (size_t i = 0; i < types.size(); ++i) {
    Type fieldType = cast<TypeAttr>(types[i]).getValue();
    StringRef fieldName = cast<StringAttr>(names[i]).getValue();
    if (failed(emitter.emitType(loc, fieldType)))
      return failure();
    os << " " << fieldName << ";\n";
  }

  os.unindent() << "};\n\n";
  return success();
}

/// Emit the CANN-standard function signature and body.
static LogicalResult printCannFuncOp(CodeEmitter &emitter,
                                     func::FuncOp funcOp) {
  CodeEmitter::Scope scope(emitter);
  auto &os = emitter.ostream();

  // cann.num_inputs must be present (set by CanonicalizeCannSignaturePass).
  if (!funcOp->hasAttr("cann.num_inputs"))
    return funcOp.emitOpError("missing cann.num_inputs attribute; "
                               "run --canonicalize-cann-signature first");

  auto args = funcOp.getArguments();
  int numArgs = (int)args.size();

  // Layout: [0..N-3] = inputs+outputs (all GM_ADDR), [N-2] = workspace
  // (memref<ui8>), [N-1] = tiling (!emitasc.py_struct).
  if (numArgs < 4)
    return funcOp.emitOpError(
        "CANN function must have at least 4 args "
        "(inputs, outputs, workspace, tiling)");

  BlockArgument tilingArg = args[numArgs - 1];
  auto tilingType = dyn_cast<emitasc::PyStructType>(tilingArg.getType());
  if (!tilingType)
    return funcOp.emitOpError("last argument must be !emitasc.py_struct");

  // Validate workspace arg is memref<ui8>.
  auto wsType = dyn_cast<MemRefType>(args[numArgs - 2].getType());
  if (!wsType || !wsType.getElementType().isUnsignedInteger(8))
    return funcOp.emitOpError(
        "second-to-last argument must be memref<ui8> workspace");

  // Emit function header
  os << "extern \"C\" __global__ __aicore__ void " << funcOp.getName() << "(\n";
  os.indent();

  // Emit input/output GM_ADDR args (all memref args)
  for (int i = 0; i < numArgs - 1; ++i) {
    os << "GM_ADDR " << emitter.getOrCreateName(args[i]);
    os << ",\n";
  }

  // Emit tiling arg as struct by value
  StringRef tilingStructName = tilingType.getNameAttr().getValue();
  os << tilingStructName << " " << emitter.getOrCreateName(tilingArg) << "\n";

  os.unindent() << ") {\n";
  os.indent();

  // Emit body ops
  for (Block &block : funcOp.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      // Skip func.return (void kernel, no return value needed)
      if (isa<func::ReturnOp>(op)) {
        os << "return;\n";
        continue;
      }
      if (failed(emitOperation(emitter, op, needsSemicolon(&op))))
        return failure();
    }
  }

  os.unindent() << "}\n";
  return success();
}

} // namespace

// ─── Pre-pass: replace broken PyAsc emitter ops with emitasc.verbatim ───────
//
// PyAsc's auto-generated printOperation() for BroadcastL2Op emits a wrong
// reinterpret_cast<uint64_t> and for ReduceSum2DL2Op uses the non-existent
// AscendC::ReduceLayout enum.  Both ops live in nested SCF regions that are
// recursively emitted by PyAsc's internal emitOperation() – meaning our
// top-level tryEmitAfirOp hook never reaches them.
//
// Solution: before calling the emitter, walk every occurrence of these ops and
// replace each one with an emitasc.verbatim that produces the correct C++.
// The verbatim op is in PyAsc's PrintableOpTypes and is emitted verbatim,
// so no further interception is needed.
//
// BroadcastL2Op operands:  (dst, src, dstShape..., srcShape...)
//   constRank attr tells us the shape array length.
//   Verbatim template (block-scoped to avoid name collisions):
//     {
//       uint32_t _ds[N] = {(uint32_t)$2, ...};
//       uint32_t _ss[N] = {(uint32_t)$K, ...};
//       AscendC::Broadcast<half,N,0>($0, $1, _ds, _ss);
//     }
//
// ReduceSum2DL2Op operands: (dst, src)
//   layout attr = AR (0) or RA (1).
//   Verbatim template:
//     {
//       uint32_t _s[2]={(uint32_t)($0.GetSize()/sizeof(half)),
//                       (uint32_t)($1.GetSize()/$0.GetSize())};
//       AscendC::LocalTensor<uint8_t> _t;
//       AscendC::PopStackBuffer<uint8_t,AscendC::TPosition::LCM>(_t);
//       AscendC::ReduceSum<half,AscendC::AR>($0,$1,_t,_s,false);
//     }
//
static void fixBrokenOpEmitters(Operation *moduleOp) {
  IRRewriter rewriter(moduleOp->getContext());

  // GlobalTensorSetGlobalBufferOp → verbatim with pointer arithmetic offset.
  //
  // PyAsc auto-generates: $tensor.SetGlobalBuffer($buffer_ptr, $offset)
  // But AscendC SetGlobalBuffer(ptr, uint64_t) treats the 2nd arg as a SIZE
  // hint, NOT an element offset.  As a result, DataCopy always reads/writes
  // from the base pointer, ignoring the offset.  This breaks multi-block
  // kernels where each block operates on a different slice of the buffer.
  //
  // Fix: emit pointer arithmetic to bake the offset into the pointer:
  //   $tensor.SetGlobalBuffer($buffer_ptr + $offset);
  //
  // When the offset is absent (op.getSize() is null), emit the 1-arg form.
  moduleOp->walk([&](ascendc::GlobalTensorSetGlobalBufferOp op) {
    Value sizeVal = op.getSize();
    if (!sizeVal)
      return; // no offset — let PyAsc emit the 1-arg form unchanged

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    // Verbatim: $0 = tensor, $1 = buffer_ptr (__gm__ half*), $2 = offset (i32)
    // Emit: $0.SetGlobalBuffer($1 + $2);
    rewriter.create<emitasc::VerbatimOp>(
        loc,
        rewriter.getStringAttr("$0.SetGlobalBuffer($1 + $2)"),
        ValueRange({op.getTensor(), op.getBuffer(), sizeVal}));
    rewriter.eraseOp(op);
  });

  // BroadcastL2Op → verbatim
  moduleOp->walk([&](ascendc::BroadcastL2Op op) {
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    uint32_t rank = op.getConstRank();

    // Build verbatim string with $N placeholders.
    // Operand layout: $0=dst, $1=src, $2..$2+rank-1=dstShape, $2+rank..=srcShape
    std::string tmpl = "{\n";
    tmpl += "  uint32_t _afir_ds[" + std::to_string(rank) + "] = {";
    for (uint32_t i = 0; i < rank; ++i) {
      if (i) tmpl += ", ";
      tmpl += "(uint32_t)$" + std::to_string(2 + i);
    }
    tmpl += "};\n";
    tmpl += "  uint32_t _afir_ss[" + std::to_string(rank) + "] = {";
    for (uint32_t i = 0; i < rank; ++i) {
      if (i) tmpl += ", ";
      tmpl += "(uint32_t)$" + std::to_string(2 + rank + i);
    }
    tmpl += "};\n";
    // Determine axis: if srcShape[-1] == 1 (column broadcast), axis=1.
    // If srcShape[0] == 1 (row broadcast), axis=0.
    // Inspect the last srcShape value operand: if it is a constant 1, use axis=1.
    auto srcShapeVals = op.getSrcShape();
    int axis = 0;
    if (!srcShapeVals.empty()) {
      Value lastSrc = srcShapeVals[srcShapeVals.size() - 1];
      if (auto constOp = lastSrc.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          if (intAttr.getInt() == 1)
            axis = 1;
        }
      }
    }
    tmpl += "  AscendC::Broadcast<half, " + std::to_string(rank) +
            ", " + std::to_string(axis) + ">($0, $1, _afir_ds, _afir_ss);\n}";

    SmallVector<Value> args;
    args.push_back(op.getDst());
    args.push_back(op.getSrc());
    for (Value v : op.getDstShape())
      args.push_back(v);
    for (Value v : op.getSrcShape())
      args.push_back(v);

    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl), ValueRange(args));
    rewriter.eraseOp(op);
  });

  // ReduceSum2DL2Op → verbatim
  //
  // AR layout: src[rows, cols] → dst[rows] by summing each row.
  // RA layout: not yet implemented.
  //
  // Uses AscendC::ReduceSum<half> per-row with a 32-byte scratch VECCALC TBuf.
  // GetValue/SetValue scalar loops over individual elements are avoided because
  // the simulator's LocalTensor::GetValue() does not correctly access elements
  // beyond the first 16 when called on a TBuf::Get() tensor allocated inside a
  // loop (the simulator does not update the LocalTensor's internal size field
  // for loop-iteration-dependent InitBuffer calls).
  //
  // We look up InitBuffer/InitQueue ops in the IR to get the byte-lengths as
  // explicit SSA operands ($2 = dst_queue_bytes = rows*2,
  // $3 = src_tbuf_bytes = rows*N*2), avoiding GetSize() entirely.
  //
  // Operand layout in the emitted verbatim:
  //   $0 = dst (VECOUT TQue LocalTensor)
  //   $1 = src (VECCALC accumulator LocalTensor)
  //   $2 = dst queue byte-length (= rows * sizeof(half)) from TPipeInitQueueOp
  //   $3 = src tbuf byte-length (= rows * N * sizeof(half)) from TPipeInitBufferOp
  moduleOp->walk([&](ascendc::ReduceSum2DL2Op op) {
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    bool isAR = (op.getLayout() == ascendc::ReduceLayout::AR);

    // $2: find TPipeInitQueueOp length for the dst TQue (rows * sizeof(half)).
    Value dstQueueLenVal;
    if (auto allocOp =
            op.getDst().getDefiningOp<ascendc::TQueBindAllocTensorOp>()) {
      Value queueVal = allocOp.getQueue();
      for (auto *user : queueVal.getUsers()) {
        if (auto initQ = dyn_cast<ascendc::TPipeInitQueueOp>(user)) {
          dstQueueLenVal = initQ.getLength();
          break;
        }
      }
    }

    // $3: find TPipeInitBufferOp length for the src TBuf (rows*N*sizeof(half)).
    Value srcTBufLenVal;
    if (auto getOp = op.getSrc().getDefiningOp<ascendc::TBufGetTensorOp>()) {
      Value tbufVal = getOp.getBuffer();
      for (auto *user : tbufVal.getUsers()) {
        if (auto initB = dyn_cast<ascendc::TPipeInitBufferOp>(user)) {
          srcTBufLenVal = initB.getLength();
          break;
        }
      }
    }

    std::string tmpl = "{\n";
    if (isAR) {
      // AR: dst[r] = sum(src[r*cols .. r*cols+cols-1])
      // Use ReduceSum<half> per row with a 32-byte scratch VECCALC TBuf.
      // The TPipe is passed as the last operand so we can InitBuffer the scratch.
      // $1[r * cols] slices the src tensor to the start of row r.
      std::string pipeRef; // placeholder name for pipe arg
      if (dstQueueLenVal && srcTBufLenVal) {
        // $2 = dst_bytes, $3 = src_bytes, $4 = pipe
        tmpl += "  uint32_t _afir_rows = (uint32_t)($2 / sizeof(half));\n";
        tmpl += "  uint32_t _afir_cols = (uint32_t)($3 / $2);\n";
        pipeRef = "$4";
      } else if (dstQueueLenVal) {
        // $2 = dst_bytes, $3 = pipe
        tmpl += "  uint32_t _afir_rows = (uint32_t)($2 / sizeof(half));\n";
        tmpl += "  uint32_t _afir_cols = (uint32_t)($1.GetSize() / $2);\n";
        pipeRef = "$3";
      } else {
        // $2 = pipe
        tmpl += "  uint32_t _afir_rows = (uint32_t)($0.GetSize() / sizeof(half));\n";
        tmpl += "  uint32_t _afir_cols = (uint32_t)($1.GetSize() / $0.GetSize());\n";
        pipeRef = "$2";
      }
      // Use TWO separate TBufs: _afir_tbuf_dst (result) and _afir_tbuf_ws (workspace).
      // ReduceSum requires dst != sharedTmpBuffer; aliasing them gives wrong results
      // on arch 3101 because the intermediate tree-reduction overwrites the output.
      tmpl += "  AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_tbuf_dst;\n";
      tmpl += "  AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_tbuf_ws;\n";
      tmpl += "  " + pipeRef + ".InitBuffer(_afir_tbuf_dst, 32);\n";
      tmpl += "  " + pipeRef + ".InitBuffer(_afir_tbuf_ws, 32);\n";
      tmpl += "  AscendC::LocalTensor<half> _afir_scalar = _afir_tbuf_dst.Get<half>();\n";
      tmpl += "  AscendC::LocalTensor<half> _afir_ws = _afir_tbuf_ws.Get<half>();\n";
      tmpl += "  for (uint32_t _afir_r = 0; _afir_r < _afir_rows; _afir_r++) {\n";
      tmpl += "    AscendC::ReduceSum<half>(_afir_scalar, $1[_afir_r * _afir_cols],\n";
      tmpl += "                            _afir_ws, (int32_t)_afir_cols);\n";
      tmpl += "    $0.SetValue(_afir_r, _afir_scalar.GetValue(0));\n";
      tmpl += "  }\n}";
    } else {
      tmpl += "  // RA layout not yet implemented\n}";
    }

    // Find the TPipe value: walk enclosing function for PipeOp.
    Value pipeVal;
    if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
      funcOp.walk([&](ascendc::PipeOp pipeOp) {
        pipeVal = pipeOp.getResult();
        return WalkResult::interrupt();
      });
    }

    SmallVector<Value> args = {op.getDst(), op.getSrc()};
    if (dstQueueLenVal)
      args.push_back(dstQueueLenVal);
    if (srcTBufLenVal)
      args.push_back(srcTBufLenVal);
    if (pipeVal)
      args.push_back(pipeVal);

    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl), ValueRange(args));
    rewriter.eraseOp(op);
  });
}

LogicalResult mlir::translateToCannKernel(Operation *op, raw_ostream &os) {
  auto moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp)
    return op->emitOpError("expected a module op");

  // Replace ops whose PyAsc emitters generate wrong C++ with verbatim.
  fixBrokenOpEmitters(op);

  CodeEmitter emitter(os);
  CodeEmitter::Scope scope(emitter);

  os << "#include \"kernel_operator.h\"\n";
  // adv_api headers required by BroadcastL2Op and ReduceSum2DL2Op emitters.
  // These are not included by kernel_operator.h but are available via the
  // tikcfw/include search path added by the compiler driver.
  os << "#include \"adv_api/broadcast/broadcast.h\"\n";
  os << "#include \"adv_api/reduce/reduce.h\"\n";
  os << "\n";

  // First pass: emit TilingData struct declarations from aicore funcs
  for (Operation &child : moduleOp.getBody()->getOperations()) {
    auto funcOp = dyn_cast<func::FuncOp>(child);
    if (!funcOp)
      continue;
    if (!funcOp->hasAttr(ascendc::attr::global))
      continue;

    auto args = funcOp.getArguments();
    if (args.empty())
      continue;
    auto tilingType =
        dyn_cast<emitasc::PyStructType>(args.back().getType());
    if (!tilingType)
      continue;

    if (failed(emitTilingStructDecl(emitter, funcOp.getLoc(), tilingType)))
      return failure();
  }

  // Second pass: emit aicore kernel functions only.
  // Non-aicore ops (transform sequences, helper modules, etc.) are skipped —
  // they are pipeline infrastructure, not C++ kernel code.
  for (Operation &child : moduleOp.getBody()->getOperations()) {
    auto funcOp = dyn_cast<func::FuncOp>(child);
    if (!funcOp || !funcOp->hasAttr(ascendc::attr::global))
      continue;
    if (failed(printCannFuncOp(emitter, funcOp)))
      return failure();
  }

  return success();
}
