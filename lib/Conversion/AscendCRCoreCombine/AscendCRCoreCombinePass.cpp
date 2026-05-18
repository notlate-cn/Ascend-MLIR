//===- AscendCRCoreCombinePass.cpp - RCore SyncAll combine ----------------===//
//
// Part of the Ascend-MLIR Project
//
// Rewrites the per-core scalar write in an RCore reduce kernel into the
// AF kRCore two-segment pattern:
//   segment 1: each core writes its partial to workspace[block_idx]
//   SyncAllHard
//   segment 2: block 0 sums workspace[0..block_dim] → final out
//
// Runs after canonicalize-cann-signature.  See the .td description for the
// IR shape produced.
//===----------------------------------------------------------------------===//

#include "Conversion/AscendCRCoreCombine/AscendCRCoreCombinePass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#define GEN_PASS_DECL_ASCENDCRCORECOMBINEPASS
#define GEN_PASS_DEF_ASCENDCRCORECOMBINEPASS
#include "Conversion/Passes.h.inc"

#define DEBUG_TYPE "ascendc-rcore-combine"

using namespace mlir;

namespace mlir::afir {
namespace {

// Walk a function for the *single* data_copy_l2 op whose destination GM
// tensor is bound (via set_global_buffer) to a BlockArgument of the func —
// i.e. the "kernel's final write to GM output" instruction.  Returns
// {set_global_buffer, data_copy_l2, output_arg} on success.
struct OutputWrite {
  ascendc::GlobalTensorSetGlobalBufferOp setGB;
  ascendc::DataCopyL2Op                    dcp;
  BlockArgument                            outputArg;
};

// For an ascendc.global_tensor SSA value, find the (single) set_global_buffer
// op that binds it to a buffer.
static ascendc::GlobalTensorSetGlobalBufferOp
findBinding(Value gt) {
  ascendc::GlobalTensorSetGlobalBufferOp setGB;
  for (Operation *user : gt.getUsers()) {
    if (auto s = dyn_cast<ascendc::GlobalTensorSetGlobalBufferOp>(user)) {
      if (setGB)
        return {}; // multiple bindings — ambiguous
      setGB = s;
    }
  }
  return setGB;
}

// Find the unique data_copy_l2 in `func` whose dst is a global_tensor bound
// (via set_global_buffer) to a func BlockArgument — i.e. the kernel's GM
// write of its scalar / per-core result.
static std::optional<OutputWrite> findFinalGMWrite(func::FuncOp func) {
  OutputWrite found{};
  bool multiple = false;
  func.walk([&](ascendc::DataCopyL2Op dcp) {
    Value dst = dcp.getDst();
    // The dst type must be a global tensor for this to be a UB→GM write.
    auto gtOp = dst.getDefiningOp<ascendc::GlobalTensorOp>();
    if (!gtOp)
      return;
    auto setGB = findBinding(dst);
    if (!setGB)
      return;
    Value buf = setGB.getBuffer();
    // Allow either the raw block arg or a reinterpret_cast(block arg).
    BlockArgument blkArg;
    if (auto cast = buf.getDefiningOp<emitasc::ReinterpretCastOp>())
      blkArg = dyn_cast<BlockArgument>(cast.getOperand());
    else
      blkArg = dyn_cast<BlockArgument>(buf);
    if (!blkArg || blkArg.getOwner() != &func.getBody().front())
      return;
    if (found.dcp)
      multiple = true;
    found.setGB = setGB;
    found.dcp = dcp;
    found.outputArg = blkArg;
  });
  if (!found.dcp || multiple)
    return std::nullopt;
  return found;
}

// Walk back from the in-bounds scf.if condition (cmpi ult, block_idx*step, ub)
// to recover the step and ub Values.  block_dim = ceildivui(ub, step).
static bool extractBlockDimRecipe(scf::IfOp ifOp, Value &step, Value &ub,
                                   Value &blockIdx) {
  auto cmp = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::ult)
    return false;
  auto mul = cmp.getLhs().getDefiningOp<arith::MulIOp>();
  if (!mul)
    return false;
  // Either operand could be block_idx; identify it by op kind.
  if (mul.getLhs().getDefiningOp<ascendc::GetBlockIdxOp>()) {
    blockIdx = mul.getLhs();
    step = mul.getRhs();
  } else if (mul.getRhs().getDefiningOp<ascendc::GetBlockIdxOp>()) {
    blockIdx = mul.getRhs();
    step = mul.getLhs();
  } else {
    return false;
  }
  ub = cmp.getRhs();
  return true;
}

// Find the TPipe instance in the func.  AscendC kernels have exactly one.
static Value findTPipe(func::FuncOp func) {
  Value result;
  func.walk([&](ascendc::PipeOp p) { result = p.getResult(); });
  return result;
}

// Find the workspace func arg (the canonical CANN-ABI memref<...ui8>).
static BlockArgument findWorkspaceArg(func::FuncOp func) {
  for (BlockArgument arg : func.getBody().front().getArguments()) {
    auto mrt = dyn_cast<MemRefType>(arg.getType());
    if (!mrt) continue;
    auto it = dyn_cast<IntegerType>(mrt.getElementType());
    if (it && it.getWidth() == 8 && it.isUnsignedInteger())
      return arg;
  }
  return {};
}

static void transformOneFunc(func::FuncOp func) {
  auto rt = func->getAttrOfType<StringAttr>("afir.reduce_template");
  if (!rt || rt.getValue() != "RCore")
    return;

  auto write = findFinalGMWrite(func);
  if (!write) {
    LLVM_DEBUG(llvm::dbgs() << "[rcore-combine] " << func.getName()
                            << ": no unique final GM write — skipping\n");
    return;
  }

  // Containing scf.if = parallelize's in-bounds guard.
  auto parentIf = write->dcp->getParentOfType<scf::IfOp>();
  if (!parentIf) {
    LLVM_DEBUG(llvm::dbgs() << "[rcore-combine] " << func.getName()
                            << ": final write not inside scf.if — skipping\n");
    return;
  }

  Value step, ub, blockIdx;
  if (!extractBlockDimRecipe(parentIf, step, ub, blockIdx)) {
    LLVM_DEBUG(llvm::dbgs() << "[rcore-combine] " << func.getName()
                            << ": couldn't extract block_dim recipe\n");
    return;
  }

  Value pipeVal = findTPipe(func);
  BlockArgument wsArg = findWorkspaceArg(func);
  if (!pipeVal || !wsArg) {
    LLVM_DEBUG(llvm::dbgs() << "[rcore-combine] " << func.getName()
                            << ": missing TPipe or workspace arg\n");
    return;
  }

  MLIRContext *ctx = func.getContext();
  Location loc = write->dcp.getLoc();
  Type elemTy =
      cast<ascendc::LocalTensorType>(write->dcp.getSrc().getType()).getElementType();
  if (!elemTy.isF32()) {
    // P1: only f32 for now.  f16/bf16 want a different reduce_sum_2d_l2 layout
    // size table; defer until needed.
    LLVM_DEBUG(llvm::dbgs() << "[rcore-combine] " << func.getName()
                            << ": non-f32 elem type, skipping\n");
    return;
  }

  // --- Step 1: redirect the racy write to workspace[block_idx] ---
  // Replace set_global_buffer's buffer with a workspace view, and add
  // block_idx as the optional offset arg (in elements).
  OpBuilder b(write->setGB);
  // Reinterpret workspace `memref<?xui8>` as `memref<?xf32, 22>` (GM space 22).
  auto wsFloatTy = MemRefType::get(
      {ShapedType::kDynamic}, elemTy, MemRefLayoutAttrInterface{},
      IntegerAttr::get(IntegerType::get(ctx, 32), 22));
  Value wsFloat =
      b.create<emitasc::ReinterpretCastOp>(loc, wsFloatTy, wsArg);
  // Per-core partial slot offset (in elements): 64 ints (= 256 bytes) reserved
  // at the start of workspace for the soft-sync flag area, then block_idx of
  // contiguous float slots.  Keep in sync with the verbatim combine block.
  Value blockIdxI32 = b.create<arith::IndexCastOp>(
      loc, IntegerType::get(ctx, 32), blockIdx);
  Value c64 = b.create<arith::ConstantIntOp>(
      loc, IntegerType::get(ctx, 32), 64);
  Value slotOff = b.create<arith::AddIOp>(loc, blockIdxI32, c64);
  // Replace the operand: setGB now binds to (ws_view, offset = slotOff).
  write->setGB.getBufferMutable().assign(wsFloat);
  if (write->setGB.getSize())
    write->setGB.getSizeMutable().assign(slotOff);
  else {
    OpBuilder rb(write->setGB);
    rb.create<ascendc::GlobalTensorSetGlobalBufferOp>(
        write->setGB.getLoc(), write->setGB.getTensor(), wsFloat, slotOff);
    write->setGB.erase();
  }

  // --- Step 2: after the scf.if, emit sync + block-0 combine ---
  b.setInsertionPointAfter(parentIf);

  // block_dim = ceildivsi(ub, step).  Use the signed variant — afir-translate
  // has a printer for it (and matches what TilePlanGen emits in block_dim_expr).
  Value blockDim = b.create<arith::CeilDivSIOp>(loc, ub, step);
  Value blockDimI32 = b.create<arith::IndexCastOp>(
      loc, IntegerType::get(ctx, 32), blockDim);

  // The original output arg.  Used inside the combine block.
  Value outArg = write->outputArg;

  // Emit the soft-sync barrier + block-0 combine via a single verbatim.
  // We use the explicit-arg `SyncAll<false>(gmWs, ubWs, usedCores)` (soft-
  // sync via GM counter) rather than the no-arg variant — the latter uses
  // hardware cross-core flags which the sim build of AscendC doesn't
  // implement (the kernel hangs at session.plan / setflag_error).
  //
  // Workspace partition (we own all of it, runtime gives 16MB):
  //   [0          .. 64*sizeof(int32_t))    sync flag area (≥ usedCores ints)
  //   [256        .. 256 + numAICores*4)    per-core partial slots (float)
  // The redirect above writes the per-core partial at byte offset
  // `256 + block_idx * sizeof(float)`.
  //
  // Operand mapping:
  //   $0 = TPipe   (used as v_pipe.InitBuffer)
  //   $1 = workspace block arg  (raw GM_ADDR)
  //   $2 = output GM arg        (raw GM_ADDR)
  //   $3 = block_dim_rt (i32) — also the soft-sync usedCores
  std::string tmpl =
      "{\n"
      "  // Make sure each core's partial DataCopyPad to workspace[slot] is\n"
      "  // committed to GM before the soft-sync barrier — otherwise block 0\n"
      "  // can read stale slots.\n"
      "  AscendC::PipeBarrier<PIPE_ALL>();\n"
      "  AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_sync_ub_buf;\n"
      "  $0.InitBuffer(_afir_sync_ub_buf, 64u * (uint32_t)sizeof(int32_t));\n"
      "  AscendC::LocalTensor<int32_t> _afir_sync_ub =\n"
      "      _afir_sync_ub_buf.Get<int32_t>();\n"
      "  AscendC::GlobalTensor<int32_t> _afir_sync_gm;\n"
      "  _afir_sync_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>($1));\n"
      "  AscendC::SyncAll<false>(_afir_sync_gm, _afir_sync_ub, (int32_t)$3);\n"
      "}\n"
      "if (AscendC::GetBlockIdx() == 0) {\n"
      "  uint32_t _afir_bd = (uint32_t)$3;\n"
      "  uint32_t _afir_bytes = ((_afir_bd * (uint32_t)sizeof(float) + 31u) / 32u) * 32u;\n"
      "  AscendC::TBuf<AscendC::TPosition::VECIN> _afir_ws_buf;\n"
      "  AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_res_buf;\n"
      "  AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_tmp_buf;\n"
      "  $0.InitBuffer(_afir_ws_buf, _afir_bytes);\n"
      "  $0.InitBuffer(_afir_res_buf, 32);\n"
      "  $0.InitBuffer(_afir_tmp_buf, _afir_bytes);\n"
      "  AscendC::LocalTensor<float> _afir_ws_lt = _afir_ws_buf.Get<float>();\n"
      "  AscendC::LocalTensor<float> _afir_res_lt = _afir_res_buf.Get<float>();\n"
      "  AscendC::LocalTensor<uint8_t> _afir_tmp_lt = _afir_tmp_buf.Get<uint8_t>();\n"
      "  AscendC::GlobalTensor<float> _afir_ws_gt;\n"
      "  _afir_ws_gt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>($1 + 256));\n"
      "  uint32_t _afir_padded = _afir_bytes / (uint32_t)sizeof(float);\n"
      "  AscendC::DataCopy(_afir_ws_lt, _afir_ws_gt, _afir_padded);\n"
      "  AscendC::PipeBarrier<PIPE_ALL>();\n"
      "  // Scalar sum of the block_dim partials.  block_dim ≤ numAICores (≈40)\n"
      "  // so a scalar loop is cheaper than the vectorized ReduceSum setup;\n"
      "  // also sidesteps ReduceSum<AR>'s minimum-row-width requirements.\n"
      "  float _afir_total = 0.0f;\n"
      "  for (uint32_t _afir_i = 0; _afir_i < _afir_bd; _afir_i++) {\n"
      "    _afir_total += _afir_ws_lt.GetValue(_afir_i);\n"
      "  }\n"
      "  _afir_res_lt.SetValue(0, _afir_total);\n"
      "  AscendC::PipeBarrier<PIPE_ALL>();\n"
      "  AscendC::GlobalTensor<float> _afir_out_gt;\n"
      "  _afir_out_gt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>($2));\n"
      "  AscendC::DataCopyExtParams _afir_dcp{\n"
      "      (uint16_t)1, (uint32_t)sizeof(float), (uint32_t)0, (uint32_t)0, (uint32_t)0};\n"
      "  AscendC::DataCopyPad(_afir_out_gt, _afir_res_lt, _afir_dcp);\n"
      "}\n";
  b.create<emitasc::VerbatimOp>(
      loc, b.getStringAttr(tmpl),
      ValueRange({pipeVal, wsArg, outArg, blockDimI32}));

  // CanonicalizeCannSignature's isArgWritten walker requires
  // arg → reinterpret_cast → set_global_buffer → data_copy_l2; for a rank-0
  // DPS init (`memref<f32>`) there's no reinterpret_cast in the chain, so
  // the heuristic misses it and counts the init as an input.  After we
  // redirect the per-core write away from the init and add a fresh write
  // inside the combine, the chain is `arg → set_global_buffer (inside the
  // verbatim, invisible to the walker) → data_copy_pad`.  Either way the
  // walker doesn't see the init as written.  Fix the attribute directly:
  // for RCore the last `real` arg before workspace is the DPS output.
  if (auto ni = func->getAttrOfType<IntegerAttr>("cann.num_inputs")) {
    int64_t cur = ni.getInt();
    int64_t outIdx = (int64_t)write->outputArg.getArgNumber();
    if (cur > outIdx) {
      func->setAttr("cann.num_inputs",
                    IntegerAttr::get(IntegerType::get(ctx, 32),
                                      static_cast<int32_t>(outIdx)));
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "[rcore-combine] " << func.getName()
                          << ": transformed\n");
}

struct AscendCRCoreCombinePass
    : public ::impl::AscendCRCoreCombinePassBase<AscendCRCoreCombinePass> {
  using AscendCRCoreCombinePassBase::AscendCRCoreCombinePassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> funcs;
    module.walk([&](func::FuncOp f) { funcs.push_back(f); });
    (void)module;
    for (func::FuncOp f : funcs)
      transformOneFunc(f);
  }
};

} // namespace

std::unique_ptr<Pass> createAscendCRCoreCombinePass() {
  return std::make_unique<AscendCRCoreCombinePass>();
}

} // namespace mlir::afir
