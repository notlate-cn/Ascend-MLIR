//===- AscendCParallelizePass.cpp - Replace parallel scf.for with get_block_idx ===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// Converts annotated outermost scf.for tile loops into multi-core AiCore
// dispatch:
//
//   scf.for %i = 0 to %M step %TB_M {
//     <body>
//   }
//   {ascendc.parallel = true}
//
// becomes:
//
//   %block_idx = ascendc.get_block_idx : index
//   %i         = arith.muli %block_idx, %TB_M
//   %in_bound  = arith.cmpi ult %i, %M
//   scf.if %in_bound { <body> }
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Backend/Codegen/AscendCParallelizePass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

#define GEN_PASS_DECL_ASCENDCPARALLELIZEPASS
#define GEN_PASS_DEF_ASCENDCPARALLELIZEPASS
#include "Conversion/Ascend/Passes.h.inc"

#define DEBUG_TYPE "ascendc-parallelize"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir::afir {

//===----------------------------------------------------------------------===//
// Helper: find top-level parallel scf.for loops in a func
//===----------------------------------------------------------------------===//

/// Collect all top-level scf.for loops directly in the function entry block
/// (i.e. not nested inside another loop or region op).  These are the TB-level
/// loops that each need to be parallelized independently.
static SmallVector<scf::ForOp> findTopLevelParallelFors(func::FuncOp func) {
  SmallVector<scf::ForOp> result;
  Block &entry = func.getBody().front();
  for (Operation &op : entry.without_terminator()) {
    if (auto forOp = dyn_cast<scf::ForOp>(&op);
        forOp && forOp->hasAttr("ascendc.parallel"))
      result.push_back(forOp);
  }
  return result;
}


//===----------------------------------------------------------------------===//
// Parallelization transformation
//===----------------------------------------------------------------------===//

/// Replace the outermost scf.for loop with get_block_idx dispatch.
/// The inner loops (Tb-level tiles) are kept intact inside an scf.if guard.
///
/// Transforms:
///   scf.for %i = 0 to %UB step %STEP { <body> }
/// into:
///   %block_idx = ascendc.get_block_idx
///   %iVal      = arith.muli %block_idx, %STEP
///   %inBound   = arith.cmpi ult, %iVal, %UB
///   scf.if %inBound { <body with %i replaced by %iVal> }
static LogicalResult parallelizeOneLoop(scf::ForOp outerFor,
                                        OpBuilder &builder) {
  if (!outerFor)
    return failure();

  // Only transform if the loop starts at 0 (generated tiling always does).
  auto isConstZero = [](Value v) {
    if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
      return cst.value() == 0;
    return false;
  };
  if (!isConstZero(outerFor.getLowerBound()))
    return failure();

  Value ubOuter = outerFor.getUpperBound(); // e.g. %M or %dim
  Value stepOuter = outerFor.getStep();     // e.g. TB_M or TB

  Location loc = outerFor.getLoc();
  builder.setInsertionPoint(outerFor);
  IndexType idxTy = builder.getIndexType();

  // AiCore block index → outer loop induction variable value.
  Value blockIdx = builder.create<ascendc::GetBlockIdxOp>(loc, idxTy);
  Value iVal = builder.create<arith::MulIOp>(loc, blockIdx, stepOuter);

  // Guard: i < UB (handles partial last block).
  Value inBound =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, iVal, ubOuter);

  // Replace the outer loop IV with iVal before moving ops.
  outerFor.getInductionVar().replaceAllUsesWith(iVal);

  // Create scf.if to guard the body.
  auto ifOp = builder.create<scf::IfOp>(loc, inBound, /*withElseRegion=*/false);
  Block *thenBlock = ifOp.thenBlock();

  // Move all ops from the outer loop body (including any nested inner loops)
  // into the then block.
  SmallVector<Operation *> bodyOps;
  for (auto &op : outerFor.getBody()->without_terminator())
    bodyOps.push_back(&op);
  for (Operation *op : bodyOps)
    op->moveBefore(thenBlock->getTerminator());

  // Erase the now-empty outer loop.
  outerFor.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct AscendCParallelizePass
    : public ::impl::AscendCParallelizePassBase<AscendCParallelizePass> {
  using AscendCParallelizePassBase::AscendCParallelizePassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getContext());
    // Collect ALL top-level scf.for loops before modifying them (they will be
    // erased one by one, so collecting first avoids iterator invalidation).
    SmallVector<scf::ForOp> topFors = findTopLevelParallelFors(func);
    if (topFors.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[ascendc-parallelize] No annotated outer loops found to "
                    "parallelize\n");
      return;
    }
    for (scf::ForOp forOp : topFors) {
      if (failed(parallelizeOneLoop(forOp, builder))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[ascendc-parallelize] Skipping loop (lb != 0)\n");
      }
    }
  }
};

std::unique_ptr<Pass> createAscendCParallelizePass() {
  return std::make_unique<AscendCParallelizePass>();
}

} // namespace mlir::afir
