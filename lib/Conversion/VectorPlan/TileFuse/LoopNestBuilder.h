#pragma once
#include "Conversion/VectorPlan/TilePlan.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::afir {

struct LoopNestResult {
  llvm::DenseMap<int, mlir::Value>     loopIVs;
  /// Outer-only IVs (before adding the inner tile offset).  For axes with
  /// both an outer and an inner loop this holds the outer IV; for axes with
  /// only one loop level it is the same as loopIVs.
  llvm::DenseMap<int, mlir::Value>     outerLoopIVs;
  llvm::SmallVector<mlir::scf::ForOp>  allForOps;
  llvm::SmallVector<mlir::scf::ForOp>  bcastForOps;
  mlir::Block                         *innermostBody = nullptr;
  llvm::SmallVector<mlir::Value>       iterArgs;

  /// Tail-peel info for the innermost inner-tile axis.  When `hasTail` is
  /// true, the innermost scf.for's ub has already been rewritten to
  /// `mainInnerUb` (= floor(remaining / step) * step), and the GroupEmitter
  /// must, after closing that for, emit an `scf.if (mainInnerUb < remaining)`
  /// whose then-block re-emits the body with composed IV = outerOfTailIV +
  /// mainInnerUb and slice-size override `{innerTileAxisIdx: tailSize}`
  /// (tailSize is computed inside the if's then-block).
  ///
  /// Constraint: requires `XBLOCK_SUB | XBLOCK` so the tail block only fires
  /// on the tail core (last block when `extent % XBLOCK != 0`).
  bool                                 hasTail = false;
  int                                  innerTileAxisIdx = -1;
  mlir::Value                          remaining;       // min(XBLOCK, extent-outer_iv)
  mlir::Value                          mainInnerUb;     // floor(remaining/T)*T
  mlir::Value                          outerOfTailIV;   // c0 if axis has no Outer
};

LoopNestResult buildLoopNest(mlir::OpBuilder &builder,
                              mlir::Location loc,
                              const mlir::vector_plan::TilePlan &plan,
                              mlir::ValueRange initTensors);

} // namespace mlir::afir
