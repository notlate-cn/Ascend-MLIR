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
};

LoopNestResult buildLoopNest(mlir::OpBuilder &builder,
                              mlir::Location loc,
                              const mlir::vector_plan::TilePlan &plan,
                              mlir::ValueRange initTensors);

} // namespace mlir::afir
