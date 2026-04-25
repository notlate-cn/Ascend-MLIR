#pragma once

#include "Conversion/VectorPlan/GroupInfo.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::afir {

enum class FusionKind { None, Vertical, Horizontal };

struct FusionGroup {
  int32_t                             id;
  llvm::SmallVector<linalg::LinalgOp> members;      // unordered during fusion
  llvm::DenseSet<Value>               boundaryIn;   // values defined outside group
  mlir::vector_plan::GroupInfo::Kind  kind;
  llvm::SmallVector<mlir::vector_plan::AxisInfo> canonicalAxes;

  bool containsOp(Operation *op) const {
    return llvm::any_of(members, [op](linalg::LinalgOp m) {
      return m.getOperation() == op;
    });
  }
};

struct CanFuseOptions {
  int32_t maxReduceEpilogueOps     = 3;
  int32_t maxHorizontalExtraInputs = 4;
  bool    enableReductionSplit     = false;
};

} // namespace mlir::afir
