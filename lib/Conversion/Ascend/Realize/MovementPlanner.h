//===- MovementPlanner.h - Ascend movement plan -------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MOVEMENTPLANNER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MOVEMENTPLANNER_H

#include "RealizeTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::ascend {
class TargetMemoryModel;
} // namespace mlir::ascend

namespace mlir::ascend::realize {

class MovementPlanner {
public:
  FailureOr<MovementPlan> build(const PlacementPlan &placement,
                                const StaticMemoryPlan &staticMemory) const;
  FailureOr<MovementPlan>
  build(const PlacementPlan &placement, const StaticMemoryPlan &staticMemory,
        const ::mlir::ascend::TargetMemoryModel &memoryModel) const;
};

} // namespace mlir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MOVEMENTPLANNER_H
