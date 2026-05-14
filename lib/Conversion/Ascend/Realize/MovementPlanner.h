//===- MovementPlanner.h - Ascend movement plan -------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MOVEMENTPLANNER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MOVEMENTPLANNER_H

#include "RealizeTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::afir::ascend::realize {

class MovementPlanner {
public:
  FailureOr<MovementPlan> build(const PlacementPlan &placement,
                                const StaticMemoryPlan &staticMemory) const;
};

} // namespace mlir::afir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MOVEMENTPLANNER_H
