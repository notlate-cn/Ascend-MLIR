//===- MovementPlanner.cpp - Ascend movement plan ------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/MovementPlanner.h"

namespace mlir::afir::ascend::realize {

FailureOr<MovementPlan>
MovementPlanner::build(const PlacementPlan &placement,
                       const StaticMemoryPlan &staticMemory) const {
  if (staticMemory.kernelId != placement.kernelId)
    return failure();
  if (staticMemory.trackedPlaceCount != placement.selectedPlaceCount)
    return failure();
  if (staticMemory.mode != "empty_workspace")
    return failure();
  if (staticMemory.workspaceSlotCount != 0)
    return failure();
  if (staticMemory.peakUsageKnown)
    return failure();

  MovementPlan plan;
  plan.kernelId = placement.kernelId;
  plan.mode = "gm_noop";
  plan.crossPlaceEdgeCount = 0;
  plan.movementCount = 0;
  plan.redundantMovementCount = 0;
  (void)staticMemory;
  return plan;
}

} // namespace mlir::afir::ascend::realize
