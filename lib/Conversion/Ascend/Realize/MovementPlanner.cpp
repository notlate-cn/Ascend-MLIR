//===- MovementPlanner.cpp - Ascend movement plan ------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "MovementPlanner.h"

namespace mlir::afir::ascend::realize {

FailureOr<MovementPlan>
MovementPlanner::build(const PlacementPlan &placement,
                       const StaticMemoryPlan &staticMemory) const {
  if (staticMemory.kernelId != placement.kernelId)
    return failure();
  if (staticMemory.trackedPlaceCount != placement.selectedPlaceCount)
    return failure();

  MovementPlan plan;
  plan.kernelId = placement.kernelId;
  if (placement.onChipPlaceCount > 0) {
    plan.mode = "movement_planning";
    plan.crossPlaceEdgeCount = placement.onChipPlaceCount;
    plan.movementDemandCount = placement.onChipPlaceCount;
    plan.pathSelectionDeferredCount = plan.movementDemandCount;
    plan.workspaceReuseCandidateCount = staticMemory.workspaceSlotCount;
    plan.materializationDeferred = true;
    return plan;
  }

  plan.mode = "gm_noop";
  plan.crossPlaceEdgeCount = 0;
  plan.movementCount = 0;
  plan.redundantMovementCount = 0;
  return plan;
}

} // namespace mlir::afir::ascend::realize
