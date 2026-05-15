//===- MovementPlanner.cpp - Ascend movement plan ------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "MovementPlanner.h"

#include "Target/Ascend/TargetMemoryModel.h"

namespace mlir::afir::ascend::realize {
namespace {

static void populateMovementSteps(const StaticMemoryPlan &staticMemory,
                                  MovementPlan &plan) {
  for (const StaticMemoryWorkspaceSlot &slot : staticMemory.workspaceSlots) {
    MovementStep step;
    step.stepId = plan.movementSteps.size();
    step.valueId = slot.valueId;
    step.slotId = slot.slotId;
    step.srcPlace = MemoryPlace::GM;
    step.dstPlace = slot.place;
    step.pathSelected = false;
    step.pathSelectionDeferred = true;
    step.staticByteSizeKnown = slot.staticByteSizeKnown;
    step.byteSize = slot.byteSize;
    plan.movementSteps.push_back(step);
  }
}

static void updateMovementCountsFromSteps(MovementPlan &plan) {
  if (plan.movementSteps.empty())
    return;

  plan.movementDemandCount = plan.movementSteps.size();
  plan.crossPlaceEdgeCount = plan.movementDemandCount;
  plan.selectedPathCount = 0;
  plan.pathSelectionDeferredCount = 0;
  for (const MovementStep &step : plan.movementSteps) {
    if (step.pathSelected)
      ++plan.selectedPathCount;
    if (step.pathSelectionDeferred)
      ++plan.pathSelectionDeferredCount;
  }
}

} // namespace

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
    populateMovementSteps(staticMemory, plan);
    if (plan.movementSteps.empty()) {
      plan.crossPlaceEdgeCount = placement.onChipPlaceCount;
      plan.movementDemandCount = placement.onChipPlaceCount;
      plan.pathSelectionDeferredCount = plan.movementDemandCount;
    } else {
      updateMovementCountsFromSteps(plan);
    }
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

FailureOr<MovementPlan> MovementPlanner::build(
    const PlacementPlan &placement, const StaticMemoryPlan &staticMemory,
    const ::mlir::ascend::TargetMemoryModel &memoryModel) const {
  FailureOr<MovementPlan> plan = build(placement, staticMemory);
  if (failed(plan))
    return failure();
  if (plan->mode != "movement_planning" || plan->movementSteps.empty())
    return plan;

  for (MovementStep &step : plan->movementSteps) {
    SmallVector<::mlir::ascend::PathEdge> directPaths =
        memoryModel.findDirectPaths(step.srcPlace, step.dstPlace);
    if (directPaths.empty())
      continue;

    step.pathSelected = true;
    step.pathSelectionDeferred = false;
    step.pathVariant = directPaths.front().pathVariant;
  }
  updateMovementCountsFromSteps(*plan);
  return plan;
}

} // namespace mlir::afir::ascend::realize
