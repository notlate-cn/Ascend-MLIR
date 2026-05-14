//===- StaticMemoryPlanner.cpp - Ascend static memory plan ---------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "StaticMemoryPlanner.h"

namespace mlir::afir::ascend::realize {

FailureOr<StaticMemoryPlan>
StaticMemoryPlanner::build(const PlacementPlan &placement) const {
  StaticMemoryPlan plan;
  plan.kernelId = placement.kernelId;
  plan.trackedPlaceCount = placement.selectedPlaceCount;
  if (placement.onChipPlaceCount == 0) {
    plan.mode = "empty_workspace";
    return plan;
  }

  plan.mode = "workspace_layout";
  plan.localBufferCount = placement.onChipPlaceCount;
  plan.liveIntervalCount = placement.onChipPlaceCount;
  plan.workspaceSlotCount = placement.onChipPlaceCount;
  plan.peakUsageKnown = true;
  plan.peakUsageUnitCount = plan.workspaceSlotCount;
  plan.capacityCheckDeferred = true;
  return plan;
}

} // namespace mlir::afir::ascend::realize
