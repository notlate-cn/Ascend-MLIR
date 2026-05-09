//===- StaticMemoryPlanner.cpp - Ascend static memory plan ---------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/StaticMemoryPlanner.h"

namespace mlir::afir::ascend::realize {

FailureOr<StaticMemoryPlan>
StaticMemoryPlanner::build(const PlacementPlan &placement) const {
  StaticMemoryPlan plan;
  plan.kernelId = placement.kernelId;
  plan.mode = "empty_workspace";
  plan.trackedPlaceCount = placement.selectedPlaceCount;
  plan.workspaceSlotCount = 0;
  plan.peakUsageKnown = false;
  return plan;
}

} // namespace mlir::afir::ascend::realize
