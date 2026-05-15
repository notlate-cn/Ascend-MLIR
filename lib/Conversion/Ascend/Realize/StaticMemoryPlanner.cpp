//===- StaticMemoryPlanner.cpp - Ascend static memory plan ---------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "StaticMemoryPlanner.h"

namespace mlir::afir::ascend::realize {

FailureOr<StaticMemoryPlan>
StaticMemoryPlanner::build(const PlacementPlan &placement) const {
  BufferizedKernelIR bufferizedIR;
  bufferizedIR.kernelId = placement.kernelId;
  return build(placement, bufferizedIR);
}

FailureOr<StaticMemoryPlan>
StaticMemoryPlanner::build(const PlacementPlan &placement,
                           const BufferizedKernelIR &bufferizedIR) const {
  if (!bufferizedIR.kernelId.empty() &&
      bufferizedIR.kernelId != placement.kernelId)
    return failure();

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
  if (bufferizedIR.staticByteSizeKnown) {
    plan.peakUsageBytesKnown = true;
    plan.localBufferByteCount = bufferizedIR.vectorTemporaryByteCount;
    plan.workspaceByteCount = bufferizedIR.vectorTemporaryByteCount;
    plan.peakUsageByteCount = bufferizedIR.vectorTemporaryByteCount;
  }
  plan.capacityCheckDeferred = true;
  return plan;
}

} // namespace mlir::afir::ascend::realize
