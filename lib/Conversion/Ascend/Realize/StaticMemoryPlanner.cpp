//===- StaticMemoryPlanner.cpp - Ascend static memory plan ---------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "StaticMemoryPlanner.h"

#include "Target/Ascend/TargetMemoryModel.h"

namespace mlir::afir::ascend::realize {
namespace {

constexpr MemoryPlace kVectorTemporaryPlace = MemoryPlace::VECIN;

static void populateVectorTemporarySlots(const BufferizedKernelIR &bufferizedIR,
                                         StaticMemoryPlan &plan) {
  uint64_t nextOffset = 0;
  for (const BufferizedValueFact &fact : bufferizedIR.valueFacts) {
    if (!fact.isVectorTemporary)
      continue;

    StaticMemoryLiveInterval interval;
    interval.valueId = fact.valueId;
    interval.start = plan.liveIntervals.size();
    interval.end = interval.start + 1;
    interval.place = kVectorTemporaryPlace;
    interval.staticByteSizeKnown = fact.staticByteSizeKnown;
    interval.byteSize = fact.byteSize;
    plan.liveIntervals.push_back(interval);

    StaticMemoryWorkspaceSlot slot;
    slot.slotId = plan.workspaceSlots.size();
    slot.valueId = fact.valueId;
    slot.offset = nextOffset;
    slot.place = kVectorTemporaryPlace;
    slot.staticByteSizeKnown = fact.staticByteSizeKnown;
    slot.byteSize = fact.byteSize;
    plan.workspaceSlots.push_back(slot);

    if (fact.staticByteSizeKnown)
      nextOffset += fact.byteSize;
  }
}

} // namespace

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
  populateVectorTemporarySlots(bufferizedIR, plan);
  unsigned plannedSlotCount = plan.workspaceSlots.empty()
                                  ? placement.onChipPlaceCount
                                  : plan.workspaceSlots.size();
  plan.localBufferCount = plannedSlotCount;
  plan.liveIntervalCount = plannedSlotCount;
  plan.workspaceSlotCount = plannedSlotCount;
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

FailureOr<StaticMemoryPlan> StaticMemoryPlanner::build(
    const PlacementPlan &placement, const BufferizedKernelIR &bufferizedIR,
    const ::mlir::ascend::TargetMemoryModel &memoryModel) const {
  FailureOr<StaticMemoryPlan> plan = build(placement, bufferizedIR);
  if (failed(plan))
    return failure();

  if (plan->mode != "workspace_layout" || !plan->peakUsageBytesKnown)
    return plan;

  FailureOr<::mlir::ascend::CapacityRule> capacity =
      memoryModel.getCapacity(kVectorTemporaryPlace);
  if (failed(capacity) || capacity->availableCapacityBytes < 0)
    return failure();

  if (plan->peakUsageByteCount >
      static_cast<uint64_t>(capacity->availableCapacityBytes))
    return failure();

  plan->capacityCheckDeferred = false;
  return plan;
}

} // namespace mlir::afir::ascend::realize
