//===- StaticMemoryPlanner.cpp - Ascend static memory plan ---------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "StaticMemoryPlanner.h"

#include "Target/Ascend/TargetMemoryModel.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace mlir::afir::ascend::realize {
namespace {

constexpr MemoryPlace kVectorTemporaryPlace = MemoryPlace::VECIN;

static void populateVectorTemporarySlots(const BufferizedKernelIR &bufferizedIR,
                                         StaticMemoryPlan &plan) {
  struct PhysicalSlot {
    uint64_t offset = 0;
    uint64_t capacity = 0;
    unsigned liveUntil = 0;
  };

  SmallVector<PhysicalSlot, 4> physicalSlots;
  uint64_t nextOffset = 0;
  uint64_t logicalLocalBytes = 0;
  uint64_t peakWorkspaceBytes = 0;
  unsigned peakWorkspaceUnits = 0;
  bool allWorkspaceExprsKnown = true;
  SmallVector<std::string, 4> workspaceExprTerms;

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

    if (fact.staticByteSizeKnown) {
      logicalLocalBytes += fact.byteSize;
      if (fact.byteSize != 0)
        workspaceExprTerms.push_back(std::to_string(fact.byteSize));
    } else {
      if (!fact.byteSizeExprKnown)
        allWorkspaceExprsKnown = false;
      else
        workspaceExprTerms.push_back(fact.byteSizeExpr);
    }

    unsigned physicalSlotIndex = physicalSlots.size();
    if (fact.staticByteSizeKnown) {
      for (auto [index, physicalSlot] : llvm::enumerate(physicalSlots)) {
        if (physicalSlot.liveUntil <= interval.start &&
            physicalSlot.capacity >= fact.byteSize) {
          physicalSlotIndex = static_cast<unsigned>(index);
          break;
        }
      }
    }

    uint64_t slotOffset = nextOffset;
    if (physicalSlotIndex == physicalSlots.size()) {
      PhysicalSlot physicalSlot;
      physicalSlot.offset = nextOffset;
      physicalSlot.capacity = fact.staticByteSizeKnown ? fact.byteSize : 0;
      physicalSlot.liveUntil = interval.end;
      physicalSlots.push_back(physicalSlot);
      if (fact.staticByteSizeKnown)
        nextOffset += fact.byteSize;
    } else {
      PhysicalSlot &physicalSlot = physicalSlots[physicalSlotIndex];
      slotOffset = physicalSlot.offset;
      physicalSlot.liveUntil = interval.end;
    }

    StaticMemoryWorkspaceSlot slot;
    slot.slotId = plan.workspaceSlots.size();
    slot.valueId = fact.valueId;
    slot.offset = slotOffset;
    slot.place = kVectorTemporaryPlace;
    slot.staticByteSizeKnown = fact.staticByteSizeKnown;
    slot.byteSize = fact.byteSize;
    slot.byteSizeExprKnown = fact.byteSizeExprKnown;
    slot.byteSizeExpr = fact.byteSizeExpr;
    plan.workspaceSlots.push_back(slot);

    unsigned liveUnits = 0;
    uint64_t liveBytes = 0;
    for (const PhysicalSlot &physicalSlot : physicalSlots) {
      if (physicalSlot.liveUntil <= interval.start)
        continue;
      ++liveUnits;
      liveBytes += physicalSlot.capacity;
    }
    peakWorkspaceUnits = std::max(peakWorkspaceUnits, liveUnits);
    peakWorkspaceBytes = std::max(peakWorkspaceBytes, liveBytes);
  }

  if (!plan.workspaceSlots.empty()) {
    plan.peakUsageUnitCount = peakWorkspaceUnits;
    if (bufferizedIR.staticByteSizeKnown) {
      plan.localBufferByteCount = logicalLocalBytes;
      plan.workspaceByteCount = nextOffset;
      plan.peakUsageByteCount = peakWorkspaceBytes;
    }
    if (!bufferizedIR.staticByteSizeKnown && allWorkspaceExprsKnown &&
        !workspaceExprTerms.empty()) {
      std::string expr;
      llvm::raw_string_ostream os(expr);
      llvm::interleave(workspaceExprTerms, os,
                       [&os](const std::string &term) { os << term; },
                       " + ");
      plan.workspaceSizeExprKnown = true;
      plan.workspaceSizeExpr = os.str();
    }
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
  if (plan.peakUsageUnitCount == 0)
    plan.peakUsageUnitCount = plan.workspaceSlotCount;
  if (bufferizedIR.staticByteSizeKnown) {
    plan.peakUsageBytesKnown = true;
    if (plan.localBufferByteCount == 0)
      plan.localBufferByteCount = bufferizedIR.vectorTemporaryByteCount;
    if (plan.workspaceByteCount == 0)
      plan.workspaceByteCount = bufferizedIR.vectorTemporaryByteCount;
    if (plan.peakUsageByteCount == 0)
      plan.peakUsageByteCount = plan.workspaceByteCount;
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
