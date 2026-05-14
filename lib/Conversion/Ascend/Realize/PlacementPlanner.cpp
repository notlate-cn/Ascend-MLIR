//===- PlacementPlanner.cpp - Ascend realize placement plan --------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "PlacementPlanner.h"

#include "Target/Ascend/TargetMemoryModel.h"

#include <algorithm>

namespace mlir::afir::ascend::realize {
namespace {

bool supportsVecCalcPlacement(
    const ::mlir::ascend::TargetMemoryModel &memoryModel) {
  constexpr ::mlir::ascend::MemoryPlace place =
      ::mlir::ascend::MemoryPlace::VECCALC;
  if (!memoryModel.supportsMemoryPlace(place))
    return false;

  FailureOr<::mlir::ascend::CapacityRule> capacity =
      memoryModel.getCapacity(place);
  if (failed(capacity) || capacity->availableCapacityBytes <= 0)
    return false;

  return memoryModel.isPlaceVisibleTo(place,
                                      ::mlir::ascend::ExecutionUnit::Vector);
}

PlacementPlan buildGmDefaultPlan(const BufferizedKernelIR &bufferizedIR) {
  PlacementPlan plan;
  plan.kernelId = bufferizedIR.kernelId;
  plan.mode = "gm_default";
  plan.selectedPlaceCount = bufferizedIR.bufferValueCount;
  plan.gmPlaceCount = bufferizedIR.bufferValueCount;
  plan.onChipPlaceCount = 0;
  plan.deferredLocalPlaceCount = bufferizedIR.temporaryValueCount;
  return plan;
}

} // namespace

FailureOr<PlacementPlan>
PlacementPlanner::build(const BufferizedKernelIR &bufferizedIR) const {
  return buildGmDefaultPlan(bufferizedIR);
}

FailureOr<PlacementPlan> PlacementPlanner::build(
    const BufferizedKernelIR &bufferizedIR,
    const ::mlir::ascend::TargetMemoryModel &memoryModel) const {
  if (!supportsVecCalcPlacement(memoryModel))
    return buildGmDefaultPlan(bufferizedIR);

  PlacementPlan plan;
  plan.kernelId = bufferizedIR.kernelId;
  plan.mode = "target_aware";
  plan.selectedPlaceCount = bufferizedIR.bufferValueCount;
  unsigned vectorTemporaryCount =
      std::min(bufferizedIR.vectorTemporaryValueCount,
               bufferizedIR.temporaryValueCount);
  plan.onChipPlaceCount = vectorTemporaryCount;
  unsigned nonVectorTemporaryCount =
      bufferizedIR.temporaryValueCount - vectorTemporaryCount;
  plan.gmPlaceCount =
      bufferizedIR.inputValueCount + bufferizedIR.outputValueCount +
      nonVectorTemporaryCount;
  plan.deferredLocalPlaceCount = nonVectorTemporaryCount;
  return plan;
}

} // namespace mlir::afir::ascend::realize
