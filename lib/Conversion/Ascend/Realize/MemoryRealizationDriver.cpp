//===- MemoryRealizationDriver.cpp - Ascend memory realization -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/MemoryRealizationDriver.h"

namespace mlir::afir::ascend::realize {

FailureOr<MemoryRealizationPlan>
MemoryRealizationDriver::materialize(const PlacementPlan &placement,
                                     const StaticMemoryPlan &staticMemory,
                                     const MovementPlan &movement) const {
  if (staticMemory.kernelId != placement.kernelId ||
      movement.kernelId != placement.kernelId)
    return failure();

  MemoryRealizationPlan plan;
  plan.kernelId = placement.kernelId;
  plan.mode = "read_only_freeze";
  plan.frozen = true;
  plan.verificationScope = "plan_identity_only";
  plan.planIdsVerified = true;
  plan.materializedAllocCount = 0;
  plan.materializedCopyCount = 0;
  return plan;
}

} // namespace mlir::afir::ascend::realize
