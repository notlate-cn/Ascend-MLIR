//===- PlacementPlanner.cpp - Ascend realize placement plan --------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/PlacementPlanner.h"

namespace mlir::afir::ascend::realize {

FailureOr<PlacementPlan>
PlacementPlanner::build(const BufferizedKernelIR &bufferizedIR) const {
  PlacementPlan plan;
  plan.kernelId = bufferizedIR.kernelId;
  plan.mode = "gm_default";
  plan.selectedPlaceCount = bufferizedIR.bufferValueCount;
  plan.gmPlaceCount = bufferizedIR.bufferValueCount;
  plan.onChipPlaceCount = 0;
  plan.deferredLocalPlaceCount = bufferizedIR.temporaryValueCount;
  return plan;
}

} // namespace mlir::afir::ascend::realize
