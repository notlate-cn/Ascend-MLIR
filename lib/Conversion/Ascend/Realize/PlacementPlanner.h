//===- PlacementPlanner.h - Ascend realize placement plan -----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_PLACEMENTPLANNER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_PLACEMENTPLANNER_H

#include "RealizeTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::ascend {
class TargetMemoryModel;
} // namespace mlir::ascend

namespace mlir::ascend::realize {

class PlacementPlanner {
public:
  FailureOr<PlacementPlan> build(const BufferizedKernelIR &bufferizedIR) const;
  FailureOr<PlacementPlan>
  build(const BufferizedKernelIR &bufferizedIR,
        const ::mlir::ascend::TargetMemoryModel &memoryModel) const;
};

} // namespace mlir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_PLACEMENTPLANNER_H
