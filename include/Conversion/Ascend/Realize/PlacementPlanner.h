//===- PlacementPlanner.h - Ascend realize placement plan -----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_PLACEMENTPLANNER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_PLACEMENTPLANNER_H

#include "Conversion/Ascend/Realize/RealizeTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::ascend {
class TargetMemoryModel;
} // namespace mlir::ascend

namespace mlir::afir::ascend::realize {

class PlacementPlanner {
public:
  FailureOr<PlacementPlan> build(const BufferizedKernelIR &bufferizedIR) const;
  FailureOr<PlacementPlan>
  build(const BufferizedKernelIR &bufferizedIR,
        const ::mlir::ascend::TargetMemoryModel &memoryModel) const;
};

} // namespace mlir::afir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_PLACEMENTPLANNER_H
