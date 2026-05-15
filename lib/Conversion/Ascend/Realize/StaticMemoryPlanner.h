//===- StaticMemoryPlanner.h - Ascend static memory plan ------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_STATICMEMORYPLANNER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_STATICMEMORYPLANNER_H

#include "RealizeTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::ascend {
class TargetMemoryModel;
} // namespace mlir::ascend

namespace mlir::afir::ascend::realize {

class StaticMemoryPlanner {
public:
  FailureOr<StaticMemoryPlan> build(const PlacementPlan &placement) const;
  FailureOr<StaticMemoryPlan> build(const PlacementPlan &placement,
                                    const BufferizedKernelIR &bufferizedIR) const;
  FailureOr<StaticMemoryPlan>
  build(const PlacementPlan &placement, const BufferizedKernelIR &bufferizedIR,
        const ::mlir::ascend::TargetMemoryModel &memoryModel) const;
};

} // namespace mlir::afir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_STATICMEMORYPLANNER_H
