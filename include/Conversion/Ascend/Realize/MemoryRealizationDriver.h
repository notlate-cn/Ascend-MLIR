//===- MemoryRealizationDriver.h - Ascend memory realization --*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MEMORYREALIZATIONDRIVER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MEMORYREALIZATIONDRIVER_H

#include "Conversion/Ascend/Realize/RealizeTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::afir::ascend::realize {

class MemoryRealizationDriver {
public:
  FailureOr<MemoryRealizationPlan>
  materialize(const PlacementPlan &placement,
              const StaticMemoryPlan &staticMemory,
              const MovementPlan &movement) const;
};

} // namespace mlir::afir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MEMORYREALIZATIONDRIVER_H
