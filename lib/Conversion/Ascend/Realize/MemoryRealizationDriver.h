//===- MemoryRealizationDriver.h - Ascend memory realization --*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MEMORYREALIZATIONDRIVER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MEMORYREALIZATIONDRIVER_H

#include "RealizeTypes.h"
#include "TranslateMemoryBridge.h"

namespace mlir::ascend::realize {

enum class MemoryRealizationMode {
  PlanOnly,
  MemorySpaceAnnotate,
};

class MemoryRealizationDriver {
public:
  FailureOr<MemoryRealizationPlan>
  materialize(const PlacementPlan &placement,
              const StaticMemoryPlan &staticMemory,
              const MovementPlan &movement) const;

  LogicalResult materialize(ModuleOp module,
                            MutableArrayRef<RealizePlanBundle> bundles,
                            MemoryRealizationMode mode) const;

  FailureOr<llvm::StringMap<unsigned>>
  annotateMemorySpaces(ModuleOp module) const;
  FailureOr<llvm::StringMap<TranslateBridgeMaterializationCounts>>
  materializeMovementSteps(ModuleOp module,
                           MutableArrayRef<RealizePlanBundle> bundles) const;
  void markMemorySpaceMaterialized(
      MemoryRealizationPlan &plan, unsigned annotationCount,
      const TranslateBridgeMaterializationCounts &materializationCounts) const;
};

} // namespace mlir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MEMORYREALIZATIONDRIVER_H
