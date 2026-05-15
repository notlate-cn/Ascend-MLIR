//===- MemoryRealizationDriver.h - Ascend memory realization --*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MEMORYREALIZATIONDRIVER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MEMORYREALIZATIONDRIVER_H

#include "RealizeTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::afir::ascend::realize {

struct Phase5BridgeMaterializationCounts {
  unsigned materializedAllocCount = 0;
  unsigned materializedCopyCount = 0;
};

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
  FailureOr<llvm::StringMap<Phase5BridgeMaterializationCounts>>
  materializeMovementSteps(ModuleOp module,
                           llvm::ArrayRef<RealizePlanBundle> bundles) const;
  FailureOr<llvm::StringMap<Phase5BridgeMaterializationCounts>>
  materializePhase5Bridge(ModuleOp module) const;
  void markMemorySpaceMaterialized(
      MemoryRealizationPlan &plan, unsigned annotationCount,
      const Phase5BridgeMaterializationCounts &materializationCounts) const;
};

} // namespace mlir::afir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MEMORYREALIZATIONDRIVER_H
