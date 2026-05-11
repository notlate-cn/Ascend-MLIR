//===- MemoryRealizationDriver.h - Ascend memory realization --*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MEMORYREALIZATIONDRIVER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MEMORYREALIZATIONDRIVER_H

#include "Conversion/Ascend/Realize/RealizeTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::afir::ascend::realize {

struct Phase5BridgeMaterializationCounts {
  unsigned materializedAllocCount = 0;
  unsigned materializedCopyCount = 0;
};

class MemoryRealizationDriver {
public:
  FailureOr<MemoryRealizationPlan>
  materialize(const PlacementPlan &placement,
              const StaticMemoryPlan &staticMemory,
              const MovementPlan &movement) const;

  FailureOr<llvm::StringMap<unsigned>>
  annotateMemorySpaces(ModuleOp module) const;
  FailureOr<llvm::StringMap<Phase5BridgeMaterializationCounts>>
  materializePhase5Bridge(ModuleOp module) const;
  void markMemorySpaceMaterialized(
      MemoryRealizationPlan &plan, unsigned annotationCount,
      const Phase5BridgeMaterializationCounts &materializationCounts) const;
};

} // namespace mlir::afir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MEMORYREALIZATIONDRIVER_H
