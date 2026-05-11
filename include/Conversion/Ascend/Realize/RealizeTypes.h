//===- RealizeTypes.h - Ascend realize data model -----------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_REALIZETYPES_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_REALIZETYPES_H

#include "Conversion/Ascend/Common/Attributes.h"

#include <string>

namespace mlir::afir::ascend::realize {

using ::mlir::afir::ascend::kKernelAttr;
using ::mlir::afir::ascend::kOpRoleAttr;
using ::mlir::afir::ascend::kScheduleDecisionIdAttr;
using ::mlir::afir::ascend::kStructuredLoweringAttr;

// Realize-level placement names used by materialization plans. They are
// intentionally distinct from the target profile hardware memory hierarchy.
enum class MemoryPlace { GM, VECIN, VECCALC, VECOUT, A1, B1, A2, B2, CO1 };

struct RealizeKernelView {
  std::string kernelId;
  std::string decisionId;
  std::string structuredLowering;
  unsigned scheduledOps = 0;
};

struct BufferizedKernelIR {
  std::string kernelId;
  std::string mode = "gm_only";
  // Maintained by BufferizationDriver: bufferValueCount equals input + output
  // + temporary, and vectorTemporaryValueCount is a subset of temporary.
  unsigned bufferValueCount = 0;
  unsigned inputValueCount = 0;
  unsigned outputValueCount = 0;
  unsigned temporaryValueCount = 0;
  unsigned vectorTemporaryValueCount = 0;
};

struct PlacementPlan {
  std::string kernelId;
  std::string mode = "none";
  unsigned selectedPlaceCount = 0;
  unsigned gmPlaceCount = 0;
  unsigned onChipPlaceCount = 0;
  unsigned deferredLocalPlaceCount = 0;
};

struct StaticMemoryPlan {
  std::string kernelId;
  std::string mode = "none";
  unsigned trackedPlaceCount = 0;
  unsigned workspaceSlotCount = 0;
  bool peakUsageKnown = false;
};

struct MovementPlan {
  std::string kernelId;
  std::string mode = "none";
  unsigned crossPlaceEdgeCount = 0;
  unsigned movementCount = 0;
  unsigned redundantMovementCount = 0;
};

struct MemoryRealizationPlan {
  std::string kernelId;
  std::string mode = "none";
  bool frozen = false;
  std::string verificationScope = "none";
  bool planIdsVerified = false;
  unsigned materializedAllocCount = 0;
  unsigned materializedCopyCount = 0;
};

struct RealizePlanBundle {
  RealizeKernelView kernel;
  BufferizedKernelIR bufferizedIR;
  PlacementPlan placement;
  StaticMemoryPlan staticMemory;
  MovementPlan movement;
  MemoryRealizationPlan realization;
};

} // namespace mlir::afir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_REALIZETYPES_H
