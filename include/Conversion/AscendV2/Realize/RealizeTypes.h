//===- RealizeTypes.h - Ascend V2 realize data model -----------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_REALIZETYPES_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_REALIZETYPES_H

#include "Conversion/AscendV2/Common/Attributes.h"

#include <string>

namespace mlir::afir::ascend::v2::realize {

using ::mlir::afir::ascend::v2::kKernelAttr;
using ::mlir::afir::ascend::v2::kScheduleDecisionIdAttr;
using ::mlir::afir::ascend::v2::kStructuredLoweringAttr;

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
  unsigned bufferValueCount = 0;
};

struct PlacementPlan {
  std::string kernelId;
  unsigned selectedPlaceCount = 0;
};

struct StaticMemoryPlan {
  std::string kernelId;
  unsigned workspaceSlotCount = 0;
};

struct MovementPlan {
  std::string kernelId;
  unsigned movementCount = 0;
};

struct MemoryRealizationPlan {
  std::string kernelId;
  bool frozen = false;
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

} // namespace mlir::afir::ascend::v2::realize

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_REALIZETYPES_H
