//===- RealizeTypes.h - Ascend V2 realize data model -----------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_REALIZETYPES_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_REALIZETYPES_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::afir::ascend::v2::realize {

inline constexpr llvm::StringLiteral kScheduleDecisionIdAttr =
    "ascend.v2.schedule.decision_id";
inline constexpr llvm::StringLiteral kStructuredLoweringAttr =
    "ascend.v2.schedule.structured_lowering";
inline constexpr llvm::StringLiteral kKernelAttr = "ascend.v2.kernel";

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
