//===- RealizeTypes.h - Ascend realize data model -----------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_REALIZETYPES_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_REALIZETYPES_H

#include "Conversion/Ascend/Common/Attributes.h"
#include "Target/Ascend/TargetProfile.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace mlir::ascend::realize {

using ::mlir::ascend::kKernelAttr;
using ::mlir::ascend::kKernelizeOpRoleCube;
using ::mlir::ascend::kKernelizeOpRoleVector;
using ::mlir::ascend::kOpRoleAttr;
using ::mlir::ascend::kOpRolesAttr;
using ::mlir::ascend::kScheduleContractAttr;
using ::mlir::ascend::kScheduleDecisionIdAttr;

using MemoryPlace = ::mlir::ascend::MemoryPlace;

struct RealizeKernelView {
  std::string kernelId;
  std::string decisionId;
  std::string scheduleContract;
  unsigned scheduledOps = 0;
};

enum class BufferizedValueRole { Input, Temporary, Output };

struct BufferizedValueFact {
  unsigned valueId = 0;
  BufferizedValueRole role = BufferizedValueRole::Input;
  bool isVectorTemporary = false;
  bool staticByteSizeKnown = false;
  uint64_t byteSize = 0;
  bool byteSizeExprKnown = false;
  std::string byteSizeExpr;
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
  bool staticByteSizeKnown = false;
  uint64_t inputByteCount = 0;
  uint64_t outputByteCount = 0;
  uint64_t temporaryByteCount = 0;
  uint64_t vectorTemporaryByteCount = 0;
  llvm::SmallVector<BufferizedValueFact, 8> valueFacts;
};

struct PlacementPlan {
  std::string kernelId;
  std::string mode = "none";
  unsigned selectedPlaceCount = 0;
  unsigned gmPlaceCount = 0;
  unsigned onChipPlaceCount = 0;
  unsigned deferredLocalPlaceCount = 0;
};

struct StaticMemoryLiveInterval {
  unsigned valueId = 0;
  unsigned start = 0;
  unsigned end = 0;
  MemoryPlace place = MemoryPlace::VECCALC;
  bool staticByteSizeKnown = false;
  uint64_t byteSize = 0;
};

struct StaticMemoryWorkspaceSlot {
  unsigned slotId = 0;
  unsigned valueId = 0;
  uint64_t offset = 0;
  MemoryPlace place = MemoryPlace::VECCALC;
  bool staticByteSizeKnown = false;
  uint64_t byteSize = 0;
  bool byteSizeExprKnown = false;
  std::string byteSizeExpr;
};

struct StaticMemoryPlan {
  std::string kernelId;
  std::string mode = "none";
  unsigned trackedPlaceCount = 0;
  unsigned localBufferCount = 0;
  unsigned liveIntervalCount = 0;
  unsigned workspaceSlotCount = 0;
  bool peakUsageKnown = false;
  unsigned peakUsageUnitCount = 0;
  bool peakUsageBytesKnown = false;
  uint64_t localBufferByteCount = 0;
  uint64_t workspaceByteCount = 0;
  uint64_t peakUsageByteCount = 0;
  bool workspaceSizeExprKnown = false;
  std::string workspaceSizeExpr;
  bool capacityCheckDeferred = false;
  llvm::SmallVector<StaticMemoryLiveInterval, 8> liveIntervals;
  llvm::SmallVector<StaticMemoryWorkspaceSlot, 8> workspaceSlots;
};

struct MovementStep {
  unsigned stepId = 0;
  unsigned valueId = 0;
  unsigned slotId = 0;
  MemoryPlace srcPlace = MemoryPlace::GM;
  MemoryPlace dstPlace = MemoryPlace::VECCALC;
  bool pathSelected = false;
  unsigned pathVariant = 0;
  bool pathSelectionDeferred = true;
  bool staticByteSizeKnown = false;
  uint64_t byteSize = 0;
};

struct MovementPlan {
  std::string kernelId;
  std::string mode = "none";
  unsigned crossPlaceEdgeCount = 0;
  unsigned movementCount = 0;
  unsigned redundantMovementCount = 0;
  unsigned movementDemandCount = 0;
  unsigned selectedPathCount = 0;
  unsigned pathSelectionDeferredCount = 0;
  unsigned workspaceReuseCandidateCount = 0;
  unsigned dynamicViewChainRewriteCount = 0;
  unsigned deferredViewChainRewriteCount = 0;
  bool materializationDeferred = false;
  llvm::SmallVector<MovementStep, 8> movementSteps;
};

struct MemoryRealizationPlan {
  std::string kernelId;
  std::string mode = "none";
  bool frozen = false;
  std::string verificationScope = "none";
  bool planIdsVerified = false;
  unsigned memorySpaceAnnotationCount = 0;
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

} // namespace mlir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_REALIZETYPES_H
