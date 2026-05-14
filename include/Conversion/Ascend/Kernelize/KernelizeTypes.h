//===- KernelizeTypes.h - Ascend kernelize data model --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZETYPES_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZETYPES_H

#include "Conversion/Ascend/Common/Attributes.h"

#include "llvm/ADT/StringRef.h"

namespace mlir::afir::ascend::kernelize {

using ::mlir::afir::ascend::kKernelAttr;
using ::mlir::afir::ascend::kNormalizedAttr;
using ::mlir::afir::ascend::kOpRoleAttr;
using ::mlir::afir::ascend::kOpRolesAttr;
using ::mlir::afir::ascend::kPrimaryAttr;

inline constexpr llvm::StringLiteral kBranchRootAttr = "ascend.branch_root";
inline constexpr llvm::StringLiteral kBranchGroupAttr = "ascend.branch_group";
inline constexpr llvm::StringLiteral kMergeRootAttr = "ascend.merge_root";
inline constexpr llvm::StringLiteral kMergeGroupAttr = "ascend.merge_group";

enum class AccessPatternKind {
  NotApplicable,
  Elementwise,
  Broadcast,
  Reduction,
  Contraction,
  Gather,
  Scatter,
  LayoutTransform,
  Unknown
};

enum class OpRole {
  Primary,
  Cube,
  Vector,
  Reduction,
  Injective,
  Indexing,
  LayoutTransform,
  Branch,
  Merge,
  Barrier,
  Unsupported
};

enum class CandidateKind {
  Fusion,
  Merged,
  HorizontalFusion,
  FallbackSingleOp,
  HandwrittenPattern
};

enum class KernelizePrimitiveKind {
  Unknown,
  ElementwiseChain,
  ConsumerIntoPrimary,
  ReductionInlining,
  FallbackSingleOp,
  HandwrittenPattern
};

enum class KernelPatternEdgeKind {
  DataDependency,
  Overlap,
  MustCoLocate,
  MustSeparate,
  ScheduleBarrier
};

struct OperationId {
  unsigned value = 0;
};

struct KernelizeConfig {
  unsigned maxPrimitivePerOp = 4;
  unsigned maxOpsPerCandidate = 32;
  unsigned maxPrimaryRolesPerCandidate = 2;
  unsigned maxHorizontalFusionGroupSize = 8;
};

inline llvm::StringRef stringifyAccessPattern(AccessPatternKind kind) {
  switch (kind) {
  case AccessPatternKind::NotApplicable:
    return "NotApplicable";
  case AccessPatternKind::Elementwise:
    return "Elementwise";
  case AccessPatternKind::Broadcast:
    return "Broadcast";
  case AccessPatternKind::Reduction:
    return "Reduction";
  case AccessPatternKind::Contraction:
    return "Contraction";
  case AccessPatternKind::Gather:
    return "Gather";
  case AccessPatternKind::Scatter:
    return "Scatter";
  case AccessPatternKind::LayoutTransform:
    return "LayoutTransform";
  case AccessPatternKind::Unknown:
    return "Unknown";
  }
  return "Unknown";
}

inline llvm::StringRef stringifyOpRole(OpRole role) {
  switch (role) {
  case OpRole::Primary:
    return "Primary";
  case OpRole::Cube:
    return "Cube";
  case OpRole::Vector:
    return "Vector";
  case OpRole::Reduction:
    return "Reduction";
  case OpRole::Injective:
    return "Injective";
  case OpRole::Indexing:
    return "Indexing";
  case OpRole::LayoutTransform:
    return "LayoutTransform";
  case OpRole::Branch:
    return "Branch";
  case OpRole::Merge:
    return "Merge";
  case OpRole::Barrier:
    return "Barrier";
  case OpRole::Unsupported:
    return "Unsupported";
  }
  return "Unsupported";
}

inline llvm::StringRef stringifyCandidateKind(CandidateKind kind) {
  switch (kind) {
  case CandidateKind::Fusion:
    return "Fusion";
  case CandidateKind::Merged:
    return "Merged";
  case CandidateKind::HorizontalFusion:
    return "HorizontalFusion";
  case CandidateKind::FallbackSingleOp:
    return "FallbackSingleOp";
  case CandidateKind::HandwrittenPattern:
    return "HandwrittenPattern";
  }
  return "Fusion";
}

inline llvm::StringRef
stringifyKernelizePrimitiveKind(KernelizePrimitiveKind kind) {
  switch (kind) {
  case KernelizePrimitiveKind::Unknown:
    return "Unknown";
  case KernelizePrimitiveKind::ElementwiseChain:
    return "ElementwiseChain";
  case KernelizePrimitiveKind::ConsumerIntoPrimary:
    return "ConsumerIntoPrimary";
  case KernelizePrimitiveKind::ReductionInlining:
    return "ReductionInlining";
  case KernelizePrimitiveKind::FallbackSingleOp:
    return "FallbackSingleOp";
  case KernelizePrimitiveKind::HandwrittenPattern:
    return "HandwrittenPattern";
  }
  return "Unknown";
}

inline llvm::StringRef
stringifyKernelPatternEdgeKind(KernelPatternEdgeKind kind) {
  switch (kind) {
  case KernelPatternEdgeKind::DataDependency:
    return "DataDependency";
  case KernelPatternEdgeKind::Overlap:
    return "Overlap";
  case KernelPatternEdgeKind::MustCoLocate:
    return "MustCoLocate";
  case KernelPatternEdgeKind::MustSeparate:
    return "MustSeparate";
  case KernelPatternEdgeKind::ScheduleBarrier:
    return "ScheduleBarrier";
  }
  return "DataDependency";
}

} // namespace mlir::afir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZETYPES_H
