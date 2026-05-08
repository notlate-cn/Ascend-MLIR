//===- KernelizeTypes.h - Ascend V2 kernelize data model --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_KERNELIZETYPES_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_KERNELIZETYPES_H

#include "llvm/ADT/StringRef.h"

namespace mlir::afir::ascend::v2::kernelize {

inline constexpr llvm::StringLiteral kNormalizedAttr = "ascend.v2.normalized";
inline constexpr llvm::StringLiteral kOpRoleAttr = "ascend.v2.op_role";
inline constexpr llvm::StringLiteral kOpRolesAttr = "ascend.v2.op_roles";
inline constexpr llvm::StringLiteral kKernelAttr = "ascend.v2.kernel";
inline constexpr llvm::StringLiteral kPrimaryAttr = "ascend.v2.primary";
inline constexpr llvm::StringLiteral kBranchRootAttr = "ascend.v2.branch_root";
inline constexpr llvm::StringLiteral kBranchGroupAttr = "ascend.v2.branch_group";
inline constexpr llvm::StringLiteral kMergeRootAttr = "ascend.v2.merge_root";
inline constexpr llvm::StringLiteral kMergeGroupAttr = "ascend.v2.merge_group";

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
  unsigned maxBranchesPerCandidate = 4;
  unsigned maxPrimaryRolesPerCandidate = 2;
  unsigned maxHorizontalFusionGroupSize = 8;
  unsigned localTopKPerPrimaryOpNeighborhood = 8;
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

} // namespace mlir::afir::ascend::v2::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_KERNELIZETYPES_H
