//===- ScheduleTypes.h - Ascend schedule data model -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULETYPES_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULETYPES_H

#include "Conversion/Ascend/Common/Attributes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <cstdint>
#include <string>
#include <utility>

namespace mlir::afir::ascend::schedule {

using ::mlir::afir::ascend::kKernelAttr;
using ::mlir::afir::ascend::kKernelizeHandwrittenKindAttentionSdpa;
using ::mlir::afir::ascend::kKernelizeHandwrittenKindAttr;
using ::mlir::afir::ascend::kKernelizeTemplateFamiliesAttr;
using ::mlir::afir::ascend::kOpRoleCube;
using ::mlir::afir::ascend::kOpRoleMemory;
using ::mlir::afir::ascend::kOpRoleAttr;
using ::mlir::afir::ascend::kOpRoleReduction;
using ::mlir::afir::ascend::kOpRoleVector;
using ::mlir::afir::ascend::kOpRolesAttr;
using ::mlir::afir::ascend::kPrimaryAttr;
using ::mlir::afir::ascend::kScheduleDecisionIdAttr;
using ::mlir::afir::ascend::kScheduleGuardMarkersAttr;
using ::mlir::afir::ascend::kScheduleKernelMetadataAttr;
using ::mlir::afir::ascend::kScheduleSelectedTileShapeAttr;
using ::mlir::afir::ascend::kScheduleTailPlanAttr;
using ::mlir::afir::ascend::kScheduleTailMarkersAttr;
using ::mlir::afir::ascend::kScheduleTailPoliciesAttr;
using ::mlir::afir::ascend::kScheduleTargetTilePolicyAttr;
using ::mlir::afir::ascend::kStructuredLoweringAttr;

inline constexpr llvm::StringLiteral kScheduleFamilyAttr =
    "ascend.schedule.family";
inline constexpr llvm::StringLiteral kScheduleTemplateAttr =
    "ascend.schedule.template";
inline constexpr llvm::StringLiteral kScheduleRuntimeTopKAttr =
    "ascend.schedule.runtime_top_k";

enum class OpRole { Unknown, Vector, Reduction, Cube, Memory };

enum class AxisKind {
  Parallel,
  Reduction,
  Unknown,
};

enum class AxisExecutionRole {
  BindCoreCandidate,
  KernelLoopCandidate,
  VectorizeCandidate,
  FullReduction,
  ChunkedReduction,
  BroadcastProjection,
  LayoutCarry,
};

enum class AxisTailPolicy {
  MustDivide,
  MaskedTail,
  ScalarEpilogue,
  PadAndMask,
  FullExtent,
};

enum class PrimitiveAxisUseKind {
  DataCopy,
  VectorCompute,
  Reduction,
  GatherIndex,
  CubeM,
  CubeN,
  CubeK,
  WriteBack,
};

enum class TailBufferingMode {
  SeparateTailBuffer,
  ReuseMainBufferAfterDrain,
};

enum class CoalescingHintKind {
  Vectorizable,
  LinearizeOnly,
};

enum class AxisBarrierKind {
  None,
  UnsupportedIndexingMap,
  UnsupportedIteratorType,
  RankMismatch,
};

enum class GuardKind {
  ShapeStaticEqual,
  ShapeDynamic,
  DivisibleBy,
  PositiveExtent,
};

enum class GuardAxisDomain {
  ResultDim,
  LogicalAxis,
};

struct ScheduleGuard {
  GuardKind kind = GuardKind::ShapeDynamic;
  GuardAxisDomain axisDomain = GuardAxisDomain::ResultDim;
  unsigned dim = 0;
  int64_t value = ShapedType::kDynamic;
  std::string text;
};

struct PatternOpView {
  Operation *op = nullptr;
  unsigned ordinal = 0;
  OpRole role = OpRole::Unknown;
  bool primary = false;
};

struct KernelPatternView {
  std::string kernelId;
  SmallVector<PatternOpView> ops;
  SmallVector<Operation *> primaryOps;
  SmallVector<std::string> templateFamilies;
  std::string handwrittenKind;
  OpRole dominantRole = OpRole::Unknown;
};

struct LogicalAxisInfo {
  unsigned logicalAxisId = 0;
  AxisKind kind = AxisKind::Unknown;
  int64_t staticExtent = ShapedType::kDynamic;
  SmallVector<std::pair<Operation *, unsigned>> rawAxes;
};

struct AxisCoalescingBarrier {
  Operation *op = nullptr;
  AxisBarrierKind kind = AxisBarrierKind::None;
  std::string reason;
};

struct AxisScheduleConstraint {
  unsigned logicalAxisId = 0;
  AxisKind kind = AxisKind::Unknown;
  SmallVector<AxisExecutionRole, 3> allowedRoles;
  SmallVector<AxisTailPolicy, 3> allowedTailPolicies;
  SmallVector<PrimitiveAxisUseKind, 4> primitiveUses;
  int64_t semanticAlignmentGranularity = 0;
  // Compatibility alias for older schedule consumers. New code should use
  // allowedTailPolicies and pick the selected policy explicitly.
  AxisTailPolicy tailPolicy = AxisTailPolicy::MustDivide;
  uint32_t coalescingGroupId = 0;
};

struct AxisCoalescingHint {
  uint32_t groupId = 0;
  CoalescingHintKind kind = CoalescingHintKind::LinearizeOnly;
  SmallVector<unsigned, 2> memberAxisIds;
};

struct CoalescedAxisInfo {
  SmallVector<LogicalAxisInfo> logicalAxes;
  SmallVector<unsigned> parallelAxes;
  SmallVector<unsigned> reductionAxes;
  SmallVector<unsigned> broadcastAxes;
  SmallVector<AxisCoalescingBarrier> barriers;
  SmallVector<AxisScheduleConstraint> axisScheduleConstraints;
  SmallVector<AxisCoalescingHint> axisCoalescingHints;
};

struct TargetTilePolicy {
  std::string policyId = "target_default_32";
  int64_t defaultParallelTile = 32;
  int64_t semanticAlignmentGranularity = 16;
  int64_t vectorBufferCount = 4;
  SmallVector<AxisTailPolicy, 5> tailPolicyPreference = {
      AxisTailPolicy::MaskedTail, AxisTailPolicy::ScalarEpilogue,
      AxisTailPolicy::PadAndMask, AxisTailPolicy::FullExtent,
      AxisTailPolicy::MustDivide};
};

struct ScheduleProblem {
  std::string kernelId;
  OpRole dominantRole = OpRole::Unknown;
  unsigned resultRank = 0;
  unsigned resultElementBitWidth = 0;
  SmallVector<int64_t> resultShape;
  CoalescedAxisInfo axes;
  TargetTilePolicy targetTilePolicy;
  unsigned guardBudget = 8;
  SmallVector<std::string> templateTags;
  SmallVector<std::string> structureConstraints;
  SmallVector<std::string> shapeConstraints;
};

struct ScheduleTemplate {
  std::string family;
  std::string name;
  SmallVector<std::string> tags;
  unsigned minRank = 0;
  unsigned maxRank = 0;
  unsigned priority = 0;
};

struct TileShape {
  SmallVector<int64_t> tileSizes;
};

struct ScheduleInstance {
  std::string instanceId;
  ScheduleTemplate tmpl;
  TileShape tileShape;
  int64_t estimatedCost = 0;
  SmallVector<ScheduleGuard> candidateGuards;
  SmallVector<ScheduleGuard> decisionGuards;
  SmallVector<std::string> reasonKinds;
};

struct ScheduledAxisTailPlan {
  unsigned logicalAxisId = 0;
  AxisTailPolicy selectedPolicy = AxisTailPolicy::MaskedTail;
  SmallVector<PrimitiveAxisUseKind, 4> affectedPrimitiveUses;
  int64_t extent = ShapedType::kDynamic;
  int64_t tileSize = ShapedType::kDynamic;
  int64_t alignmentGranularity = 0;
  int64_t mainExtent = ShapedType::kDynamic;
  int64_t tailExtent = ShapedType::kDynamic;
  TailBufferingMode tailBufferingMode =
      TailBufferingMode::SeparateTailBuffer;
  bool emitsRuntimeGuard = false;
};

struct ScheduleDecision {
  std::string decisionId;
  ScheduleInstance instance;
  SmallVector<ScheduledAxisTailPlan, 4> tailPlans;
};

struct ScheduleDecisionSet {
  std::string kernelId;
  SmallVector<ScheduleDecision, 4> decisions;
  unsigned runtimeTopK = 1;
};

struct ScheduleSearchOptions {
  unsigned compileTimeTopK = 4;
  unsigned runtimeTopK = 1;
  unsigned maxAxisProductTileShapes = 64;
};

struct ShapeBucketKey {
  std::string kernelId;
  std::string family;
  SmallVector<int64_t> resultShape;
};

struct TuningResultKey {
  ShapeBucketKey bucket;
  std::string templateName;
  SmallVector<int64_t> tileShape;
};

struct ScheduleCacheReport {
  unsigned shapeBucketLookups = 0;
  unsigned shapeBucketMisses = 0;
  unsigned tuningLookups = 0;
  unsigned tuningMisses = 0;
  unsigned selectedDecisionEntries = 0;
  unsigned guardBudgetPruned = 0;
  unsigned negativeCacheHits = 0;
  unsigned negativeCacheEntries = 0;
  unsigned persistentTuningHits = 0;
};

inline OpRole parseOpRole(llvm::StringRef value) {
  return llvm::StringSwitch<OpRole>(value)
      .Case("cube", OpRole::Cube)
      .Case("Cube", OpRole::Cube)
      .Case("reduction", OpRole::Reduction)
      .Case("Reduction", OpRole::Reduction)
      .Case("vector", OpRole::Vector)
      .Case("Vector", OpRole::Vector)
      .Case("memory", OpRole::Memory)
      .Case("Memory", OpRole::Memory)
      .Default(OpRole::Unknown);
}

inline bool opRolesAttrHasRole(ArrayAttr roles, OpRole role) {
  for (Attribute attr : roles) {
    auto roleAttr = dyn_cast<StringAttr>(attr);
    if (roleAttr && parseOpRole(roleAttr.getValue()) == role)
      return true;
  }
  return false;
}

inline OpRole deriveOpRole(Operation *op) {
  if (auto roles = op->getAttrOfType<ArrayAttr>(kOpRolesAttr)) {
    for (OpRole role :
         {OpRole::Cube, OpRole::Reduction, OpRole::Vector, OpRole::Memory}) {
      if (opRolesAttrHasRole(roles, role))
        return role;
    }
  }

  auto roleAttr = op->getAttrOfType<StringAttr>(kOpRoleAttr);
  return roleAttr ? parseOpRole(roleAttr.getValue()) : OpRole::Unknown;
}

inline llvm::StringRef stringifyAxisKind(AxisKind kind) {
  switch (kind) {
  case AxisKind::Parallel:
    return "parallel";
  case AxisKind::Reduction:
    return "reduction";
  case AxisKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

inline llvm::StringRef stringifyAxisExecutionRole(AxisExecutionRole role) {
  switch (role) {
  case AxisExecutionRole::BindCoreCandidate:
    return "bind_core";
  case AxisExecutionRole::KernelLoopCandidate:
    return "kernel_loop";
  case AxisExecutionRole::VectorizeCandidate:
    return "vectorize";
  case AxisExecutionRole::FullReduction:
    return "full_reduction";
  case AxisExecutionRole::ChunkedReduction:
    return "chunked_reduction";
  case AxisExecutionRole::BroadcastProjection:
    return "broadcast_projection";
  case AxisExecutionRole::LayoutCarry:
    return "layout_carry";
  }
  return "layout_carry";
}

inline llvm::StringRef stringifyAxisTailPolicy(AxisTailPolicy policy) {
  switch (policy) {
  case AxisTailPolicy::MustDivide:
    return "must_divide";
  case AxisTailPolicy::MaskedTail:
    return "masked_tail";
  case AxisTailPolicy::ScalarEpilogue:
    return "scalar_epilogue";
  case AxisTailPolicy::PadAndMask:
    return "pad_and_mask";
  case AxisTailPolicy::FullExtent:
    return "full_extent";
  }
  return "must_divide";
}

inline llvm::StringRef
stringifyPrimitiveAxisUseKind(PrimitiveAxisUseKind kind) {
  switch (kind) {
  case PrimitiveAxisUseKind::DataCopy:
    return "data_copy";
  case PrimitiveAxisUseKind::VectorCompute:
    return "vector_compute";
  case PrimitiveAxisUseKind::Reduction:
    return "reduction";
  case PrimitiveAxisUseKind::GatherIndex:
    return "gather_index";
  case PrimitiveAxisUseKind::CubeM:
    return "cube_m";
  case PrimitiveAxisUseKind::CubeN:
    return "cube_n";
  case PrimitiveAxisUseKind::CubeK:
    return "cube_k";
  case PrimitiveAxisUseKind::WriteBack:
    return "write_back";
  }
  return "data_copy";
}

inline llvm::StringRef stringifyTailBufferingMode(TailBufferingMode mode) {
  switch (mode) {
  case TailBufferingMode::SeparateTailBuffer:
    return "separate_tail_buffer";
  case TailBufferingMode::ReuseMainBufferAfterDrain:
    return "reuse_main_buffer_after_drain";
  }
  return "separate_tail_buffer";
}

inline llvm::StringRef stringifyGuardKind(GuardKind kind) {
  switch (kind) {
  case GuardKind::ShapeStaticEqual:
    return "shape_static_equal";
  case GuardKind::ShapeDynamic:
    return "shape_dynamic";
  case GuardKind::DivisibleBy:
    return "divisible_by";
  case GuardKind::PositiveExtent:
    return "positive_extent";
  }
  return "shape_dynamic";
}

inline llvm::StringRef stringifyGuardAxisDomain(GuardAxisDomain domain) {
  switch (domain) {
  case GuardAxisDomain::ResultDim:
    return "result_dim";
  case GuardAxisDomain::LogicalAxis:
    return "logical_axis";
  }
  return "result_dim";
}

inline llvm::StringRef stringifyCoalescingHintKind(CoalescingHintKind kind) {
  switch (kind) {
  case CoalescingHintKind::Vectorizable:
    return "vectorizable";
  case CoalescingHintKind::LinearizeOnly:
    return "linearize_only";
  }
  return "linearize_only";
}

inline llvm::StringRef stringifyOpRole(OpRole role) {
  switch (role) {
  case OpRole::Cube:
    return "cube";
  case OpRole::Reduction:
    return "reduction";
  case OpRole::Vector:
    return "vector";
  case OpRole::Memory:
    return "memory";
  case OpRole::Unknown:
    return "unknown";
  }
  return "unknown";
}

} // namespace mlir::afir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULETYPES_H
