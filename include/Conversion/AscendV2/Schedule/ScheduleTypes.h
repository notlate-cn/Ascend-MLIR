//===- ScheduleTypes.h - Ascend V2 schedule data model -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_SCHEDULETYPES_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_SCHEDULETYPES_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <string>
#include <utility>

namespace mlir::afir::ascend::v2::schedule {

inline constexpr llvm::StringLiteral kKernelAttr = "ascend.v2.kernel";
inline constexpr llvm::StringLiteral kPrimaryAttr = "ascend.v2.primary";
inline constexpr llvm::StringLiteral kOpRoleAttr = "ascend.v2.op_role";
inline constexpr llvm::StringLiteral kScheduleFamilyAttr =
    "ascend.v2.schedule.family";
inline constexpr llvm::StringLiteral kScheduleTemplateAttr =
    "ascend.v2.schedule.template";
inline constexpr llvm::StringLiteral kScheduleDecisionIdAttr =
    "ascend.v2.schedule.decision_id";
inline constexpr llvm::StringLiteral kScheduleRuntimeTopKAttr =
    "ascend.v2.schedule.runtime_top_k";
inline constexpr llvm::StringLiteral kStructuredLoweringAttr =
    "ascend.v2.schedule.structured_lowering";

enum class OpRole { Unknown, Vector, Reduction, Cube, Memory };

enum class AxisKind {
  Parallel,
  Reduction,
  Broadcast,
  Unknown,
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

struct CoalescedAxisInfo {
  SmallVector<LogicalAxisInfo> logicalAxes;
  SmallVector<unsigned> parallelAxes;
  SmallVector<unsigned> reductionAxes;
  SmallVector<unsigned> broadcastAxes;
  SmallVector<AxisCoalescingBarrier> barriers;
};

struct ScheduleProblem {
  std::string kernelId;
  OpRole dominantRole = OpRole::Unknown;
  unsigned resultRank = 0;
  SmallVector<int64_t> resultShape;
  CoalescedAxisInfo axes;
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

struct ScheduleDecision {
  std::string decisionId;
  ScheduleInstance instance;
  SmallVector<ScheduleGuard> candidateGuards;
  SmallVector<ScheduleGuard> decisionGuards;
};

struct ScheduleDecisionSet {
  std::string kernelId;
  SmallVector<ScheduleDecision, 4> decisions;
  unsigned runtimeTopK = 1;
};

struct ScheduleSearchOptions {
  unsigned compileTimeTopK = 4;
};

inline OpRole parseOpRole(llvm::StringRef value) {
  return llvm::StringSwitch<OpRole>(value)
      .Case("cube", OpRole::Cube)
      .Case("reduction", OpRole::Reduction)
      .Case("vector", OpRole::Vector)
      .Case("memory", OpRole::Memory)
      .Default(OpRole::Unknown);
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

} // namespace mlir::afir::ascend::v2::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_SCHEDULETYPES_H
