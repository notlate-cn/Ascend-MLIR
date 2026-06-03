//===- ScheduleDecision.cpp - Ascend schedule decisions ----------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "ScheduleDecision.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <iterator>
#include <utility>

namespace mlir::ascend::schedule {
namespace {

void printMaybeDynamic(int64_t value, llvm::raw_ostream &os) {
  if (ShapedType::isDynamic(value))
    os << "?";
  else
    os << value;
}

void printPrimitiveUses(ArrayRef<PrimitiveAxisUseKind> uses,
                        llvm::raw_ostream &os) {
  os << "[";
  for (auto [index, use] : llvm::enumerate(uses)) {
    if (index != 0)
      os << ",";
    os << stringifyPrimitiveAxisUseKind(use);
  }
  os << "]";
}

void printAxisExecutionRoles(ArrayRef<AxisExecutionRole> roles,
                             llvm::raw_ostream &os) {
  os << "[";
  for (auto [index, role] : llvm::enumerate(roles)) {
    if (index != 0)
      os << ",";
    os << stringifyAxisExecutionRole(role);
  }
  os << "]";
}

void printTileParams(ArrayRef<ScheduleTileParam> tileParams,
                     llvm::raw_ostream &os) {
  if (tileParams.empty()) {
    os << "[]";
    return;
  }

  for (auto [index, param] : llvm::enumerate(tileParams)) {
    if (index != 0)
      os << " ";
    os << "[name=" << param.name << " axis=" << param.logicalAxisId
       << " binding=" << stringifyTileParamBinding(param.binding)
       << " axis_kind=" << stringifyAxisKind(param.axisKind)
       << " default=";
    printMaybeDynamic(param.defaultValue, os);
    os << " upper_bound=";
    printMaybeDynamic(param.upperBound, os);
    os << " extent=";
    printMaybeDynamic(param.extent, os);
    os << " roles=";
    printAxisExecutionRoles(param.roles, os);
    os << " primitive_uses=";
    printPrimitiveUses(param.primitiveUses, os);
    os << "]";
  }
}

void printTailPlans(ArrayRef<ScheduledAxisTailPlan> tailPlans,
                    llvm::raw_ostream &os) {
  if (tailPlans.empty()) {
    os << "[]";
    return;
  }

  for (auto [index, plan] : llvm::enumerate(tailPlans)) {
    if (index != 0)
      os << " ";
    os << "[axis=" << plan.logicalAxisId
       << " selected=" << stringifyAxisTailPolicy(plan.selectedPolicy)
       << " affected=";
    printPrimitiveUses(plan.affectedPrimitiveUses, os);
    os << " align=" << plan.alignmentGranularity
       << " buffering=" << stringifyTailBufferingMode(plan.tailBufferingMode)
       << " guard=" << (plan.emitsRuntimeGuard ? "true" : "false")
       << " extent=";
    printMaybeDynamic(plan.extent, os);
    os << " tile=";
    printMaybeDynamic(plan.tileSize, os);
    os << " main=";
    printMaybeDynamic(plan.mainExtent, os);
    os << " tail=";
    printMaybeDynamic(plan.tailExtent, os);
    os << "]";
  }
}

const AxisScheduleConstraint *
lookupAxisScheduleConstraint(const CoalescedAxisInfo &axes,
                             unsigned logicalAxisId) {
  for (const AxisScheduleConstraint &constraint :
       axes.axisScheduleConstraints) {
    if (constraint.logicalAxisId == logicalAxisId)
      return &constraint;
  }
  return nullptr;
}

bool hasAxisExecutionRole(ArrayRef<AxisExecutionRole> roles,
                          AxisExecutionRole role) {
  return llvm::is_contained(roles, role);
}

bool hasPrimitiveUse(ArrayRef<PrimitiveAxisUseKind> uses,
                     PrimitiveAxisUseKind use) {
  return llvm::is_contained(uses, use);
}

void addPrimitiveUse(SmallVectorImpl<PrimitiveAxisUseKind> &uses,
                     PrimitiveAxisUseKind use) {
  if (!hasPrimitiveUse(uses, use))
    uses.push_back(use);
}

std::string defaultTileParamName(const ScheduleProblem &problem,
                                 unsigned tileIndex) {
  if (problem.dominantRole == OpRole::Cube && tileIndex == 2)
    return "t_K";

  static constexpr llvm::StringLiteral kDefaultNames[] = {
      "TB_M", "TB_N", "Tb_M", "Tb_N"};
  if (tileIndex < std::size(kDefaultNames))
    return kDefaultNames[tileIndex].str();

  return (llvm::Twine("tile_") + llvm::Twine(tileIndex)).str();
}

void augmentPrimitiveUsesForRole(const ScheduleProblem &problem,
                                 unsigned tileIndex,
                                 SmallVectorImpl<PrimitiveAxisUseKind> &uses) {
  switch (problem.dominantRole) {
  case OpRole::Cube:
    if (tileIndex == 0)
      addPrimitiveUse(uses, PrimitiveAxisUseKind::CubeM);
    else if (tileIndex == 1)
      addPrimitiveUse(uses, PrimitiveAxisUseKind::CubeN);
    else if (tileIndex == 2)
      addPrimitiveUse(uses, PrimitiveAxisUseKind::CubeK);
    return;
  case OpRole::Memory:
    uses.clear();
    addPrimitiveUse(uses, PrimitiveAxisUseKind::DataCopy);
    addPrimitiveUse(uses, PrimitiveAxisUseKind::WriteBack);
    return;
  case OpRole::Vector:
  case OpRole::Reduction:
  case OpRole::Unknown:
    return;
  }
}

TileParamBinding selectTileParamBinding(const ScheduleProblem &problem,
                                        unsigned tileIndex,
                                        const ScheduleTileParam &param) {
  if (problem.dominantRole == OpRole::Cube && tileIndex == 2)
    return TileParamBinding::Extent;

  bool fullReduction =
      hasAxisExecutionRole(param.roles, AxisExecutionRole::FullReduction);
  bool chunkedReduction =
      hasAxisExecutionRole(param.roles, AxisExecutionRole::ChunkedReduction);
  bool runtimeParallel =
      hasAxisExecutionRole(param.roles, AxisExecutionRole::BindCoreCandidate) ||
      hasAxisExecutionRole(param.roles,
                           AxisExecutionRole::KernelLoopCandidate) ||
      hasAxisExecutionRole(param.roles, AxisExecutionRole::VectorizeCandidate);
  if (fullReduction && !chunkedReduction && !runtimeParallel)
    return TileParamBinding::Extent;
  if (runtimeParallel || param.axisKind == AxisKind::Parallel)
    return TileParamBinding::Runtime;
  return TileParamBinding::StaticFallback;
}

int64_t selectRuntimeDefaultTile(const ScheduleProblem &problem,
                                 int64_t chosenTile, int64_t extent) {
  int64_t fallback = problem.targetTilePolicy.defaultParallelTile;
  if (fallback <= 0)
    fallback = 1;
  if (!ShapedType::isDynamic(chosenTile) && chosenTile > 0)
    fallback = chosenTile;
  if (!ShapedType::isDynamic(extent) && extent > 0)
    fallback = std::min(fallback, extent);
  return std::max<int64_t>(1, fallback);
}

int64_t selectRuntimeTileUpperBound(int64_t defaultValue, int64_t extent) {
  if (!ShapedType::isDynamic(extent) && extent > 0)
    return extent;
  return defaultValue;
}

ScheduleTileParam buildTileParamForTileIndex(const ScheduleProblem &problem,
                                             unsigned tileIndex,
                                             int64_t chosenTile) {
  ScheduleTileParam param;
  param.logicalAxisId = tileIndex;
  param.name = defaultTileParamName(problem, tileIndex);
  if (tileIndex < problem.axes.logicalAxes.size()) {
    const LogicalAxisInfo &axis = problem.axes.logicalAxes[tileIndex];
    param.logicalAxisId = axis.logicalAxisId;
    param.axisKind = axis.kind;
    param.extent = axis.staticExtent;
  } else if (tileIndex < problem.resultShape.size()) {
    param.extent = problem.resultShape[tileIndex];
  }

  if (const AxisScheduleConstraint *constraint =
          lookupAxisScheduleConstraint(problem.axes, param.logicalAxisId)) {
    param.axisKind = constraint->kind;
    param.roles = constraint->allowedRoles;
    param.primitiveUses = constraint->primitiveUses;
  }
  augmentPrimitiveUsesForRole(problem, tileIndex, param.primitiveUses);

  param.binding = selectTileParamBinding(problem, tileIndex, param);
  switch (param.binding) {
  case TileParamBinding::Runtime:
    param.defaultValue =
        selectRuntimeDefaultTile(problem, ShapedType::kDynamic, param.extent);
    param.upperBound =
        selectRuntimeTileUpperBound(param.defaultValue, param.extent);
    break;
  case TileParamBinding::Extent:
    param.defaultValue =
        !ShapedType::isDynamic(param.extent) ? param.extent : chosenTile;
    param.upperBound = param.defaultValue;
    break;
  case TileParamBinding::StaticFallback:
    param.defaultValue = chosenTile;
    param.upperBound = chosenTile;
    break;
  }
  return param;
}

SmallVector<ScheduleTileParam, 4>
buildTileParams(const ScheduleProblem &problem,
                const ScheduleInstance &instance) {
  SmallVector<ScheduleTileParam, 4> tileParams;
  tileParams.reserve(instance.tileShape.tileSizes.size());
  for (auto [index, tileSize] : llvm::enumerate(instance.tileShape.tileSizes))
    tileParams.push_back(
        buildTileParamForTileIndex(problem, index, tileSize));
  return tileParams;
}

AxisTailPolicy selectConcreteTailPolicy(
    const AxisScheduleConstraint &constraint,
    const TargetTilePolicy &targetTilePolicy) {
  for (AxisTailPolicy policy : targetTilePolicy.tailPolicyPreference) {
    if (llvm::is_contained(constraint.allowedTailPolicies, policy))
      return policy;
  }
  return AxisTailPolicy::MustDivide;
}

ScheduledAxisTailPlan buildTailPlanForTileIndex(const ScheduleProblem &problem,
                                                unsigned tileIndex,
                                                int64_t tileSize,
                                                const ScheduleTileParam *param) {
  ScheduledAxisTailPlan plan;
  plan.logicalAxisId = tileIndex;
  plan.tileSize = param && param->binding == TileParamBinding::Runtime
                      ? ShapedType::kDynamic
                      : tileSize;
  plan.tailBufferingMode = TailBufferingMode::SeparateTailBuffer;

  if (tileIndex < problem.axes.logicalAxes.size()) {
    const LogicalAxisInfo &axis = problem.axes.logicalAxes[tileIndex];
    plan.logicalAxisId = axis.logicalAxisId;
    plan.extent = axis.staticExtent;
  }

  const AxisScheduleConstraint *constraint =
      lookupAxisScheduleConstraint(problem.axes, plan.logicalAxisId);
  if (!constraint) {
    plan.selectedPolicy = AxisTailPolicy::MustDivide;
    plan.emitsRuntimeGuard = true;
  } else {
    plan.selectedPolicy =
        selectConcreteTailPolicy(*constraint, problem.targetTilePolicy);
    plan.affectedPrimitiveUses = constraint->primitiveUses;
    plan.alignmentGranularity = constraint->semanticAlignmentGranularity;
    plan.emitsRuntimeGuard =
        plan.selectedPolicy == AxisTailPolicy::MustDivide;
  }

  if (!ShapedType::isDynamic(plan.extent) &&
      !ShapedType::isDynamic(plan.tileSize) && plan.tileSize > 0) {
    plan.mainExtent = (plan.extent / plan.tileSize) * plan.tileSize;
    plan.tailExtent = plan.extent - plan.mainExtent;
  }

  return plan;
}

SmallVector<ScheduledAxisTailPlan, 4>
buildTailPlans(const ScheduleProblem &problem,
               const ScheduleInstance &instance,
               ArrayRef<ScheduleTileParam> tileParams) {
  SmallVector<ScheduledAxisTailPlan, 4> tailPlans;
  tailPlans.reserve(instance.tileShape.tileSizes.size());
  for (auto [index, tileSize] : llvm::enumerate(instance.tileShape.tileSizes)) {
    const ScheduleTileParam *param =
        index < tileParams.size() ? &tileParams[index] : nullptr;
    tailPlans.push_back(
        buildTailPlanForTileIndex(problem, index, tileSize, param));
  }
  return tailPlans;
}

} // namespace

ScheduleDecisionSet buildScheduleDecisionSet(
    const ScheduleProblem &problem, llvm::ArrayRef<ScheduleInstance> instances,
    const ScheduleSearchOptions &options) {
  ScheduleDecisionSet decisionSet;
  decisionSet.kernelId = problem.kernelId;
  for (auto [index, instance] : llvm::enumerate(instances)) {
    ScheduleDecision decision;
    decision.decisionId = (llvm::Twine(problem.kernelId) + ".decision." +
                           llvm::Twine(index))
                              .str();
    decision.instance = instance;
    decision.tileParams = buildTileParams(problem, decision.instance);
    decision.tailPlans =
        buildTailPlans(problem, decision.instance, decision.tileParams);
    decisionSet.decisions.push_back(std::move(decision));
  }

  if (decisionSet.decisions.empty()) {
    decisionSet.runtimeTopK = 0;
    return decisionSet;
  }

  unsigned requestedRuntimeTopK = std::max(1u, options.runtimeTopK);
  decisionSet.runtimeTopK =
      std::min<unsigned>(requestedRuntimeTopK, decisionSet.decisions.size());
  return decisionSet;
}

void printScheduleDecisionSetReport(const ScheduleDecisionSet &decisionSet,
                                    llvm::raw_ostream &os) {
  os << "ScheduleDecisionSet:\n";
  os << "  kernel = " << decisionSet.kernelId << "\n";
  os << "  decisions = " << decisionSet.decisions.size() << "\n";
  os << "  runtime_top_k = " << decisionSet.runtimeTopK << "\n";
  os << "  selected = ";
  if (decisionSet.decisions.empty())
    os << "<none>\n";
  else {
    os << decisionSet.decisions.front().decisionId << "\n";
    const ScheduleDecision &selectedDecision = decisionSet.decisions.front();
    const ScheduleInstance &selectedInstance = selectedDecision.instance;
    os << "  candidate_guards = "
       << selectedInstance.candidateGuards.size() << "\n";
    os << "  decision_guards = " << selectedInstance.decisionGuards.size()
       << "\n";
    os << "  tile_params = ";
    printTileParams(selectedDecision.tileParams, os);
    os << "\n";
    os << "  tail_plans = ";
    printTailPlans(selectedDecision.tailPlans, os);
    os << "\n";
  }
}

} // namespace mlir::ascend::schedule
