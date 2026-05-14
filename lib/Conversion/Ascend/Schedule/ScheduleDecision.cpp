//===- ScheduleDecision.cpp - Ascend schedule decisions ----------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Schedule/ScheduleDecision.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <utility>

namespace mlir::afir::ascend::schedule {
namespace {

void printTileShape(llvm::ArrayRef<int64_t> tileSizes,
                    llvm::raw_ostream &os) {
  os << "[";
  for (auto [index, tileSize] : llvm::enumerate(tileSizes)) {
    if (index != 0)
      os << ",";
    if (ShapedType::isDynamic(tileSize))
      os << "?";
    else
      os << tileSize;
  }
  os << "]";
}

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

AxisTailPolicy selectConcreteTailPolicy(
    const AxisScheduleConstraint &constraint) {
  constexpr AxisTailPolicy kPreferenceOrder[] = {
      AxisTailPolicy::MaskedTail, AxisTailPolicy::ScalarEpilogue,
      AxisTailPolicy::PadAndMask, AxisTailPolicy::FullExtent,
      AxisTailPolicy::MustDivide};
  for (AxisTailPolicy policy : kPreferenceOrder) {
    if (llvm::is_contained(constraint.allowedTailPolicies, policy))
      return policy;
  }
  return AxisTailPolicy::MustDivide;
}

ScheduledAxisTailPlan buildTailPlanForTileIndex(const ScheduleProblem &problem,
                                                unsigned tileIndex,
                                                int64_t tileSize) {
  ScheduledAxisTailPlan plan;
  plan.logicalAxisId = tileIndex;
  plan.tileSize = tileSize;
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
    plan.selectedPolicy = selectConcreteTailPolicy(*constraint);
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
               const ScheduleInstance &instance) {
  SmallVector<ScheduledAxisTailPlan, 4> tailPlans;
  tailPlans.reserve(instance.tileShape.tileSizes.size());
  for (auto [index, tileSize] : llvm::enumerate(instance.tileShape.tileSizes)) {
    tailPlans.push_back(buildTailPlanForTileIndex(problem, index, tileSize));
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
    decision.candidateGuards = instance.candidateGuards;
    decision.decisionGuards = instance.decisionGuards;
    decision.tailPlans = buildTailPlans(problem, decision.instance);
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
    os << "  candidate_guards = "
       << selectedDecision.candidateGuards.size() << "\n";
    os << "  decision_guards = " << selectedDecision.decisionGuards.size()
       << "\n";
    os << "  selected_tile_shape = ";
    printTileShape(selectedDecision.instance.tileShape.tileSizes, os);
    os << "\n";
    os << "  tail_plans = ";
    printTailPlans(selectedDecision.tailPlans, os);
    os << "\n";
  }
}

} // namespace mlir::afir::ascend::schedule
