//===- ScheduleSearch.cpp - Ascend schedule search --------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "ScheduleSearch.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

using namespace mlir;

namespace mlir::ascend::schedule {
namespace {

int64_t normalizeExtent(int64_t extent) {
  if (ShapedType::isDynamic(extent))
    return ShapedType::kDynamic;
  return extent;
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
  return llvm::any_of(roles, [&](AxisExecutionRole candidate) {
    return candidate == role;
  });
}

int64_t getDefaultParallelTile(const ScheduleProblem &problem) {
  return problem.targetTilePolicy.defaultParallelTile;
}

int64_t getStaticTileArea(ArrayRef<int64_t> tileSizes) {
  constexpr int64_t kAreaCap = 1'000'000'000;
  int64_t area = 1;
  for (int64_t tileSize : tileSizes) {
    if (ShapedType::isDynamic(tileSize))
      continue;
    int64_t factor = std::max<int64_t>(tileSize, 1);
    if (area >= kAreaCap || factor > kAreaCap / area)
      return kAreaCap;
    area *= factor;
  }
  return area;
}

unsigned countDynamicTileSizes(ArrayRef<int64_t> tileSizes) {
  return llvm::count_if(tileSizes, [](int64_t tileSize) {
    return ShapedType::isDynamic(tileSize);
  });
}

bool hasReasonKind(const ScheduleInstance &instance, StringRef reasonKind) {
  return llvm::is_contained(instance.reasonKinds, reasonKind);
}

int64_t estimateDebugCost(const TileShape &tileShape) {
  constexpr int64_t kDynamicPenalty = 1'000'000'000;
  constexpr int64_t kAreaWeight = 1024;
  return countDynamicTileSizes(tileShape.tileSizes) * kDynamicPenalty -
         getStaticTileArea(tileShape.tileSizes) * kAreaWeight +
         static_cast<int64_t>(tileShape.tileSizes.size());
}

int64_t estimateScheduleCost(const ScheduleInstance &instance) {
  constexpr int64_t kGuardPenalty = 10'000;
  constexpr int64_t kBoundedBonus = 2'000'000'000'000;
  int64_t cost = estimateDebugCost(instance.tileShape);
  cost += static_cast<int64_t>(instance.candidateGuards.size() +
                               instance.decisionGuards.size()) *
          kGuardPenalty;
  if (hasReasonKind(instance, "bounded_parallel_reduction_tile") ||
      hasReasonKind(instance, "bounded_parallel_vector_tile"))
    cost -= kBoundedBonus;
  return cost;
}

bool isLowerRankedInstance(const ScheduleInstance &lhs,
                           const ScheduleInstance &rhs) {
  if (lhs.estimatedCost != rhs.estimatedCost)
    return lhs.estimatedCost < rhs.estimatedCost;

  unsigned lhsDynamicCount = countDynamicTileSizes(lhs.tileShape.tileSizes);
  unsigned rhsDynamicCount = countDynamicTileSizes(rhs.tileShape.tileSizes);
  if (lhsDynamicCount != rhsDynamicCount)
    return lhsDynamicCount < rhsDynamicCount;

  bool lhsBoundedReduction =
      hasReasonKind(lhs, "bounded_parallel_reduction_tile");
  bool rhsBoundedReduction =
      hasReasonKind(rhs, "bounded_parallel_reduction_tile");
  if (lhsBoundedReduction != rhsBoundedReduction)
    return lhsBoundedReduction;

  bool lhsBoundedVector = hasReasonKind(lhs, "bounded_parallel_vector_tile");
  bool rhsBoundedVector = hasReasonKind(rhs, "bounded_parallel_vector_tile");
  if (lhsBoundedVector != rhsBoundedVector)
    return lhsBoundedVector;

  int64_t lhsStaticArea = getStaticTileArea(lhs.tileShape.tileSizes);
  int64_t rhsStaticArea = getStaticTileArea(rhs.tileShape.tileSizes);
  if (lhsStaticArea != rhsStaticArea)
    return lhsStaticArea > rhsStaticArea;

  if (lhs.tileShape.tileSizes.size() != rhs.tileShape.tileSizes.size())
    return lhs.tileShape.tileSizes.size() < rhs.tileShape.tileSizes.size();

  unsigned lhsGuardCount =
      lhs.candidateGuards.size() + lhs.decisionGuards.size();
  unsigned rhsGuardCount =
      rhs.candidateGuards.size() + rhs.decisionGuards.size();
  if (lhsGuardCount != rhsGuardCount)
    return lhsGuardCount < rhsGuardCount;

  if (lhs.tmpl.family != rhs.tmpl.family)
    return lhs.tmpl.family < rhs.tmpl.family;
  if (lhs.tmpl.name != rhs.tmpl.name)
    return lhs.tmpl.name < rhs.tmpl.name;
  return std::lexicographical_compare(
      lhs.tileShape.tileSizes.begin(), lhs.tileShape.tileSizes.end(),
      rhs.tileShape.tileSizes.begin(), rhs.tileShape.tileSizes.end());
}

bool hasDynamicTileSize(const TileShape &tileShape) {
  return llvm::any_of(tileShape.tileSizes, [](int64_t tileSize) {
    return ShapedType::isDynamic(tileSize);
  });
}

bool isBoundedParallelReductionTile(const ScheduleProblem &problem,
                                    const TileShape &tileShape) {
  if (problem.dominantRole != OpRole::Reduction ||
      tileShape.tileSizes.size() != problem.axes.logicalAxes.size())
    return false;

  bool hasBoundedParallelAxis = false;
  bool hasFullReductionAxis = false;
  int64_t defaultParallelTile = getDefaultParallelTile(problem);
  for (auto [index, axis] : llvm::enumerate(problem.axes.logicalAxes)) {
    const AxisScheduleConstraint *constraint =
        lookupAxisScheduleConstraint(problem.axes, axis.logicalAxisId);
    if (!constraint)
      continue;

    int64_t tileSize = tileShape.tileSizes[index];
    if (hasAxisExecutionRole(constraint->allowedRoles,
                             AxisExecutionRole::FullReduction)) {
      if (tileSize != normalizeExtent(axis.staticExtent))
        return false;
      hasFullReductionAxis = true;
      continue;
    }

    if (constraint->kind == AxisKind::Parallel &&
        (ShapedType::isDynamic(axis.staticExtent) ||
         axis.staticExtent > defaultParallelTile) &&
        tileSize == defaultParallelTile &&
        (hasAxisExecutionRole(constraint->allowedRoles,
                              AxisExecutionRole::BindCoreCandidate) ||
         hasAxisExecutionRole(constraint->allowedRoles,
                              AxisExecutionRole::KernelLoopCandidate)))
      hasBoundedParallelAxis = true;
  }

  return hasBoundedParallelAxis && hasFullReductionAxis;
}

bool isBoundedParallelVectorTile(const ScheduleProblem &problem,
                                 const TileShape &tileShape) {
  if (problem.dominantRole != OpRole::Vector ||
      tileShape.tileSizes.size() != problem.axes.logicalAxes.size())
    return false;

  bool hasBoundedParallelAxis = false;
  int64_t defaultParallelTile = getDefaultParallelTile(problem);
  for (auto [index, axis] : llvm::enumerate(problem.axes.logicalAxes)) {
    const AxisScheduleConstraint *constraint =
        lookupAxisScheduleConstraint(problem.axes, axis.logicalAxisId);
    if (!constraint)
      continue;

    int64_t tileSize = tileShape.tileSizes[index];
    if (constraint->kind != AxisKind::Parallel) {
      if (tileSize != normalizeExtent(axis.staticExtent))
        return false;
      continue;
    }

    bool canBindOrLoop =
        hasAxisExecutionRole(constraint->allowedRoles,
                             AxisExecutionRole::BindCoreCandidate) ||
        hasAxisExecutionRole(constraint->allowedRoles,
                             AxisExecutionRole::KernelLoopCandidate);
    if (canBindOrLoop &&
        (ShapedType::isDynamic(axis.staticExtent) ||
         axis.staticExtent > defaultParallelTile) &&
        tileSize == defaultParallelTile) {
      hasBoundedParallelAxis = true;
      continue;
    }

    if (tileSize != normalizeExtent(axis.staticExtent))
      return false;
  }

  return hasBoundedParallelAxis;
}

std::string getSymbolicTileParamName(StringRef symbolName) {
  return (llvm::Twine("T_") + symbolName).str();
}

bool isFullExtentTile(int64_t tileSize, const LogicalAxisInfo &axis) {
  if (ShapedType::isDynamic(tileSize))
    return ShapedType::isDynamic(axis.staticExtent);
  return !ShapedType::isDynamic(axis.staticExtent) &&
         tileSize == axis.staticExtent;
}

bool isRuntimeTileAxis(const AxisScheduleConstraint *constraint) {
  if (!constraint)
    return false;
  return hasAxisExecutionRole(constraint->allowedRoles,
                              AxisExecutionRole::BindCoreCandidate) ||
         hasAxisExecutionRole(constraint->allowedRoles,
                              AxisExecutionRole::KernelLoopCandidate) ||
         hasAxisExecutionRole(constraint->allowedRoles,
                              AxisExecutionRole::VectorizeCandidate) ||
         hasAxisExecutionRole(constraint->allowedRoles,
                              AxisExecutionRole::ChunkedReduction);
}

bool hasGuardText(ArrayRef<ScheduleGuard> guards, StringRef text) {
  return llvm::any_of(guards, [&](const ScheduleGuard &guard) {
    return guard.text == text;
  });
}

void appendUniqueGuard(SmallVectorImpl<ScheduleGuard> &guards,
                       ScheduleGuard guard) {
  if (hasGuardText(guards, guard.text))
    return;
  guards.push_back(std::move(guard));
}

void appendSymbolicTileGuards(const ScheduleProblem &problem,
                              const TileShape &tileShape,
                              SmallVectorImpl<ScheduleGuard> &guards) {
  for (auto [index, tileSize] : llvm::enumerate(tileShape.tileSizes)) {
    if (index >= problem.axes.logicalAxes.size())
      continue;

    const LogicalAxisInfo &axis = problem.axes.logicalAxes[index];
    if (axis.symbolName.empty() || isFullExtentTile(tileSize, axis))
      continue;

    const AxisScheduleConstraint *constraint =
        lookupAxisScheduleConstraint(problem.axes, axis.logicalAxisId);
    if (!isRuntimeTileAxis(constraint))
      continue;

    std::string paramName = getSymbolicTileParamName(axis.symbolName);
    ScheduleGuard positiveGuard;
    positiveGuard.kind = GuardKind::PositiveExtent;
    positiveGuard.axisDomain = GuardAxisDomain::LogicalAxis;
    positiveGuard.dim = axis.logicalAxisId;
    positiveGuard.value = ShapedType::kDynamic;
    positiveGuard.text = (llvm::Twine(paramName) + " > 0").str();
    appendUniqueGuard(guards, std::move(positiveGuard));

    ScheduleGuard upperBoundGuard;
    upperBoundGuard.kind = GuardKind::ShapeDynamic;
    upperBoundGuard.axisDomain = GuardAxisDomain::LogicalAxis;
    upperBoundGuard.dim = axis.logicalAxisId;
    upperBoundGuard.value = ShapedType::kDynamic;
    upperBoundGuard.text =
        (llvm::Twine(paramName) + " <= " + axis.symbolName).str();
    appendUniqueGuard(guards, std::move(upperBoundGuard));

    if (!constraint || constraint->semanticAlignmentGranularity <= 0)
      continue;

    ScheduleGuard alignGuard;
    alignGuard.kind = GuardKind::DivisibleBy;
    alignGuard.axisDomain = GuardAxisDomain::LogicalAxis;
    alignGuard.dim = axis.logicalAxisId;
    alignGuard.value = constraint->semanticAlignmentGranularity;
    alignGuard.text =
        (llvm::Twine(paramName) + " % " +
         llvm::Twine(constraint->semanticAlignmentGranularity) + " == 0")
            .str();
    appendUniqueGuard(guards, std::move(alignGuard));
  }
}

void appendResultShapeGuards(const ScheduleProblem &problem,
                             SmallVectorImpl<ScheduleGuard> &guards) {
  for (auto [index, dim] : llvm::enumerate(problem.resultShape)) {
    StringRef symbolName;
    if (index < problem.axes.logicalAxes.size())
      symbolName = problem.axes.logicalAxes[index].symbolName;

    if (ShapedType::isDynamic(dim) && !symbolName.empty())
      continue;

    ScheduleGuard guard;
    guard.axisDomain = GuardAxisDomain::ResultDim;
    guard.dim = index;
    guard.value = dim;
    if (ShapedType::isDynamic(dim)) {
      guard.kind = GuardKind::PositiveExtent;
      guard.text =
          (llvm::Twine("d") + llvm::Twine(index) + " > 0").str();
    } else {
      guard.kind = GuardKind::ShapeStaticEqual;
      guard.text = (llvm::Twine("d") + llvm::Twine(index) + " == " +
                    llvm::Twine(dim))
                       .str();
    }
    guards.push_back(std::move(guard));
  }
}

void appendCandidateGuards(const ScheduleProblem &problem,
                           const TileShape &tileShape,
                           SmallVectorImpl<ScheduleGuard> &guards) {
  appendSymbolicTileGuards(problem, tileShape, guards);
  appendResultShapeGuards(problem, guards);
}

const AxisScheduleConstraint *
lookupAxisScheduleConstraintForTileIndex(const ScheduleProblem &problem,
                                         unsigned tileIndex) {
  unsigned logicalAxisId = tileIndex;
  if (tileIndex < problem.axes.logicalAxes.size())
    logicalAxisId = problem.axes.logicalAxes[tileIndex].logicalAxisId;

  return lookupAxisScheduleConstraint(problem.axes, logicalAxisId);
}

bool requiresDivisibleGuard(const AxisScheduleConstraint &constraint) {
  return constraint.allowedTailPolicies.size() == 1 &&
         constraint.allowedTailPolicies.front() == AxisTailPolicy::MustDivide;
}

void appendDecisionGuards(const ScheduleProblem &problem,
                          ArrayRef<int64_t> tileSizes,
                          SmallVectorImpl<ScheduleGuard> &guards) {
  for (auto [index, tileSize] : llvm::enumerate(tileSizes)) {
    if (ShapedType::isDynamic(tileSize) || tileSize <= 1)
      continue;

    const AxisScheduleConstraint *constraint =
        lookupAxisScheduleConstraintForTileIndex(problem, index);
    if (!constraint || !requiresDivisibleGuard(*constraint))
      continue;

    ScheduleGuard guard;
    guard.kind = GuardKind::DivisibleBy;
    guard.axisDomain = GuardAxisDomain::LogicalAxis;
    guard.dim = index;
    guard.value = tileSize;
    guard.text = (llvm::Twine("a") + llvm::Twine(index) + " % " +
                  llvm::Twine(tileSize) + " == 0")
                     .str();
    guards.push_back(std::move(guard));
  }
}

unsigned getGuardCount(const ScheduleInstance &instance) {
  return instance.candidateGuards.size() + instance.decisionGuards.size();
}

ScheduleInstance makeInstance(const ScheduleProblem &problem,
                              const ScheduleTemplate &tmpl,
                              TileShape tileShape) {
  ScheduleInstance instance;
  instance.tmpl = tmpl;
  instance.tileShape = std::move(tileShape);
  appendCandidateGuards(problem, instance.tileShape, instance.candidateGuards);
  appendDecisionGuards(problem, instance.tileShape.tileSizes,
                       instance.decisionGuards);
  if (hasDynamicTileSize(instance.tileShape))
    instance.reasonKinds.push_back("dynamic_tile");
  if (isBoundedParallelReductionTile(problem, instance.tileShape))
    instance.reasonKinds.push_back("bounded_parallel_reduction_tile");
  if (isBoundedParallelVectorTile(problem, instance.tileShape))
    instance.reasonKinds.push_back("bounded_parallel_vector_tile");
  instance.reasonKinds.push_back("cost_search");
  instance.estimatedCost = estimateScheduleCost(instance);
  return instance;
}

void assignInstanceIds(StringRef kernelId,
                       SmallVectorImpl<ScheduleInstance> &instances) {
  for (auto [index, instance] : llvm::enumerate(instances)) {
    instance.instanceId = (llvm::Twine(kernelId) + "." + instance.tmpl.family +
                           "." + llvm::Twine(index))
                              .str();
  }
}

} // namespace

ScheduleSearchResult searchScheduleInstancesWithStats(
    const ScheduleProblem &problem,
    ArrayRef<const ScheduleTemplateImplementation *> templates,
    const ScheduleSearchOptions &options) {
  SmallVector<ScheduleInstance, 8> generatedInstances;
  ScheduleSearchResult result;
  for (const ScheduleTemplateImplementation *implementation : templates) {
    SmallVector<TileShape> tileShapes =
        implementation->generateTileShapes(problem, options);
    for (TileShape &tileShape : tileShapes) {
      ++result.generatedCount;
      ScheduleInstance instance = makeInstance(
          problem, implementation->metadata(), std::move(tileShape));
      if (getGuardCount(instance) > problem.guardBudget) {
        ++result.prunedByGuardBudget;
        continue;
      }
      generatedInstances.push_back(std::move(instance));
    }
  }

  llvm::sort(generatedInstances, isLowerRankedInstance);

  unsigned keepCount =
      std::min<unsigned>(options.compileTimeTopK, generatedInstances.size());
  result.keptInstances.append(generatedInstances.begin(),
                              generatedInstances.begin() + keepCount);
  assignInstanceIds(problem.kernelId, result.keptInstances);
  return result;
}

SmallVector<ScheduleInstance, 4> searchScheduleInstances(
    const ScheduleProblem &problem,
    ArrayRef<const ScheduleTemplateImplementation *> templates,
    const ScheduleSearchOptions &options) {
  return searchScheduleInstancesWithStats(problem, templates, options)
      .keptInstances;
}

void printScheduleSearchReport(StringRef kernelId, unsigned generatedCount,
                               const ScheduleSearchOptions &options,
                               ArrayRef<ScheduleInstance> keptInstances,
                               llvm::raw_ostream &os) {
  os << "ScheduleSearch:\n";
  os << "  kernel = " << kernelId << "\n";
  os << "  generated = " << generatedCount << "\n";
  os << "  kept = " << keptInstances.size() << "\n";
  os << "  compile_time_top_k = " << options.compileTimeTopK << "\n";
  for (const ScheduleInstance &instance : keptInstances)
    os << "  instance = " << instance.instanceId << "\n";
}

void printGuardTexts(const ScheduleInstance &instance, llvm::raw_ostream &os) {
  for (const ScheduleGuard &guard : instance.candidateGuards)
    os << "  candidate_guard = " << guard.text << "\n";
  for (const ScheduleGuard &guard : instance.decisionGuards)
    os << "  decision_guard = " << guard.text << "\n";
}

void printScheduleGuardsReport(const ScheduleProblem &problem,
                               const ScheduleSearchResult &result,
                               llvm::raw_ostream &os) {
  os << "ScheduleGuards:\n";
  os << "  kernel = " << problem.kernelId << "\n";
  if (result.keptInstances.empty()) {
    os << "  selected_instance = <none>\n";
    os << "  candidate_guards = 0\n";
    os << "  decision_guards = 0\n";
  } else {
    const ScheduleInstance &selectedInstance = result.keptInstances.front();
    os << "  candidate_guards = "
       << selectedInstance.candidateGuards.size() << "\n";
    os << "  decision_guards = " << selectedInstance.decisionGuards.size()
       << "\n";
  }
  os << "  guard_budget = " << problem.guardBudget << "\n";
  os << "  pruned_by_guard_budget = " << result.prunedByGuardBudget << "\n";
  if (!result.keptInstances.empty()) {
    printGuardTexts(result.keptInstances.front(), os);
    for (const ScheduleInstance &instance :
         llvm::drop_begin(result.keptInstances)) {
      os << "  kept_instance = " << instance.instanceId << "\n";
      printGuardTexts(instance, os);
    }
  }
}

} // namespace mlir::ascend::schedule
