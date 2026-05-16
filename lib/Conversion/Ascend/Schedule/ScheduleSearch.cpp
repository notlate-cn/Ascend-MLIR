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

namespace mlir::afir::ascend::schedule {
namespace {

bool hasTileShape(ArrayRef<TileShape> tileShapes, const TileShape &candidate) {
  return llvm::any_of(tileShapes, [&](const TileShape &tileShape) {
    return tileShape.tileSizes == candidate.tileSizes;
  });
}

void appendUniqueTileShape(SmallVectorImpl<TileShape> &tileShapes,
                           TileShape candidate) {
  if (hasTileShape(tileShapes, candidate))
    return;
  tileShapes.push_back(std::move(candidate));
}

int64_t normalizeExtent(int64_t extent) {
  if (ShapedType::isDynamic(extent))
    return ShapedType::kDynamic;
  return extent;
}

TileShape getResultTile(ArrayRef<int64_t> shape) {
  TileShape tileShape;
  for (int64_t dim : shape)
    tileShape.tileSizes.push_back(normalizeExtent(dim));
  return tileShape;
}

TileShape getHalfResultTile(ArrayRef<int64_t> shape) {
  TileShape tileShape;
  for (int64_t dim : shape) {
    if (ShapedType::isDynamic(dim)) {
      tileShape.tileSizes.push_back(ShapedType::kDynamic);
      continue;
    }
    tileShape.tileSizes.push_back(dim >= 2 ? dim / 2 : dim);
  }
  return tileShape;
}

const LogicalAxisInfo *lookupAxis(const CoalescedAxisInfo &axes,
                                  unsigned logicalAxisId) {
  for (const LogicalAxisInfo &axis : axes.logicalAxes) {
    if (axis.logicalAxisId == logicalAxisId)
      return &axis;
  }
  return nullptr;
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

TileShape getFullLogicalAxisTile(const CoalescedAxisInfo &axes) {
  TileShape tileShape;
  for (const LogicalAxisInfo &axis : axes.logicalAxes)
    tileShape.tileSizes.push_back(normalizeExtent(axis.staticExtent));
  return tileShape;
}

int64_t getDefaultParallelTile(const ScheduleProblem &problem) {
  return problem.targetTilePolicy.defaultParallelTile;
}

void appendUniqueTileSize(SmallVectorImpl<int64_t> &tileSizes,
                          int64_t tileSize) {
  if (tileSize == 0)
    return;
  if (!llvm::is_contained(tileSizes, tileSize))
    tileSizes.push_back(tileSize);
}

int64_t capTileSizeToExtent(int64_t extent, int64_t tileSize) {
  if (ShapedType::isDynamic(extent))
    return tileSize > 0 ? tileSize : ShapedType::kDynamic;
  if (tileSize <= 0)
    return extent;
  return std::max<int64_t>(1, std::min(extent, tileSize));
}

bool canTileParallelAxis(const AxisScheduleConstraint *constraint) {
  return constraint && constraint->kind == AxisKind::Parallel &&
         (hasAxisExecutionRole(constraint->allowedRoles,
                               AxisExecutionRole::BindCoreCandidate) ||
          hasAxisExecutionRole(constraint->allowedRoles,
                               AxisExecutionRole::KernelLoopCandidate) ||
          hasAxisExecutionRole(constraint->allowedRoles,
                               AxisExecutionRole::VectorizeCandidate));
}

SmallVector<int64_t, 6>
getParallelAxisTileSizes(const LogicalAxisInfo &axis,
                         const AxisScheduleConstraint *constraint,
                         const ScheduleProblem &problem) {
  SmallVector<int64_t, 6> tileSizes;
  int64_t extent = normalizeExtent(axis.staticExtent);
  appendUniqueTileSize(tileSizes, extent);
  if (!canTileParallelAxis(constraint))
    return tileSizes;

  int64_t defaultParallelTile = getDefaultParallelTile(problem);
  appendUniqueTileSize(tileSizes,
                       capTileSizeToExtent(axis.staticExtent,
                                           defaultParallelTile * 4));
  appendUniqueTileSize(tileSizes,
                       capTileSizeToExtent(axis.staticExtent,
                                           defaultParallelTile * 2));
  appendUniqueTileSize(tileSizes,
                       capTileSizeToExtent(axis.staticExtent,
                                           defaultParallelTile));
  appendUniqueTileSize(tileSizes,
                       capTileSizeToExtent(axis.staticExtent,
                                           defaultParallelTile / 2));
  appendUniqueTileSize(tileSizes,
                       capTileSizeToExtent(axis.staticExtent,
                                           defaultParallelTile / 4));

  if (!ShapedType::isDynamic(axis.staticExtent)) {
    appendUniqueTileSize(tileSizes,
                         capTileSizeToExtent(axis.staticExtent,
                                             axis.staticExtent / 2));
    appendUniqueTileSize(tileSizes,
                         capTileSizeToExtent(axis.staticExtent,
                                             axis.staticExtent / 4));
    appendUniqueTileSize(tileSizes, 1);
  }

  return tileSizes;
}

SmallVector<int64_t, 4>
getAxisTileSizes(const LogicalAxisInfo &axis,
                 const AxisScheduleConstraint *constraint,
                 const ScheduleProblem &problem) {
  if (axis.kind == AxisKind::Parallel)
    return getParallelAxisTileSizes(axis, constraint, problem);

  SmallVector<int64_t, 4> tileSizes;
  appendUniqueTileSize(tileSizes, normalizeExtent(axis.staticExtent));
  return tileSizes;
}

void appendAxisProductTileShapes(const ScheduleProblem &problem,
                                 SmallVectorImpl<TileShape> &tileShapes,
                                 unsigned maxTileShapes) {
  const CoalescedAxisInfo &axes = problem.axes;
  if (axes.logicalAxes.empty())
    return;

  SmallVector<SmallVector<int64_t, 6>, 4> perAxisTileSizes;
  perAxisTileSizes.reserve(axes.logicalAxes.size());
  for (const LogicalAxisInfo &axis : axes.logicalAxes) {
    const AxisScheduleConstraint *constraint =
        lookupAxisScheduleConstraint(axes, axis.logicalAxisId);
    perAxisTileSizes.push_back(getAxisTileSizes(axis, constraint, problem));
  }

  SmallVector<TileShape, 8> worklist(1);
  for (ArrayRef<int64_t> axisTileSizes : perAxisTileSizes) {
    SmallVector<TileShape, 8> next;
    for (const TileShape &partial : worklist) {
      for (int64_t axisTileSize : axisTileSizes) {
        TileShape candidate = partial;
        candidate.tileSizes.push_back(axisTileSize);
        next.push_back(std::move(candidate));
        if (next.size() >= maxTileShapes)
          break;
      }
      if (next.size() >= maxTileShapes)
        break;
    }
    worklist = std::move(next);
  }

  for (TileShape &candidate : worklist)
    appendUniqueTileShape(tileShapes, std::move(candidate));
}

std::optional<unsigned> getLogicalAxisIndex(const CoalescedAxisInfo &axes,
                                            unsigned logicalAxisId) {
  for (auto [index, axis] : llvm::enumerate(axes.logicalAxes))
    if (axis.logicalAxisId == logicalAxisId)
      return static_cast<unsigned>(index);
  return std::nullopt;
}

void appendCoalescingHintTileShapes(const ScheduleProblem &problem,
                                    SmallVectorImpl<TileShape> &tileShapes) {
  const CoalescedAxisInfo &axes = problem.axes;
  if (axes.axisCoalescingHints.empty() || axes.logicalAxes.empty())
    return;

  int64_t groupBudget = std::max<int64_t>(
      1, problem.targetTilePolicy.defaultParallelTile *
             std::max<int64_t>(1, problem.targetTilePolicy.vectorBufferCount));
  for (const AxisCoalescingHint &hint : axes.axisCoalescingHints) {
    if (hint.memberAxisIds.size() < 2)
      continue;

    TileShape candidate = getFullLogicalAxisTile(axes);
    int64_t remainingBudget = groupBudget;
    for (unsigned memberAxisId : llvm::reverse(hint.memberAxisIds)) {
      std::optional<unsigned> axisIndex =
          getLogicalAxisIndex(axes, memberAxisId);
      if (!axisIndex)
        continue;

      const LogicalAxisInfo &axis = axes.logicalAxes[*axisIndex];
      int64_t selectedTile = normalizeExtent(axis.staticExtent);
      if (ShapedType::isDynamic(axis.staticExtent)) {
        selectedTile = remainingBudget;
      } else if (remainingBudget > 0) {
        selectedTile = std::max<int64_t>(
            1, std::min<int64_t>(axis.staticExtent, remainingBudget));
      }
      candidate.tileSizes[*axisIndex] = selectedTile;

      if (!ShapedType::isDynamic(selectedTile) && selectedTile > 0)
        remainingBudget = std::max<int64_t>(1, remainingBudget / selectedTile);
    }

    appendUniqueTileShape(tileShapes, std::move(candidate));
  }
}

TileShape getRoleDrivenReductionTile(const ScheduleProblem &problem) {
  const CoalescedAxisInfo &axes = problem.axes;
  int64_t defaultParallelTile = getDefaultParallelTile(problem);
  TileShape tileShape;
  for (const LogicalAxisInfo &axis : axes.logicalAxes) {
    int64_t tileSize = normalizeExtent(axis.staticExtent);
    const AxisScheduleConstraint *constraint =
        lookupAxisScheduleConstraint(axes, axis.logicalAxisId);
    if (!constraint) {
      tileShape.tileSizes.push_back(tileSize);
      continue;
    }

    if (hasAxisExecutionRole(constraint->allowedRoles,
                             AxisExecutionRole::FullReduction)) {
      tileShape.tileSizes.push_back(tileSize);
      continue;
    }

    if (constraint->kind == AxisKind::Parallel &&
        (hasAxisExecutionRole(constraint->allowedRoles,
                              AxisExecutionRole::BindCoreCandidate) ||
         hasAxisExecutionRole(constraint->allowedRoles,
                              AxisExecutionRole::KernelLoopCandidate))) {
      if (ShapedType::isDynamic(axis.staticExtent))
        tileSize = defaultParallelTile;
      else
        tileSize = std::min(axis.staticExtent, defaultParallelTile);
    }

    tileShape.tileSizes.push_back(tileSize);
  }
  return tileShape;
}

FailureOr<TileShape> getSplitReductionTile(const CoalescedAxisInfo &axes) {
  TileShape tileShape = getFullLogicalAxisTile(axes);
  bool changed = false;
  for (unsigned reductionAxisId : axes.reductionAxes) {
    const LogicalAxisInfo *axis = lookupAxis(axes, reductionAxisId);
    if (!axis || ShapedType::isDynamic(axis->staticExtent) ||
        axis->staticExtent < 2 || axis->staticExtent % 2 != 0)
      continue;

    for (auto [index, logicalAxis] : llvm::enumerate(axes.logicalAxes)) {
      if (logicalAxis.logicalAxisId != reductionAxisId)
        continue;
      tileShape.tileSizes[index] = axis->staticExtent / 2;
      changed = true;
      break;
    }
  }

  if (!changed)
    return failure();
  return tileShape;
}

TileShape getRoleDrivenVectorTile(const ScheduleProblem &problem) {
  const CoalescedAxisInfo &axes = problem.axes;
  int64_t defaultParallelTile = getDefaultParallelTile(problem);
  TileShape tileShape = getFullLogicalAxisTile(axes);
  if (axes.logicalAxes.size() < 2)
    return tileShape;

  bool selectedOuterParallelTile = false;
  for (auto [index, axis] : llvm::enumerate(axes.logicalAxes)) {
    const AxisScheduleConstraint *constraint =
        lookupAxisScheduleConstraint(axes, axis.logicalAxisId);
    if (!constraint || constraint->kind != AxisKind::Parallel)
      continue;
    if (selectedOuterParallelTile)
      continue;
    if (!hasAxisExecutionRole(constraint->allowedRoles,
                              AxisExecutionRole::BindCoreCandidate) &&
        !hasAxisExecutionRole(constraint->allowedRoles,
                              AxisExecutionRole::KernelLoopCandidate))
      continue;

    if (ShapedType::isDynamic(axis.staticExtent))
      tileShape.tileSizes[index] = defaultParallelTile;
    else
      tileShape.tileSizes[index] =
          std::min(axis.staticExtent, defaultParallelTile);
    selectedOuterParallelTile = true;
  }
  return tileShape;
}

TileShape getRoleDrivenCubeTile(const ScheduleProblem &problem) {
  const CoalescedAxisInfo &axes = problem.axes;
  int64_t defaultParallelTile = getDefaultParallelTile(problem);
  TileShape tileShape = getFullLogicalAxisTile(axes);
  for (auto [index, axis] : llvm::enumerate(axes.logicalAxes)) {
    const AxisScheduleConstraint *constraint =
        lookupAxisScheduleConstraint(axes, axis.logicalAxisId);
    if (!constraint || constraint->kind != AxisKind::Parallel)
      continue;
    if (!hasAxisExecutionRole(constraint->allowedRoles,
                              AxisExecutionRole::BindCoreCandidate) &&
        !hasAxisExecutionRole(constraint->allowedRoles,
                              AxisExecutionRole::KernelLoopCandidate))
      continue;

    if (ShapedType::isDynamic(axis.staticExtent))
      tileShape.tileSizes[index] = defaultParallelTile;
    else
      tileShape.tileSizes[index] =
          std::min(axis.staticExtent, defaultParallelTile);
    break;
  }
  return tileShape;
}

SmallVector<TileShape> generateTileShapes(const ScheduleProblem &problem,
                                          const ScheduleSearchOptions &options) {
  SmallVector<TileShape> tileShapes;
  switch (problem.dominantRole) {
  case OpRole::Vector:
    appendUniqueTileShape(tileShapes, getRoleDrivenVectorTile(problem));
    appendUniqueTileShape(tileShapes, getResultTile(problem.resultShape));
    appendUniqueTileShape(tileShapes, getHalfResultTile(problem.resultShape));
    break;
  case OpRole::Reduction:
    appendUniqueTileShape(tileShapes, getRoleDrivenReductionTile(problem));
    appendUniqueTileShape(tileShapes, getFullLogicalAxisTile(problem.axes));
    if (FailureOr<TileShape> splitTile = getSplitReductionTile(problem.axes);
        succeeded(splitTile))
      appendUniqueTileShape(tileShapes, std::move(*splitTile));
    break;
  case OpRole::Cube:
    appendUniqueTileShape(tileShapes, getRoleDrivenCubeTile(problem));
    appendUniqueTileShape(tileShapes, getFullLogicalAxisTile(problem.axes));
    break;
  case OpRole::Memory:
    appendUniqueTileShape(tileShapes, getFullLogicalAxisTile(problem.axes));
    break;
  case OpRole::Unknown:
    break;
  }

  if (problem.dominantRole == OpRole::Memory ||
      problem.dominantRole == OpRole::Unknown)
    return tileShapes;

  appendAxisProductTileShapes(problem, tileShapes,
                              options.maxAxisProductTileShapes);
  appendCoalescingHintTileShapes(problem, tileShapes);
  return tileShapes;
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

void appendCandidateGuards(ArrayRef<int64_t> shape,
                           SmallVectorImpl<ScheduleGuard> &guards) {
  for (auto [index, dim] : llvm::enumerate(shape)) {
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
  appendCandidateGuards(problem.resultShape, instance.candidateGuards);
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
    const ScheduleProblem &problem, ArrayRef<ScheduleTemplate> templates,
    const ScheduleSearchOptions &options) {
  SmallVector<ScheduleInstance, 8> generatedInstances;
  ScheduleSearchResult result;
  for (const ScheduleTemplate &tmpl : templates) {
    SmallVector<TileShape> tileShapes = generateTileShapes(problem, options);
    for (TileShape &tileShape : tileShapes) {
      ++result.generatedCount;
      ScheduleInstance instance =
          makeInstance(problem, tmpl, std::move(tileShape));
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
    const ScheduleProblem &problem, ArrayRef<ScheduleTemplate> templates,
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

} // namespace mlir::afir::ascend::schedule
