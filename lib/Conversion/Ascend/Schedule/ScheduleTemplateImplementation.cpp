//===- ScheduleTemplateImplementation.cpp - Schedule templates ------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "ScheduleTemplateImplementation.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <optional>
#include <utility>

using namespace mlir;

namespace mlir::ascend::schedule {
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
      int64_t chosenTile = normalizeExtent(axis.staticExtent);
      if (ShapedType::isDynamic(axis.staticExtent)) {
        chosenTile = remainingBudget;
      } else if (remainingBudget > 0) {
        chosenTile = std::max<int64_t>(
            1, std::min<int64_t>(axis.staticExtent, remainingBudget));
      }
      candidate.tileSizes[*axisIndex] = chosenTile;

      if (!ShapedType::isDynamic(chosenTile) && chosenTile > 0)
        remainingBudget = std::max<int64_t>(1, remainingBudget / chosenTile);
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

SmallVector<TileShape> generateRoleDrivenTileShapes(
    const ScheduleProblem &problem, const ScheduleSearchOptions &options) {
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

class RoleDrivenTemplateImplementation
    : public ScheduleTemplateImplementation {
public:
  RoleDrivenTemplateImplementation(ScheduleTemplate tmpl,
                                   std::string implementationKind,
                                   std::string description)
      : tmpl(std::move(tmpl)), implementationKindText(
                                  std::move(implementationKind)),
        descriptionText(std::move(description)) {}

  const ScheduleTemplate &metadata() const override { return tmpl; }
  llvm::StringRef implementationKind() const override {
    return implementationKindText;
  }
  llvm::StringRef description() const override { return descriptionText; }

  SmallVector<TileShape>
  generateTileShapes(const ScheduleProblem &problem,
                     const ScheduleSearchOptions &options) const override {
    return generateRoleDrivenTileShapes(problem, options);
  }

private:
  ScheduleTemplate tmpl;
  std::string implementationKindText;
  std::string descriptionText;
};

class SingleTilePerBlockTemplateImplementation final
    : public RoleDrivenTemplateImplementation {
public:
  explicit SingleTilePerBlockTemplateImplementation(ScheduleTemplate tmpl)
      : RoleDrivenTemplateImplementation(std::move(tmpl),
                                         kScheduleTemplateSingleTilePerBlock.str(),
                                         "one runtime tile per block") {}
};

class GroupedTilePerBlockTemplateImplementation final
    : public RoleDrivenTemplateImplementation {
public:
  explicit GroupedTilePerBlockTemplateImplementation(ScheduleTemplate tmpl)
      : RoleDrivenTemplateImplementation(std::move(tmpl),
                                         kScheduleTemplateGroupedTilePerBlock.str(),
                                         "grouped runtime tiles per block") {}
};

} // namespace

std::unique_ptr<ScheduleTemplateImplementation>
createSingleTilePerBlockTemplateImplementation(StringRef family,
                                               StringRef roleTag,
                                               unsigned minRank,
                                               unsigned maxRank,
                                               unsigned priority) {
  ScheduleTemplate tmpl;
  tmpl.family = family.str();
  tmpl.name = kScheduleTemplateSingleTilePerBlock.str();
  tmpl.tags.push_back(roleTag.str());
  tmpl.minRank = minRank;
  tmpl.maxRank = maxRank;
  tmpl.priority = priority;
  return std::make_unique<SingleTilePerBlockTemplateImplementation>(
      std::move(tmpl));
}

std::unique_ptr<ScheduleTemplateImplementation>
createGroupedTilePerBlockTemplateImplementation(ScheduleTemplate tmpl) {
  return std::make_unique<GroupedTilePerBlockTemplateImplementation>(
      std::move(tmpl));
}

std::unique_ptr<ScheduleTemplateImplementation>
createRoleDrivenTemplateImplementation(ScheduleTemplate tmpl,
                                       StringRef description) {
  return std::make_unique<RoleDrivenTemplateImplementation>(
      std::move(tmpl), tmpl.name, description.str());
}

} // namespace mlir::ascend::schedule
