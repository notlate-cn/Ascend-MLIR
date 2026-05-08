//===- ScheduleSearch.cpp - Ascend V2 schedule search --------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Schedule/ScheduleSearch.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <utility>

using namespace mlir;

namespace mlir::afir::ascend::v2::schedule {
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

TileShape getFullLogicalAxisTile(const CoalescedAxisInfo &axes) {
  TileShape tileShape;
  for (const LogicalAxisInfo &axis : axes.logicalAxes)
    tileShape.tileSizes.push_back(normalizeExtent(axis.staticExtent));
  return tileShape;
}

FailureOr<TileShape> getSplitReductionTile(const CoalescedAxisInfo &axes) {
  TileShape tileShape = getFullLogicalAxisTile(axes);
  bool changed = false;
  for (unsigned reductionAxisId : axes.reductionAxes) {
    const LogicalAxisInfo *axis = lookupAxis(axes, reductionAxisId);
    if (!axis || ShapedType::isDynamic(axis->staticExtent) ||
        axis->staticExtent < 2)
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

SmallVector<TileShape> generateTileShapes(const ScheduleProblem &problem) {
  SmallVector<TileShape> tileShapes;
  switch (problem.dominantRole) {
  case OpRole::Vector:
    appendUniqueTileShape(tileShapes, getResultTile(problem.resultShape));
    appendUniqueTileShape(tileShapes, getHalfResultTile(problem.resultShape));
    break;
  case OpRole::Reduction:
    appendUniqueTileShape(tileShapes, getFullLogicalAxisTile(problem.axes));
    if (FailureOr<TileShape> splitTile = getSplitReductionTile(problem.axes);
        succeeded(splitTile))
      appendUniqueTileShape(tileShapes, std::move(*splitTile));
    break;
  case OpRole::Cube:
    appendUniqueTileShape(tileShapes, getResultTile(problem.resultShape));
    break;
  case OpRole::Memory:
  case OpRole::Unknown:
    break;
  }

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

int64_t estimateDebugCost(const TileShape &tileShape) {
  constexpr int64_t kDynamicPenalty = 1'000'000'000;
  constexpr int64_t kAreaWeight = 1024;
  return countDynamicTileSizes(tileShape.tileSizes) * kDynamicPenalty -
         getStaticTileArea(tileShape.tileSizes) * kAreaWeight +
         static_cast<int64_t>(tileShape.tileSizes.size());
}

bool isLowerRankedInstance(const ScheduleInstance &lhs,
                           const ScheduleInstance &rhs) {
  unsigned lhsDynamicCount = countDynamicTileSizes(lhs.tileShape.tileSizes);
  unsigned rhsDynamicCount = countDynamicTileSizes(rhs.tileShape.tileSizes);
  if (lhsDynamicCount != rhsDynamicCount)
    return lhsDynamicCount < rhsDynamicCount;

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

ScheduleInstance makeInstance(const ScheduleTemplate &tmpl,
                              TileShape tileShape) {
  ScheduleInstance instance;
  instance.tmpl = tmpl;
  instance.tileShape = std::move(tileShape);
  // Coarse diagnostic scalar only. Schedule ordering is the explicit
  // lexicographic key in isLowerRankedInstance.
  instance.estimatedCost = estimateDebugCost(instance.tileShape);
  if (hasDynamicTileSize(instance.tileShape))
    instance.reasonKinds.push_back("dynamic_tile");
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
  for (const ScheduleTemplate &tmpl : templates) {
    SmallVector<TileShape> tileShapes = generateTileShapes(problem);
    for (TileShape &tileShape : tileShapes)
      generatedInstances.push_back(makeInstance(tmpl, std::move(tileShape)));
  }

  llvm::sort(generatedInstances, isLowerRankedInstance);

  ScheduleSearchResult result;
  result.generatedCount = generatedInstances.size();
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

} // namespace mlir::afir::ascend::v2::schedule
