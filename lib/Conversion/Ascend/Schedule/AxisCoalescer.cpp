//===- AxisCoalescer.cpp - Ascend logical axis coalescing --------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Schedule/AxisCoalescer.h"

#include "Conversion/Ascend/Schedule/KernelPatternView.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <optional>
#include <utility>

using namespace mlir;

namespace mlir::afir::ascend::schedule {
namespace {

void addBarrier(CoalescedAxisInfo &info, Operation *op,
                AxisBarrierKind barrierKind, std::string reason) {
  info.barriers.push_back(
      AxisCoalescingBarrier{op, barrierKind, std::move(reason)});
}

AxisKind classifyIteratorType(utils::IteratorType iteratorType,
                              CoalescedAxisInfo &info, Operation *op,
                              unsigned axis) {
  if (iteratorType == utils::IteratorType::parallel)
    return AxisKind::Parallel;
  if (iteratorType == utils::IteratorType::reduction)
    return AxisKind::Reduction;

  addBarrier(info, op, AxisBarrierKind::UnsupportedIteratorType,
             (llvm::Twine("unsupported iterator type at axis ") +
              llvm::Twine(axis))
                 .str());
  return AxisKind::Unknown;
}

std::optional<unsigned> getFirstResultRank(Operation *op) {
  if (op->getNumResults() == 0)
    return std::nullopt;

  auto shapedType = dyn_cast<ShapedType>(op->getResult(0).getType());
  if (!shapedType || !shapedType.hasRank())
    return std::nullopt;

  return shapedType.getRank();
}

unsigned getAxisCount(Operation *op, OpRole primaryRole,
                      unsigned iteratorCount) {
  if (primaryRole == OpRole::Reduction)
    return iteratorCount;

  if (primaryRole == OpRole::Vector || primaryRole == OpRole::Cube) {
    if (std::optional<unsigned> resultRank = getFirstResultRank(op))
      return std::max(*resultRank, iteratorCount);
  }

  return iteratorCount;
}

void mergeStaticExtent(SmallVectorImpl<int64_t> &staticExtents, unsigned axis,
                       int64_t extent) {
  if (axis >= staticExtents.size() || extent == ShapedType::kDynamic)
    return;

  if (staticExtents[axis] == ShapedType::kDynamic ||
      staticExtents[axis] == extent)
    staticExtents[axis] = extent;
}

void collectIndexingMapInfo(linalg::LinalgOp linalgOp, OpRole patternRole,
                            unsigned axisCount,
                            SmallVectorImpl<int64_t> &staticExtents,
                            SmallVectorImpl<bool> &broadcastAxes,
                            CoalescedAxisInfo &info) {
  Operation *op = linalgOp.getOperation();
  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  OperandRange operands = op->getOperands();

  unsigned mapCount = static_cast<unsigned>(indexingMaps.size());
  unsigned operandCount = static_cast<unsigned>(operands.size());
  if (mapCount != operandCount) {
    addBarrier(info, op, AxisBarrierKind::RankMismatch,
               (llvm::Twine("indexing map count ") + llvm::Twine(mapCount) +
                " does not match operand count " + llvm::Twine(operandCount))
                   .str());
  }

  unsigned count = std::min(mapCount, operandCount);
  unsigned inputMapCount =
      std::min<unsigned>(static_cast<unsigned>(linalgOp.getNumDpsInputs()),
                         count);
  for (unsigned mapIndex = 0; mapIndex < count; ++mapIndex) {
    AffineMap map = indexingMaps[mapIndex];
    if (!map.isProjectedPermutation()) {
      addBarrier(info, op, AxisBarrierKind::UnsupportedIndexingMap,
                 (llvm::Twine("non-projected-permutation indexing map ") +
                  llvm::Twine(mapIndex))
                     .str());
      continue;
    }

    auto shapedType = dyn_cast<ShapedType>(operands[mapIndex].getType());
    if (!shapedType || !shapedType.hasRank())
      continue;

    if (map.getNumResults() != shapedType.getRank()) {
      addBarrier(info, op, AxisBarrierKind::RankMismatch,
                 (llvm::Twine("indexing map ") + llvm::Twine(mapIndex) +
                  " result rank " + llvm::Twine(map.getNumResults()) +
                  " does not match operand rank " +
                  llvm::Twine(shapedType.getRank()))
                     .str());
      continue;
    }

    SmallVector<bool> usedAxes(axisCount, false);
    for (auto [resultIndex, expr] : llvm::enumerate(map.getResults())) {
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr)
        continue;

      unsigned axis = dimExpr.getPosition();
      if (axis >= axisCount) {
        addBarrier(info, op, AxisBarrierKind::RankMismatch,
                   (llvm::Twine("indexing map ") + llvm::Twine(mapIndex) +
                    " references axis " + llvm::Twine(axis) +
                    " outside axis count " + llvm::Twine(axisCount))
                       .str());
        continue;
      }

      usedAxes[axis] = true;
      mergeStaticExtent(staticExtents, axis,
                        shapedType.getDimSize(resultIndex));
    }

    if (patternRole == OpRole::Vector && mapIndex < inputMapCount) {
      for (unsigned axis = 0; axis < axisCount; ++axis) {
        if (!usedAxes[axis])
          broadcastAxes[axis] = true;
      }
    }
  }
}

void appendPatternRawAxes(const KernelPatternView &pattern, unsigned axisCount,
                          CoalescedAxisInfo &info) {
  for (const PatternOpView &opView : pattern.ops) {
    auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(opView.op);
    if (!linalgOp)
      continue;

    unsigned iteratorCount =
        static_cast<unsigned>(linalgOp.getIteratorTypesArray().size());
    unsigned rawAxisCount = std::min(axisCount, iteratorCount);
    for (unsigned axis = 0; axis < rawAxisCount; ++axis)
      info.logicalAxes[axis].rawAxes.push_back({opView.op, axis});

    if (iteratorCount < axisCount) {
      addBarrier(info, opView.op, AxisBarrierKind::RankMismatch,
                 (llvm::Twine("op has ") + llvm::Twine(iteratorCount) +
                  " iterator axes but pattern has " + llvm::Twine(axisCount))
                     .str());
    }
  }
}

void printAxisList(ArrayRef<unsigned> axes, llvm::raw_ostream &os) {
  os << "[";
  llvm::interleaveComma(axes, os);
  os << "]";
}

} // namespace

FailureOr<CoalescedAxisInfo> coalesceAxes(const KernelPatternView &pattern) {
  const PatternOpView *primaryOpView = selectDominantPrimaryOp(pattern);
  if (!primaryOpView || !primaryOpView->op)
    return failure();

  Operation *primaryOp = primaryOpView->op;
  auto linalgOp = dyn_cast<linalg::LinalgOp>(primaryOp);
  if (!linalgOp) {
    primaryOp->emitError() << "axis coalescing requires a linalg primary op";
    return failure();
  }

  CoalescedAxisInfo info;
  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();
  unsigned axisCount = getAxisCount(primaryOp, primaryOpView->role,
                                   static_cast<unsigned>(iteratorTypes.size()));
  SmallVector<int64_t> staticExtents(axisCount, ShapedType::kDynamic);
  SmallVector<bool> broadcastAxisMask(axisCount, false);

  for (const PatternOpView &opView : pattern.ops) {
    auto patternLinalgOp = dyn_cast_or_null<linalg::LinalgOp>(opView.op);
    if (!patternLinalgOp) {
      addBarrier(info, opView.op, AxisBarrierKind::RankMismatch,
                 "kernel pattern contains a non-linalg op");
      continue;
    }
    collectIndexingMapInfo(patternLinalgOp, primaryOpView->role, axisCount,
                           staticExtents, broadcastAxisMask, info);
  }

  info.logicalAxes.reserve(axisCount);
  for (unsigned axis = 0; axis < axisCount; ++axis) {
    AxisKind kind = AxisKind::Unknown;
    if (axis < iteratorTypes.size())
      kind = classifyIteratorType(iteratorTypes[axis], info, primaryOp, axis);
    else
      addBarrier(info, primaryOp, AxisBarrierKind::RankMismatch,
                 (llvm::Twine("missing iterator type for axis ") +
                  llvm::Twine(axis))
                     .str());

    LogicalAxisInfo axisInfo;
    axisInfo.logicalAxisId = axis;
    axisInfo.kind = kind;
    axisInfo.staticExtent = staticExtents[axis];
    info.logicalAxes.push_back(std::move(axisInfo));

    switch (kind) {
    case AxisKind::Parallel:
      info.parallelAxes.push_back(axis);
      break;
    case AxisKind::Reduction:
      info.reductionAxes.push_back(axis);
      break;
    case AxisKind::Unknown:
    default:
      break;
    }

    if (broadcastAxisMask[axis])
      info.broadcastAxes.push_back(axis);
  }

  appendPatternRawAxes(pattern, axisCount, info);

  return info;
}

void printAxisCoalescingReport(StringRef kernelId,
                               const CoalescedAxisInfo &info,
                               llvm::raw_ostream &os) {
  os << "AxisCoalescing:\n";
  os << "  kernel = " << kernelId << "\n";
  os << "  logical_axes = " << info.logicalAxes.size() << "\n";
  os << "  parallel_axes = ";
  printAxisList(info.parallelAxes, os);
  os << "\n";
  os << "  reduction_axes = ";
  printAxisList(info.reductionAxes, os);
  os << "\n";
  os << "  broadcast_axes = ";
  printAxisList(info.broadcastAxes, os);
  os << "\n";
  os << "  barriers = " << info.barriers.size() << "\n";
}

} // namespace mlir::afir::ascend::schedule
