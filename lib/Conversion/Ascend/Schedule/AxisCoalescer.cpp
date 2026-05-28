//===- AxisCoalescer.cpp - Ascend logical axis coalescing --------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "AxisCoalescer.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "KernelPatternView.h"
#include "Conversion/Ascend/Kernelize/Pattern/HandwrittenContractRegistry.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <optional>
#include <utility>

using namespace mlir;

namespace mlir::ascend::schedule {
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

LogicalResult mergeStaticExtent(SmallVectorImpl<int64_t> &staticExtents,
                                unsigned axis, int64_t extent,
                                CoalescedAxisInfo &info, Operation *op) {
  if (axis >= staticExtents.size() || extent == ShapedType::kDynamic)
    return success();

  if (staticExtents[axis] == ShapedType::kDynamic ||
      staticExtents[axis] == extent) {
    staticExtents[axis] = extent;
    return success();
  }

  std::string reason =
      (llvm::Twine("conflicting static extent for logical axis ") +
       llvm::Twine(axis) + ": " + llvm::Twine(staticExtents[axis]) + " vs " +
       llvm::Twine(extent))
          .str();
  addBarrier(info, op, AxisBarrierKind::RankMismatch, reason);
  op->emitError() << reason;
  return failure();
}

bool isDimOrConstantProjection(AffineMap map) {
  SmallVector<bool> seenDims(map.getNumDims(), false);
  for (AffineExpr expr : map.getResults()) {
    if (isa<AffineConstantExpr>(expr))
      continue;

    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      return false;

    unsigned position = dimExpr.getPosition();
    if (position >= seenDims.size() || seenDims[position])
      return false;
    seenDims[position] = true;
  }
  return true;
}

bool isPostReductionSingletonCarry(linalg::LinalgOp linalgOp,
                                   OpRole patternRole, unsigned axis,
                                   int64_t extent) {
  if (patternRole != OpRole::Reduction || extent != 1)
    return false;

  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();
  if (axis >= iteratorTypes.size())
    return false;

  return iteratorTypes[axis] != utils::IteratorType::reduction;
}

LogicalResult collectIndexingMapInfo(linalg::LinalgOp linalgOp,
                                     OpRole patternRole, unsigned axisCount,
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
    if (!isDimOrConstantProjection(map)) {
      addBarrier(info, op, AxisBarrierKind::UnsupportedIndexingMap,
                 (llvm::Twine("non-dim-or-constant-projection indexing map ") +
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
      if (isa<AffineConstantExpr>(expr))
        continue;

      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr) {
        addBarrier(info, op, AxisBarrierKind::UnsupportedIndexingMap,
                   (llvm::Twine("unsupported affine result in indexing map ") +
                    llvm::Twine(mapIndex))
                       .str());
        continue;
      }

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
      int64_t extent = shapedType.getDimSize(resultIndex);
      if (isPostReductionSingletonCarry(linalgOp, patternRole, axis, extent))
        continue;
      if (failed(mergeStaticExtent(staticExtents, axis, extent, info, op)))
        return failure();
    }

    if (patternRole == OpRole::Vector && mapIndex < inputMapCount) {
      for (unsigned axis = 0; axis < axisCount; ++axis) {
        if (!usedAxes[axis])
          broadcastAxes[axis] = true;
      }
    }
  }
  return success();
}

void appendPatternRawAxes(const KernelPatternView &pattern, unsigned axisCount,
                          CoalescedAxisInfo &info,
                          Operation *axisOnlyOp = nullptr) {
  for (const PatternOpView &opView : pattern.ops) {
    if (axisOnlyOp && opView.op != axisOnlyOp)
      continue;
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

void printCompactAxisList(ArrayRef<unsigned> axes, llvm::raw_ostream &os) {
  os << "[";
  llvm::interleave(axes, os, [&](unsigned axis) { os << axis; }, ",");
  os << "]";
}

void addRole(SmallVectorImpl<AxisExecutionRole> &roles,
             AxisExecutionRole role) {
  if (!llvm::is_contained(roles, role))
    roles.push_back(role);
}

void addTailPolicy(SmallVectorImpl<AxisTailPolicy> &policies,
                   AxisTailPolicy policy) {
  if (!llvm::is_contained(policies, policy))
    policies.push_back(policy);
}

void addPrimitiveUse(SmallVectorImpl<PrimitiveAxisUseKind> &uses,
                     PrimitiveAxisUseKind use) {
  if (!llvm::is_contained(uses, use))
    uses.push_back(use);
}

AxisTailPolicy getDefaultTailPolicy(ArrayRef<AxisTailPolicy> policies) {
  if (policies.empty())
    return AxisTailPolicy::MustDivide;
  return policies.front();
}

bool hasGatherIndexingMarker(Operation *op) {
  return op && (op->hasAttr(::mlir::ascend::kGatherDimAttr) ||
                op->hasAttr(::mlir::ascend::kEmbeddingDimAttr));
}

bool hasIntegerOrIndexElementType(Value value) {
  auto shapedType = dyn_cast<ShapedType>(value.getType());
  if (!shapedType)
    return false;
  return isa<IntegerType, IndexType>(shapedType.getElementType());
}

bool mapUsesRawAxis(AffineMap map, unsigned rawAxis) {
  for (AffineExpr expr : map.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (dimExpr && dimExpr.getPosition() == rawAxis)
      return true;
  }
  return false;
}

bool hasGatherUseOnAxis(const LogicalAxisInfo &axis) {
  for (auto [op, rawAxis] : axis.rawAxes) {
    if (!hasGatherIndexingMarker(op))
      continue;

    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp)
      continue;

    SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
    OperandRange operands = op->getOperands();
    unsigned inputMapCount =
        std::min<unsigned>(static_cast<unsigned>(linalgOp.getNumDpsInputs()),
                           indexingMaps.size());
    inputMapCount = std::min<unsigned>(inputMapCount, operands.size());

    for (unsigned mapIndex = 0; mapIndex < inputMapCount; ++mapIndex) {
      if (!hasIntegerOrIndexElementType(operands[mapIndex]))
        continue;
      if (mapUsesRawAxis(indexingMaps[mapIndex], rawAxis))
        return true;
    }
  }
  return false;
}

AxisScheduleConstraint *
lookupAxisScheduleConstraint(CoalescedAxisInfo &info, unsigned logicalAxisId) {
  for (AxisScheduleConstraint &constraint : info.axisScheduleConstraints) {
    if (constraint.logicalAxisId == logicalAxisId)
      return &constraint;
  }
  return nullptr;
}

void deriveAxisScheduleConstraints(CoalescedAxisInfo &info) {
  info.axisScheduleConstraints.clear();
  info.axisScheduleConstraints.reserve(info.logicalAxes.size());

  for (const LogicalAxisInfo &axis : info.logicalAxes) {
    AxisScheduleConstraint constraint;
    constraint.logicalAxisId = axis.logicalAxisId;
    constraint.kind = axis.kind;

    switch (axis.kind) {
    case AxisKind::Parallel:
      constraint.allowedRoles.push_back(AxisExecutionRole::BindCoreCandidate);
      constraint.allowedRoles.push_back(AxisExecutionRole::KernelLoopCandidate);
      constraint.allowedRoles.push_back(AxisExecutionRole::VectorizeCandidate);
      constraint.allowedTailPolicies.push_back(AxisTailPolicy::MaskedTail);
      constraint.allowedTailPolicies.push_back(AxisTailPolicy::ScalarEpilogue);
      constraint.primitiveUses.push_back(PrimitiveAxisUseKind::DataCopy);
      constraint.primitiveUses.push_back(PrimitiveAxisUseKind::VectorCompute);
      constraint.primitiveUses.push_back(PrimitiveAxisUseKind::WriteBack);
      break;
    case AxisKind::Reduction:
      constraint.allowedRoles.push_back(AxisExecutionRole::FullReduction);
      constraint.allowedTailPolicies.push_back(AxisTailPolicy::FullExtent);
      constraint.primitiveUses.push_back(PrimitiveAxisUseKind::Reduction);
      break;
    case AxisKind::Unknown:
      constraint.allowedTailPolicies.push_back(AxisTailPolicy::MustDivide);
      break;
    }

    if (axis.kind == AxisKind::Parallel && hasGatherUseOnAxis(axis)) {
      addTailPolicy(constraint.allowedTailPolicies,
                    AxisTailPolicy::PadAndMask);
      addPrimitiveUse(constraint.primitiveUses,
                      PrimitiveAxisUseKind::GatherIndex);
    }

    constraint.tailPolicy =
        getDefaultTailPolicy(constraint.allowedTailPolicies);
    info.axisScheduleConstraints.push_back(std::move(constraint));
  }

  for (unsigned axis : info.broadcastAxes) {
    AxisScheduleConstraint *constraint =
        lookupAxisScheduleConstraint(info, axis);
    if (!constraint || constraint->kind == AxisKind::Unknown)
      continue;
    addRole(constraint->allowedRoles,
            AxisExecutionRole::BroadcastProjection);
  }
}

SmallVector<unsigned, 4> getMaximalParallelRun(const CoalescedAxisInfo &info) {
  SmallVector<unsigned, 4> bestRun;
  SmallVector<unsigned, 4> currentRun;
  for (const LogicalAxisInfo &axis : info.logicalAxes) {
    if (axis.kind == AxisKind::Parallel) {
      currentRun.push_back(axis.logicalAxisId);
      continue;
    }

    if (currentRun.size() > bestRun.size())
      bestRun = currentRun;
    currentRun.clear();
  }

  if (currentRun.size() > bestRun.size())
    bestRun = currentRun;
  return bestRun;
}

void deriveAxisCoalescingHints(CoalescedAxisInfo &info) {
  info.axisCoalescingHints.clear();
  if (!info.barriers.empty())
    return;

  SmallVector<unsigned, 4> parallelRun = getMaximalParallelRun(info);
  if (parallelRun.size() < 2)
    return;

  AxisCoalescingHint hint;
  hint.groupId = 1;
  bool vectorizable = false;
  for (unsigned axis : parallelRun) {
    AxisScheduleConstraint *constraint =
        lookupAxisScheduleConstraint(info, axis);
    if (!constraint)
      continue;

    hint.memberAxisIds.push_back(axis);
    vectorizable |= llvm::is_contained(
        constraint->allowedRoles, AxisExecutionRole::VectorizeCandidate);
  }

  if (hint.memberAxisIds.size() < 2)
    return;

  hint.kind = vectorizable ? CoalescingHintKind::Vectorizable
                           : CoalescingHintKind::LinearizeOnly;
  for (unsigned axis : hint.memberAxisIds) {
    AxisScheduleConstraint *constraint =
        lookupAxisScheduleConstraint(info, axis);
    if (constraint)
      constraint->coalescingGroupId = hint.groupId;
  }

  info.axisCoalescingHints.push_back(std::move(hint));
}

void printAxisExecutionRoles(ArrayRef<AxisExecutionRole> roles,
                             llvm::raw_ostream &os) {
  os << "[";
  llvm::interleave(
      roles, os,
      [&](AxisExecutionRole role) {
        os << stringifyAxisExecutionRole(role);
      },
      ",");
  os << "]";
}

void printAxisTailPolicies(ArrayRef<AxisTailPolicy> policies,
                           llvm::raw_ostream &os) {
  os << "[";
  llvm::interleave(
      policies, os,
      [&](AxisTailPolicy policy) { os << stringifyAxisTailPolicy(policy); },
      ",");
  os << "]";
}

void printPrimitiveUses(ArrayRef<PrimitiveAxisUseKind> uses,
                        llvm::raw_ostream &os) {
  os << "[";
  llvm::interleave(
      uses, os,
      [&](PrimitiveAxisUseKind use) {
        os << stringifyPrimitiveAxisUseKind(use);
      },
      ",");
  os << "]";
}

bool shouldPrintTailContractFields(
    ArrayRef<AxisScheduleConstraint> constraints) {
  for (const AxisScheduleConstraint &constraint : constraints) {
    if (constraint.semanticAlignmentGranularity != 0)
      return true;
    if (llvm::is_contained(constraint.allowedTailPolicies,
                           AxisTailPolicy::PadAndMask))
      return true;
    if (llvm::is_contained(constraint.primitiveUses,
                           PrimitiveAxisUseKind::GatherIndex))
      return true;
  }
  return false;
}

const PatternOpView *selectAxisCarrierOp(const KernelPatternView &pattern) {
  const ::mlir::ascend::kernelize::HandwrittenContract *contract =
      ::mlir::ascend::kernelize::lookupHandwrittenContract(
          pattern.handwrittenKind);
  if (contract && contract->useAxisCarrierOnly)
    return selectDominantPrimaryOp(pattern);
  for (const PatternOpView &opView : pattern.ops) {
    if (opView.role == pattern.dominantRole)
      return &opView;
  }
  return selectDominantPrimaryOp(pattern);
}

} // namespace

FailureOr<CoalescedAxisInfo> coalesceAxes(const KernelPatternView &pattern) {
  const PatternOpView *axisOpView = selectAxisCarrierOp(pattern);
  if (!axisOpView || !axisOpView->op)
    return failure();

  Operation *axisOp = axisOpView->op;
  auto linalgOp = dyn_cast<linalg::LinalgOp>(axisOp);
  if (!linalgOp) {
    axisOp->emitError() << "axis coalescing requires a linalg axis carrier op";
    return failure();
  }

  CoalescedAxisInfo info;
  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();
  unsigned axisCount = getAxisCount(axisOp, axisOpView->role,
                                   static_cast<unsigned>(iteratorTypes.size()));
  SmallVector<int64_t> staticExtents(axisCount, ShapedType::kDynamic);
  SmallVector<bool> broadcastAxisMask(axisCount, false);

  const ::mlir::ascend::kernelize::HandwrittenContract *hwContract =
      ::mlir::ascend::kernelize::lookupHandwrittenContract(
          pattern.handwrittenKind);
  bool useAxisCarrierOnly = hwContract && hwContract->useAxisCarrierOnly;
  for (const PatternOpView &opView : pattern.ops) {
    if (useAxisCarrierOnly && opView.op != axisOp)
      continue;
    auto patternLinalgOp = dyn_cast_or_null<linalg::LinalgOp>(opView.op);
    if (!patternLinalgOp) {
      addBarrier(info, opView.op, AxisBarrierKind::RankMismatch,
                 "kernel pattern contains a non-linalg op");
      continue;
    }
    if (failed(collectIndexingMapInfo(patternLinalgOp, axisOpView->role,
                                      axisCount, staticExtents,
                                      broadcastAxisMask, info)))
      return failure();
  }

  info.logicalAxes.reserve(axisCount);
  for (unsigned axis = 0; axis < axisCount; ++axis) {
    AxisKind kind = AxisKind::Unknown;
    if (axis < iteratorTypes.size())
      kind = classifyIteratorType(iteratorTypes[axis], info, axisOp, axis);
    else
      addBarrier(info, axisOp, AxisBarrierKind::RankMismatch,
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

  appendPatternRawAxes(pattern, axisCount, info,
                       useAxisCarrierOnly ? axisOp : nullptr);
  deriveAxisScheduleConstraints(info);
  deriveAxisCoalescingHints(info);

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
  os << "  axis_constraints = [\n";
  bool printTailContract =
      shouldPrintTailContractFields(info.axisScheduleConstraints);
  for (const AxisScheduleConstraint &constraint :
       info.axisScheduleConstraints) {
    os << "    axis=" << constraint.logicalAxisId
       << " kind=" << stringifyAxisKind(constraint.kind) << " roles=";
    printAxisExecutionRoles(constraint.allowedRoles, os);
    os << " tail=" << stringifyAxisTailPolicy(constraint.tailPolicy);
    if (constraint.coalescingGroupId != 0)
      os << " group=" << constraint.coalescingGroupId;
    if (printTailContract) {
      os << " allowed_tail=";
      printAxisTailPolicies(constraint.allowedTailPolicies, os);
      os << " primitive_uses=";
      printPrimitiveUses(constraint.primitiveUses, os);
      os << " semantic_align="
         << constraint.semanticAlignmentGranularity;
    }
    os << "\n";
  }
  os << "  ]\n";
  os << "  coalescing_hints = [\n";
  for (const AxisCoalescingHint &hint : info.axisCoalescingHints) {
    os << "    group=" << hint.groupId
       << " kind=" << stringifyCoalescingHintKind(hint.kind)
       << " members=";
    printCompactAxisList(hint.memberAxisIds, os);
    os << "\n";
  }
  os << "  ]\n";
}

} // namespace mlir::ascend::schedule
