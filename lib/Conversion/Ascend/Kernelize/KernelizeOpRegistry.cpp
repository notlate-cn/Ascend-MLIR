//===- KernelizeOpRegistry.cpp - Ascend kernelize op registry ------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "KernelizeOpRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace mlir::afir::ascend::kernelize {
namespace {

bool matchLinalgOp(Operation *op) { return isa<linalg::LinalgOp>(op); }

bool matchArithConstantOp(Operation *op) { return isa<arith::ConstantOp>(op); }

bool matchTensorViewOp(Operation *op) {
  return isa<tensor::CastOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp,
             tensor::ExtractSliceOp, tensor::ReshapeOp>(op);
}

IteratorKind convertIteratorType(utils::IteratorType iteratorType) {
  switch (iteratorType) {
  case utils::IteratorType::parallel:
    return IteratorKind::Parallel;
  case utils::IteratorType::reduction:
    return IteratorKind::Reduction;
  default:
    return IteratorKind::Unknown;
  }
}

IteratorKind getIteratorKind(Attribute attr) {
  if (auto iteratorTypeAttr = dyn_cast<linalg::IteratorTypeAttr>(attr))
    return convertIteratorType(iteratorTypeAttr.getValue());

  if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
    if (stringAttr.getValue() == stringifyIteratorKind(IteratorKind::Parallel))
      return IteratorKind::Parallel;
    if (stringAttr.getValue() == stringifyIteratorKind(IteratorKind::Reduction))
      return IteratorKind::Reduction;
  }

  return IteratorKind::Unknown;
}

void appendUniqueRank(SmallVectorImpl<unsigned> &ranks, unsigned rank) {
  if (!llvm::is_contained(ranks, rank))
    ranks.push_back(rank);
}

void populateResultRanks(Operation *op, KernelizeOpSemanticInfo &info) {
  for (Type resultType : op->getResultTypes()) {
    if (auto shapedType = dyn_cast<ShapedType>(resultType))
      if (shapedType.hasRank())
        appendUniqueRank(info.resultRanks,
                         static_cast<unsigned>(shapedType.getRank()));
  }

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return;

  for (Value init : linalgOp.getDpsInits()) {
    auto shapedType = dyn_cast<ShapedType>(init.getType());
    if (shapedType && shapedType.hasRank())
      appendUniqueRank(info.resultRanks,
                       static_cast<unsigned>(shapedType.getRank()));
  }
}

void populateIndexingMaps(Operation *op, KernelizeOpSemanticInfo &info) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    llvm::append_range(info.indexingMaps, linalgOp.getIndexingMapsArray());
    return;
  }

  auto indexingMaps = op->getAttrOfType<ArrayAttr>("indexing_maps");
  if (!indexingMaps)
    return;

  for (Attribute attr : indexingMaps) {
    auto mapAttr = dyn_cast<AffineMapAttr>(attr);
    if (!mapAttr)
      continue;
    info.indexingMaps.push_back(mapAttr.getValue());
  }
}

bool isIdentityOnLeadingDims(AffineMap map) {
  if (map.getNumResults() != map.getNumDims())
    return false;

  for (auto [index, expr] : llvm::enumerate(map.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr || dimExpr.getPosition() != index)
      return false;
  }
  return true;
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

bool hasConstantResult(AffineMap map) {
  return llvm::any_of(map.getResults(), [](AffineExpr expr) {
    return isa<AffineConstantExpr>(expr);
  });
}

bool mapUsesIteratorKind(AffineMap map, ArrayRef<IteratorKind> iteratorTypes,
                         IteratorKind kind) {
  if (!isDimOrConstantProjection(map))
    return false;

  for (AffineExpr expr : map.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      continue;
    unsigned position = dimExpr.getPosition();
    if (position < iteratorTypes.size() && iteratorTypes[position] == kind)
      return true;
  }
  return false;
}

bool isContractionIndexing(linalg::LinalgOp linalgOp,
                           ArrayRef<AffineMap> indexingMaps,
                           ArrayRef<IteratorKind> iteratorTypes) {
  if (!llvm::is_contained(iteratorTypes, IteratorKind::Reduction))
    return false;

  unsigned inputCount = linalgOp.getNumDpsInputs();
  unsigned initCount = linalgOp.getNumDpsInits();
  if (inputCount < 2 || initCount == 0 ||
      indexingMaps.size() != inputCount + initCount)
    return false;

  unsigned reductionInputCount = 0;
  for (unsigned i = 0; i < inputCount; ++i)
    if (mapUsesIteratorKind(indexingMaps[i], iteratorTypes,
                            IteratorKind::Reduction))
      ++reductionInputCount;
  if (reductionInputCount < 2)
    return false;

  for (unsigned i = inputCount, e = indexingMaps.size(); i < e; ++i)
    if (mapUsesIteratorKind(indexingMaps[i], iteratorTypes,
                            IteratorKind::Reduction))
      return false;

  return true;
}

enum class ParallelIndexingKind {
  Elementwise,
  Broadcast,
  LayoutTransform,
  Unknown
};

ParallelIndexingKind classifyParallelIndexing(ArrayRef<AffineMap> indexingMaps,
                                              unsigned resultRank) {
  if (indexingMaps.empty())
    return ParallelIndexingKind::Unknown;

  bool hasProjectedMap = false;
  bool hasNonIdentityFullRankMap = false;
  for (AffineMap map : indexingMaps) {
    if (!isDimOrConstantProjection(map))
      return ParallelIndexingKind::Unknown;

    if (map.getNumResults() > resultRank)
      return ParallelIndexingKind::Unknown;

    if (map.getNumResults() == 0)
      continue;

    if (map.getNumResults() < resultRank) {
      hasProjectedMap = true;
      continue;
    }

    if (map.getNumResults() != resultRank)
      return ParallelIndexingKind::Unknown;

    if (hasConstantResult(map)) {
      hasProjectedMap = true;
      continue;
    }

    if (!isIdentityOnLeadingDims(map))
      hasNonIdentityFullRankMap = true;
  }

  if (hasNonIdentityFullRankMap)
    return ParallelIndexingKind::LayoutTransform;
  if (hasProjectedMap)
    return ParallelIndexingKind::Broadcast;
  return ParallelIndexingKind::Elementwise;
}

void populateIteratorInfo(Operation *op, KernelizeOpSemanticInfo &info,
                          ArrayRef<IteratorKind> fallbackIteratorTypes = {}) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    for (utils::IteratorType iteratorType : linalgOp.getIteratorTypesArray())
      info.iteratorKinds.push_back(convertIteratorType(iteratorType));
  }

  auto iteratorTypes = op->getAttrOfType<ArrayAttr>("iterator_types");
  if (info.iteratorKinds.empty() && iteratorTypes) {
    for (Attribute iteratorType : iteratorTypes)
      info.iteratorKinds.push_back(getIteratorKind(iteratorType));
  } else if (info.iteratorKinds.empty()) {
    info.iteratorKinds.append(fallbackIteratorTypes.begin(),
                              fallbackIteratorTypes.end());
  }
}

bool hasReductionIterator(ArrayRef<IteratorKind> iteratorKinds) {
  return llvm::is_contained(iteratorKinds, IteratorKind::Reduction);
}

bool hasOnlyParallelIterators(ArrayRef<IteratorKind> iteratorKinds) {
  if (iteratorKinds.empty())
    return false;
  return llvm::all_of(iteratorKinds, [](IteratorKind iteratorKind) {
    return iteratorKind == IteratorKind::Parallel;
  });
}

LogicalResult populateLinalgSemanticInfo(Operation *op,
                                         KernelizeOpSemanticInfo &info) {
  info.participation = KernelizeParticipationKind::Analyze;
  info.modelName = "linalg";
  info.seedPolicy = KernelizeSeedPolicy::MaySeed;
  info.traits.push_back(KernelizeSemanticTrait::Structured);

  populateResultRanks(op, info);
  populateIndexingMaps(op, info);
  populateIteratorInfo(op, info);

  if (info.resultRanks.size() > 1) {
    info.participation = KernelizeParticipationKind::Unsupported;
    info.accessPattern = AccessPatternKind::Unknown;
    info.unsupportedReason = "linalg op has inconsistent ranked result ranks";
    return success();
  }

  unsigned resultRank =
      info.resultRanks.empty() ? 0 : info.resultRanks.front();
  if (hasReductionIterator(info.iteratorKinds)) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
        linalgOp && isContractionIndexing(linalgOp, info.indexingMaps,
                                          info.iteratorKinds)) {
      info.accessPattern = AccessPatternKind::Contraction;
      return success();
    }
    info.accessPattern = AccessPatternKind::Reduction;
    info.seedPolicy = KernelizeSeedPolicy::NonSeedWhenFused;
    return success();
  }

  if (hasOnlyParallelIterators(info.iteratorKinds)) {
    switch (classifyParallelIndexing(info.indexingMaps, resultRank)) {
    case ParallelIndexingKind::Elementwise:
      info.accessPattern = AccessPatternKind::Elementwise;
      break;
    case ParallelIndexingKind::Broadcast:
      info.accessPattern = AccessPatternKind::Broadcast;
      break;
    case ParallelIndexingKind::LayoutTransform:
      info.accessPattern = AccessPatternKind::LayoutTransform;
      break;
    case ParallelIndexingKind::Unknown:
      info.accessPattern = AccessPatternKind::Unknown;
      break;
    }
    return success();
  }

  info.accessPattern = AccessPatternKind::Unknown;
  return success();
}

LogicalResult populateArithConstantSemanticInfo(
    Operation *, KernelizeOpSemanticInfo &info) {
  info.participation = KernelizeParticipationKind::Ignore;
  info.accessPattern = AccessPatternKind::NotApplicable;
  info.seedPolicy = KernelizeSeedPolicy::NeverSeed;
  info.modelName = "arith_constant";
  return success();
}

LogicalResult populateTensorViewSemanticInfo(Operation *op,
                                             KernelizeOpSemanticInfo &info) {
  info.participation = KernelizeParticipationKind::Transparent;
  info.accessPattern = AccessPatternKind::LayoutTransform;
  info.seedPolicy = KernelizeSeedPolicy::NeverSeed;
  info.traits.push_back(KernelizeSemanticTrait::TensorView);
  info.modelName = "tensor_view";
  if (isa<tensor::ReshapeOp>(op))
    info.transparentOperandIndices.push_back(0);
  return success();
}

} // namespace

void registerDefaultKernelizeOpModels(KernelizeOpModelRegistry &registry) {
  registry.registerModel(
      {"linalg", matchLinalgOp, populateLinalgSemanticInfo});
  registry.registerModel({"arith_constant", matchArithConstantOp,
                          populateArithConstantSemanticInfo});
  registry.registerModel(
      {"tensor_view", matchTensorViewOp, populateTensorViewSemanticInfo});
}

} // namespace mlir::afir::ascend::kernelize
