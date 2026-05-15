//===- KernelizeOpRegistry.cpp - Ascend kernelize op registry ------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "KernelizeOpRegistry.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

unsigned getFirstRankedShapedOutputRank(Operation *op) {
  for (Type resultType : op->getResultTypes()) {
    if (auto shapedType = dyn_cast<ShapedType>(resultType))
      if (shapedType.hasRank())
        return shapedType.getRank();
  }

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return 0;

  for (Value init : linalgOp.getDpsInits()) {
    auto shapedType = dyn_cast<ShapedType>(init.getType());
    if (shapedType && shapedType.hasRank())
      return shapedType.getRank();
  }

  return 0;
}

void populateIndexingMaps(Operation *op, OpSemanticSummary &summary) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    llvm::append_range(summary.indexingMaps, linalgOp.getIndexingMapsArray());
    return;
  }

  auto indexingMaps = op->getAttrOfType<ArrayAttr>("indexing_maps");
  if (!indexingMaps)
    return;

  for (Attribute attr : indexingMaps) {
    auto mapAttr = dyn_cast<AffineMapAttr>(attr);
    if (!mapAttr)
      continue;
    summary.indexingMaps.push_back(mapAttr.getValue());
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

void populateIteratorSummary(Operation *op, OpSemanticSummary &summary,
                             ArrayRef<IteratorKind> fallbackIteratorTypes = {}) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    for (utils::IteratorType iteratorType : linalgOp.getIteratorTypesArray())
      summary.iteratorTypes.push_back(convertIteratorType(iteratorType));
  }

  auto iteratorTypes = op->getAttrOfType<ArrayAttr>("iterator_types");
  if (summary.iteratorTypes.empty() && iteratorTypes) {
    for (Attribute iteratorType : iteratorTypes)
      summary.iteratorTypes.push_back(getIteratorKind(iteratorType));
  } else if (summary.iteratorTypes.empty()) {
    summary.iteratorTypes.append(fallbackIteratorTypes.begin(),
                                 fallbackIteratorTypes.end());
  }

  bool sawNonParallel = false;
  for (IteratorKind iteratorType : summary.iteratorTypes) {
    if (iteratorType == IteratorKind::Reduction) {
      summary.hasReductionIterator = true;
      sawNonParallel = true;
      continue;
    }
    if (iteratorType != IteratorKind::Parallel)
      sawNonParallel = true;
  }
  summary.hasOnlyParallelIterators =
      !summary.iteratorTypes.empty() && !sawNonParallel;
}

void populateLinalgSemanticSummary(Operation *op,
                                   OpSemanticSummary &summary) {
  summary.resultRank = getFirstRankedShapedOutputRank(op);
  populateIndexingMaps(op, summary);
  populateIteratorSummary(op, summary);

  if (summary.hasReductionIterator) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
        linalgOp && isContractionIndexing(linalgOp, summary.indexingMaps,
                                          summary.iteratorTypes)) {
      summary.accessPattern = AccessPatternKind::Contraction;
      return;
    }
    summary.accessPattern = AccessPatternKind::Reduction;
    return;
  }

  if (summary.hasOnlyParallelIterators) {
    switch (classifyParallelIndexing(summary.indexingMaps,
                                     summary.resultRank)) {
    case ParallelIndexingKind::Elementwise:
      summary.accessPattern = AccessPatternKind::Elementwise;
      break;
    case ParallelIndexingKind::Broadcast:
      summary.accessPattern = AccessPatternKind::Broadcast;
      break;
    case ParallelIndexingKind::LayoutTransform:
      summary.accessPattern = AccessPatternKind::LayoutTransform;
      break;
    case ParallelIndexingKind::Unknown:
      summary.accessPattern = AccessPatternKind::Unknown;
      break;
    }
    return;
  }

  summary.accessPattern = AccessPatternKind::Unknown;
}

} // namespace

KernelizeOpRegistry KernelizeOpRegistry::buildDefault() {
  KernelizeOpRegistry registry;
  registry.registerModel({matchLinalgOp, populateLinalgSemanticSummary});
  return registry;
}

void KernelizeOpRegistry::registerModel(KernelizeOpModel model) {
  models.push_back(model);
}

const KernelizeOpModel *
KernelizeOpRegistry::lookupModel(Operation *op) const {
  for (const KernelizeOpModel &model : models)
    if (model.match && model.match(op))
      return &model;
  return nullptr;
}

bool KernelizeOpRegistry::isTargetOp(Operation *op) const {
  return lookupModel(op) != nullptr;
}

OpSemanticSummary KernelizeOpRegistry::summarize(Operation *op,
                                                 OperationId opId) const {
  OpSemanticSummary summary;
  summary.op = op;
  summary.opId = opId;

  const KernelizeOpModel *model = lookupModel(op);
  if (!model || !model->populate) {
    summary.accessPattern = AccessPatternKind::Unknown;
    return summary;
  }

  model->populate(op, summary);
  return summary;
}

} // namespace mlir::afir::ascend::kernelize
