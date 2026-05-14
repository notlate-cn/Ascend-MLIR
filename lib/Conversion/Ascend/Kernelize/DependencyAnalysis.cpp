//===- DependencyAnalysis.cpp - Ascend kernel dependency analysis ------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/DependencyAnalysis.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace mlir;

namespace mlir::afir::ascend::kernelize {
namespace {

bool isTargetLinalgOp(Operation *op) {
  return isa<linalg::LinalgOp>(op);
}

StringRef getIteratorTypeName(Attribute attr) {
  if (auto stringAttr = dyn_cast<StringAttr>(attr))
    return stringAttr.getValue();

  SmallString<32> storage;
  llvm::raw_svector_ostream os(storage);
  attr.print(os);
  StringRef printed(storage);
  if (printed.contains("parallel"))
    return "parallel";
  if (printed.contains("reduction"))
    return "reduction";

  return "unknown";
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

enum class ParallelIndexingKind {
  Elementwise,
  Broadcast,
  LayoutTransform,
  Unknown
};

ParallelIndexingKind
classifyParallelIndexing(ArrayRef<AffineMap> indexingMaps,
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
                             ArrayRef<StringRef> fallbackIteratorTypes = {}) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    for (utils::IteratorType iteratorType : linalgOp.getIteratorTypesArray()) {
      switch (iteratorType) {
      case utils::IteratorType::parallel:
        summary.iteratorTypes.push_back("parallel");
        break;
      case utils::IteratorType::reduction:
        summary.iteratorTypes.push_back("reduction");
        break;
      default:
        summary.iteratorTypes.push_back("unknown");
        break;
      }
    }
  }

  auto iteratorTypes = op->getAttrOfType<ArrayAttr>("iterator_types");
  if (summary.iteratorTypes.empty() && iteratorTypes) {
    for (Attribute iteratorType : iteratorTypes)
      summary.iteratorTypes.push_back(getIteratorTypeName(iteratorType));
  } else if (summary.iteratorTypes.empty()) {
    summary.iteratorTypes.append(fallbackIteratorTypes.begin(),
                                 fallbackIteratorTypes.end());
  }

  bool sawNonParallel = false;
  for (StringRef typeName : summary.iteratorTypes) {
    if (typeName == "reduction") {
      summary.hasReductionIterator = true;
      sawNonParallel = true;
      continue;
    }
    if (typeName != "parallel")
      sawNonParallel = true;
  }
  summary.hasOnlyParallelIterators =
      !summary.iteratorTypes.empty() && !sawNonParallel;
}

OpSemanticSummary buildSemanticSummary(Operation *op, OperationId opId) {
  OpSemanticSummary summary;
  summary.op = op;
  summary.opId = opId;
  summary.resultRank = getFirstRankedShapedOutputRank(op);
  populateIndexingMaps(op, summary);

  StringRef opName = op->getName().getStringRef();
  if (opName == "linalg.matmul") {
    populateIteratorSummary(
        op, summary,
        ArrayRef<StringRef>{"parallel", "parallel", "reduction"});
    summary.accessPattern = AccessPatternKind::Contraction;
    return summary;
  }

  if (opName == "linalg.batch_matmul") {
    populateIteratorSummary(
        op, summary,
        ArrayRef<StringRef>{"parallel", "parallel", "parallel", "reduction"});
    summary.accessPattern = AccessPatternKind::Contraction;
    return summary;
  }

  if (opName == "linalg.transpose") {
    SmallVector<StringRef> parallelIterators(summary.resultRank, "parallel");
    populateIteratorSummary(op, summary, parallelIterators);
    summary.accessPattern = AccessPatternKind::LayoutTransform;
    return summary;
  }

  if (opName == "linalg.fill") {
    populateIteratorSummary(op, summary);
    summary.accessPattern = AccessPatternKind::Elementwise;
    return summary;
  }

  if (!isa<linalg::LinalgOp>(op) &&
      !op->getAttrOfType<ArrayAttr>("iterator_types")) {
    summary.accessPattern = AccessPatternKind::Unknown;
    return summary;
  }

  populateIteratorSummary(op, summary);

  if (summary.hasReductionIterator) {
    summary.accessPattern = AccessPatternKind::Reduction;
    return summary;
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
    return summary;
  }

  summary.accessPattern = AccessPatternKind::Unknown;
  return summary;
}

void sortAndUniqueByOpId(SmallVectorImpl<Operation *> &ops,
                         const DenseMap<Operation *, OperationId> &opIds) {
  llvm::sort(ops, [&](Operation *lhs, Operation *rhs) {
    return opIds.lookup(lhs).value < opIds.lookup(rhs).value;
  });
  ops.erase(std::unique(ops.begin(), ops.end()), ops.end());
}

} // namespace

FailureOr<DependencyAnalysisResult>
DependencyAnalyzer::analyze(ModuleOp module) const {
  DependencyAnalysisResult result;

  module.walk([&](Operation *op) {
    if (!isTargetLinalgOp(op))
      return;

    OperationId opId{static_cast<unsigned>(result.index.orderedOps.size())};
    result.index.orderedOps.push_back(op);
    result.index.opIds.try_emplace(op, opId);
  });

  for (Operation *op : result.index.orderedOps) {
    OperationId opId = result.index.opIds.lookup(op);
    result.summaries.try_emplace(op, buildSemanticSummary(op, opId));

    for (Value operand : op->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      if (!producer || !result.index.opIds.contains(producer))
        continue;

      result.index.producers[op].push_back(producer);
      result.index.consumers[producer].push_back(op);
    }
  }

  for (Operation *op : result.index.orderedOps) {
    sortAndUniqueByOpId(result.index.producers[op], result.index.opIds);
    sortAndUniqueByOpId(result.index.consumers[op], result.index.opIds);
  }

  return result;
}

void emitDependencyAnalysisReport(raw_ostream &os,
                                  const DependencyAnalysisResult &result) {
  os << "DependencyAnalysis\n";
  for (Operation *op : result.index.orderedOps) {
    const OpSemanticSummary &summary = result.summaries.lookup(op);
    auto producerIt = result.index.producers.find(op);
    auto consumerIt = result.index.consumers.find(op);
    size_t producerCount = producerIt == result.index.producers.end()
                               ? 0
                               : producerIt->second.size();
    size_t consumerCount = consumerIt == result.index.consumers.end()
                               ? 0
                               : consumerIt->second.size();
    os << "  op_id = " << summary.opId.value << " op = \""
       << op->getName().getStringRef() << "\" access = \""
       << stringifyAccessPattern(summary.accessPattern)
       << "\" producers = " << producerCount
       << " consumers = " << consumerCount
       << " result_rank = " << summary.resultRank << " iterators = [";
    llvm::interleaveComma(summary.iteratorTypes, os);
    os << "] has_reduction = "
       << (summary.hasReductionIterator ? "true" : "false")
       << " only_parallel = "
       << (summary.hasOnlyParallelIterators ? "true" : "false") << "\n";
  }
}

} // namespace mlir::afir::ascend::kernelize
