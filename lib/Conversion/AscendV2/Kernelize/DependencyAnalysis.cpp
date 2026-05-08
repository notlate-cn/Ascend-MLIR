//===- DependencyAnalysis.cpp - Ascend V2 kernel dependency analysis ------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Kernelize/DependencyAnalysis.h"

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

namespace mlir::afir::ascend::v2::kernelize {
namespace {

bool isTargetLinalgOp(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  return opName == "linalg.generic" || opName == "linalg.matmul" ||
         opName == "linalg.batch_matmul";
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

unsigned getFirstRankedTensorResultRank(Operation *op) {
  for (Type resultType : op->getResultTypes()) {
    if (auto rankedTensorType = dyn_cast<RankedTensorType>(resultType))
      return rankedTensorType.getRank();
  }
  return 0;
}

void populateIndexingMaps(Operation *op, OpSemanticSummary &summary) {
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
    if (!map.isProjectedPermutation())
      return ParallelIndexingKind::Unknown;

    if (map.getNumResults() > resultRank)
      return ParallelIndexingKind::Unknown;

    if (map.getNumResults() < resultRank) {
      hasProjectedMap = true;
      continue;
    }

    if (map.getNumResults() != resultRank)
      return ParallelIndexingKind::Unknown;

    if (!isIdentityOnLeadingDims(map))
      hasNonIdentityFullRankMap = true;
  }

  if (hasProjectedMap)
    return ParallelIndexingKind::Broadcast;
  if (hasNonIdentityFullRankMap)
    return ParallelIndexingKind::LayoutTransform;
  return ParallelIndexingKind::Elementwise;
}

void populateIteratorSummary(Operation *op, OpSemanticSummary &summary,
                             ArrayRef<StringRef> fallbackIteratorTypes = {}) {
  auto iteratorTypes = op->getAttrOfType<ArrayAttr>("iterator_types");

  if (iteratorTypes) {
    for (Attribute iteratorType : iteratorTypes)
      summary.iteratorTypes.push_back(getIteratorTypeName(iteratorType));
  } else {
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
  summary.resultRank = getFirstRankedTensorResultRank(op);
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

  if (!op->getAttrOfType<ArrayAttr>("iterator_types")) {
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

} // namespace mlir::afir::ascend::v2::kernelize
