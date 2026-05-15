//===- DependencyAnalysis.cpp - Ascend kernel dependency analysis ------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "DependencyAnalysis.h"

#include "KernelizeOpRegistry.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace mlir;

namespace mlir::afir::ascend::kernelize {
namespace {

void sortAndUniqueByOpId(SmallVectorImpl<Operation *> &ops,
                         const DenseMap<Operation *, OperationId> &opIds) {
  llvm::sort(ops, [&](Operation *lhs, Operation *rhs) {
    return opIds.lookup(lhs).value < opIds.lookup(rhs).value;
  });
  ops.erase(std::unique(ops.begin(), ops.end()), ops.end());
}

void collectAnalyzedProducers(Value value, const ProducerConsumerIndex &index,
                              SmallVectorImpl<Operation *> &producers,
                              DenseSet<Operation *> &visited) {
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return;

  if (index.opIds.contains(producer)) {
    producers.push_back(producer);
    return;
  }

  if (!visited.insert(producer).second || producer->getNumRegions() != 0)
    return;

  for (Value operand : producer->getOperands()) {
    if (!isa<TensorType>(operand.getType()))
      continue;
    collectAnalyzedProducers(operand, index, producers, visited);
  }
}

} // namespace

FailureOr<DependencyAnalysisResult>
DependencyAnalyzer::analyze(ModuleOp module) const {
  DependencyAnalysisResult result;
  KernelizeOpRegistry registry = KernelizeOpRegistry::buildDefault();

  module.walk([&](Operation *op) {
    if (!registry.isTargetOp(op))
      return;

    OperationId opId{static_cast<unsigned>(result.index.orderedOps.size())};
    result.index.orderedOps.push_back(op);
    result.index.opIds.try_emplace(op, opId);
  });

  for (Operation *op : result.index.orderedOps) {
    OperationId opId = result.index.opIds.lookup(op);
    result.summaries.try_emplace(op, registry.summarize(op, opId));

    for (Value operand : op->getOperands()) {
      SmallVector<Operation *, 4> operandProducers;
      DenseSet<Operation *> visited;
      collectAnalyzedProducers(operand, result.index, operandProducers,
                               visited);
      sortAndUniqueByOpId(operandProducers, result.index.opIds);
      for (Operation *producer : operandProducers) {
        result.index.producers[op].push_back(producer);
        result.index.consumers[producer].push_back(op);
      }
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
    llvm::interleaveComma(summary.iteratorTypes, os,
                          [&](IteratorKind iteratorType) {
                            os << stringifyIteratorKind(iteratorType);
                          });
    os << "] has_reduction = "
       << (summary.hasReductionIterator ? "true" : "false")
       << " only_parallel = "
       << (summary.hasOnlyParallelIterators ? "true" : "false") << "\n";
  }
}

} // namespace mlir::afir::ascend::kernelize
