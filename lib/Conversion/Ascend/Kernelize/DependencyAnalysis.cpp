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
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <utility>

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

bool hasTensorResult(Operation *op) {
  return llvm::any_of(op->getResultTypes(),
                      [](Type type) { return isa<TensorType>(type); });
}

void collectAnalyzedProducers(
    Value value, const ProducerConsumerIndex &index,
    const DenseMap<Operation *, KernelizeOpSemanticInfo> &resolved,
    SmallVectorImpl<Operation *> &producers,
    SmallVectorImpl<Operation *> &unsupportedProducers,
    DenseSet<Operation *> &visited) {
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return;

  if (index.opIds.contains(producer)) {
    producers.push_back(producer);
    return;
  }

  if (!visited.insert(producer).second)
    return;

  auto resolvedIt = resolved.find(producer);
  if (resolvedIt == resolved.end())
    return;

  const KernelizeOpSemanticInfo &info = resolvedIt->second;
  if (info.participation == KernelizeParticipationKind::Unsupported) {
    if (hasTensorResult(producer))
      unsupportedProducers.push_back(producer);
    return;
  }

  if (info.participation != KernelizeParticipationKind::Transparent)
    return;

  if (producer->getNumRegions() != 0)
    return;

  for (Value operand : producer->getOperands()) {
    if (!isa<TensorType>(operand.getType()))
      continue;
    collectAnalyzedProducers(operand, index, resolved, producers,
                             unsupportedProducers, visited);
  }
}

OpSemanticSummary makeSummary(Operation *op, OperationId opId,
                              const KernelizeOpSemanticInfo &info) {
  OpSemanticSummary summary;
  summary.op = op;
  summary.opId = opId;
  summary.participation = info.participation;
  summary.accessPattern = info.accessPattern;
  summary.iteratorTypes.append(info.iteratorKinds.begin(),
                               info.iteratorKinds.end());
  summary.indexingMaps.append(info.indexingMaps.begin(),
                              info.indexingMaps.end());
  summary.resultRanks.append(info.resultRanks.begin(), info.resultRanks.end());
  summary.traits.append(info.traits.begin(), info.traits.end());
  summary.modelName = info.modelName;
  summary.unsupportedReason = info.unsupportedReason;
  summary.resultRank =
      summary.resultRanks.empty() ? 0 : summary.resultRanks.front();
  summary.hasReductionIterator =
      llvm::is_contained(summary.iteratorTypes, IteratorKind::Reduction);
  summary.hasOnlyParallelIterators =
      !summary.iteratorTypes.empty() &&
      llvm::all_of(summary.iteratorTypes, [](IteratorKind kind) {
        return kind == IteratorKind::Parallel;
      });
  return summary;
}

} // namespace

FailureOr<DependencyAnalysisResult>
DependencyAnalyzer::analyze(ModuleOp module) const {
  DependencyAnalysisResult result;
  KernelizeOpModelRegistry registry;
  registerDefaultKernelizeOpModels(registry);

  DenseMap<Operation *, KernelizeOpSemanticInfo> resolved;
  module.walk([&](Operation *op) {
    FailureOr<KernelizeOpSemanticInfo> info = registry.resolve(op);
    if (failed(info))
      return;
    resolved.try_emplace(op, std::move(*info));
  });

  WalkResult unsupportedResult = module.walk([&](Operation *op) -> WalkResult {
    auto it = resolved.find(op);
    if (it == resolved.end())
      return WalkResult::advance();

    const KernelizeOpSemanticInfo &info = it->second;
    if (info.participation == KernelizeParticipationKind::Unsupported &&
        info.modelName != "unknown" && !info.unsupportedReason.empty()) {
      op->emitError() << "unsupported Kernelize op semantics: "
                      << info.unsupportedReason;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
  if (unsupportedResult.wasInterrupted())
    return failure();

  module.walk([&](Operation *op) {
    auto it = resolved.find(op);
    if (it == resolved.end() ||
        it->second.participation != KernelizeParticipationKind::Analyze)
      return;

    OperationId opId{static_cast<unsigned>(result.index.orderedOps.size())};
    result.index.orderedOps.push_back(op);
    result.index.opIds.try_emplace(op, opId);
  });

  DenseMap<Operation *, DenseSet<Operation *>> reportedUnsupportedProducers;
  for (Operation *op : result.index.orderedOps) {
    OperationId opId = result.index.opIds.lookup(op);
    auto resolvedIt = resolved.find(op);
    assert(resolvedIt != resolved.end() &&
           "analyzed op must have resolved semantic info");
    result.summaries.try_emplace(op,
                                 makeSummary(op, opId, resolvedIt->second));

    for (Value operand : op->getOperands()) {
      SmallVector<Operation *, 4> operandProducers;
      SmallVector<Operation *, 4> unsupportedProducers;
      DenseSet<Operation *> visited;
      collectAnalyzedProducers(operand, result.index, resolved,
                               operandProducers, unsupportedProducers,
                               visited);
      for (Operation *unsupported : unsupportedProducers) {
        if (!reportedUnsupportedProducers[op].insert(unsupported).second)
          continue;
        std::string reason = "no kernelize semantic model for op " +
                             unsupported->getName().getStringRef().str();
        result.unsupportedProducers.push_back({unsupported, op, reason});
      }
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

  if (!result.unsupportedProducers.empty()) {
    const UnsupportedProducerDiagnostic &diag =
        result.unsupportedProducers.front();
    diag.producer->emitError()
        << "unsupported Kernelize tensor producer \""
        << diag.producer->getName().getStringRef() << "\" consumed by \""
        << diag.consumer->getName().getStringRef() << "\"";
    return failure();
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
       << op->getName().getStringRef() << "\" participation = \""
       << stringifyKernelizeParticipation(summary.participation)
       << "\" model = \"" << summary.modelName << "\" traits = [";
    llvm::interleaveComma(summary.traits, os,
                          [&](KernelizeSemanticTrait trait) {
                            os << "\""
                               << stringifyKernelizeSemanticTrait(trait)
                               << "\"";
                          });
    os << "] access = \"" << stringifyAccessPattern(summary.accessPattern)
       << "\" result_ranks = [";
    llvm::interleaveComma(summary.resultRanks, os,
                          [&](unsigned rank) { os << rank; });
    os << "]";
    if (!summary.unsupportedReason.empty())
      os << " unsupported_reason = \"" << summary.unsupportedReason << "\"";
    os << " producers = " << producerCount
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
  if (!result.unsupportedProducers.empty()) {
    os << "UnsupportedProducers\n";
    for (const UnsupportedProducerDiagnostic &diag :
         result.unsupportedProducers) {
      os << "  producer = \"" << diag.producer->getName().getStringRef()
         << "\" consumer = \"" << diag.consumer->getName().getStringRef()
         << "\" unsupported_reason = \"" << diag.reason << "\"\n";
    }
  }
}

} // namespace mlir::afir::ascend::kernelize
