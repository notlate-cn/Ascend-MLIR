//===- DependencyAnalysis.cpp - Ascend kernel dependency analysis ------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/Analysis/DependencyAnalysis.h"

#include "Conversion/Ascend/Common/SymbolConstraints.h"
#include "Conversion/Ascend/Kernelize/Semantic/KernelizeOpRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
#include <optional>
#include <utility>

using namespace mlir;

namespace mlir::ascend::kernelize {
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

  SmallVector<Value, 4> transparentOperands;
  if (!info.transparentOperandIndices.empty()) {
    for (unsigned operandIndex : info.transparentOperandIndices) {
      if (operandIndex >= producer->getNumOperands())
        continue;
      transparentOperands.push_back(producer->getOperand(operandIndex));
    }
  } else {
    transparentOperands.append(producer->operand_begin(),
                               producer->operand_end());
  }

  for (Value operand : transparentOperands) {
    if (!isa<TensorType>(operand.getType()))
      continue;
    collectAnalyzedProducers(operand, index, resolved, producers,
                             unsupportedProducers, visited);
  }
}

void collectDependencyOperands(Operation *op,
                               SmallVectorImpl<Value> &operands) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    for (OpOperand *inputOperand : linalgOp.getDpsInputOperands())
      operands.push_back(inputOperand->get());
    return;
  }

  operands.append(op->operand_begin(), op->operand_end());
}

OpSemanticSummary makeSummary(Operation *op, OperationId opId,
                              const KernelizeOpSemanticInfo &info) {
  OpSemanticSummary summary;
  summary.op = op;
  summary.opId = opId;
  summary.participation = info.participation;
  summary.accessPattern = info.accessPattern;
  summary.seedPolicy = info.seedPolicy;
  summary.iteratorTypes.append(info.iteratorKinds.begin(),
                               info.iteratorKinds.end());
  summary.indexingMaps.append(info.indexingMaps.begin(),
                              info.indexingMaps.end());
  summary.resultRanks.append(info.resultRanks.begin(), info.resultRanks.end());
  summary.traits.append(info.traits.begin(), info.traits.end());
  summary.preferredTemplateFamilies.append(
      info.preferredTemplateFamilies.begin(),
      info.preferredTemplateFamilies.end());
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

std::optional<unsigned>
lookupAxisId(const symbol::SymbolConstraintTable &symbols, Value value,
             int64_t dim) {
  symbol::DimRef ref{value, dim};
  for (auto [classIndex, klass] : llvm::enumerate(symbols.classes)) {
    if (llvm::is_contained(klass.members, ref))
      return static_cast<unsigned>(classIndex);
  }
  return std::nullopt;
}

void refineAxisKind(FunctionAxisSpace &space, unsigned axisId,
                    IteratorKind iteratorKind) {
  if (axisId >= space.axes.size() || iteratorKind == IteratorKind::Unknown)
    return;

  LogicalAxis &axis = space.axes[axisId];
  if (axis.kind == IteratorKind::Unknown) {
    axis.kind = iteratorKind;
    return;
  }
  if (axis.kind != iteratorKind)
    axis.hasMixedIteratorKinds = true;
}

LogicalResult mergeIteratorAxis(SmallVectorImpl<OpAxisRef> &opAxes,
                                unsigned iteratorIdx, unsigned axisId,
                                StringRef symbolName,
                                IteratorKind iteratorKind, Operation *op) {
  if (iteratorIdx >= opAxes.size())
    return success();

  OpAxisRef &axis = opAxes[iteratorIdx];
  if (!axis.hasAxis()) {
    axis.axisId = axisId;
    axis.symbolName = symbolName.str();
    axis.iteratorKind = iteratorKind;
    return success();
  }

  if (axis.axisId == static_cast<int64_t>(axisId))
    return success();

  return op->emitError()
         << "conflicting symbol axes for iterator " << iteratorIdx << ": "
         << axis.symbolName << " vs " << symbolName;
}

LogicalResult mapValueDimsToIterators(
    Value value, AffineMap map, const symbol::SymbolConstraintTable &symbols,
    ArrayRef<IteratorKind> iteratorTypes, FunctionAxisSpace &space,
    SmallVectorImpl<OpAxisRef> &opAxes, Operation *op) {
  auto shapedType = dyn_cast<ShapedType>(value.getType());
  if (!shapedType || !shapedType.hasRank())
    return success();
  if (map.getNumResults() != static_cast<unsigned>(shapedType.getRank()))
    return success();

  for (auto [dim, expr] : llvm::enumerate(map.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      continue;

    unsigned iteratorIdx = dimExpr.getPosition();
    if (iteratorIdx >= opAxes.size())
      continue;

    std::optional<unsigned> axisId =
        lookupAxisId(symbols, value, static_cast<int64_t>(dim));
    if (!axisId)
      continue;

    IteratorKind iteratorKind = iteratorIdx < iteratorTypes.size()
                                    ? iteratorTypes[iteratorIdx]
                                    : IteratorKind::Unknown;
    StringRef symbolName = space.axes[*axisId].symbolName;
    if (failed(mergeIteratorAxis(opAxes, iteratorIdx, *axisId, symbolName,
                                 iteratorKind, op)))
      return failure();
    refineAxisKind(space, *axisId, iteratorKind);
  }

  return success();
}

LogicalResult buildLinalgOpAxisMap(
    linalg::LinalgOp linalgOp, const OpSemanticSummary &summary,
    const symbol::SymbolConstraintTable &symbols, FunctionAxisSpace &space,
    DenseMap<Operation *, SmallVector<OpAxisRef, 4>> &opAxisMap) {
  Operation *op = linalgOp.getOperation();
  if (summary.iteratorTypes.empty())
    return success();

  SmallVector<OpAxisRef, 4> axes(summary.iteratorTypes.size());
  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();

  unsigned operandMapCount =
      std::min<unsigned>(indexingMaps.size(), op->getNumOperands());
  for (unsigned operandIndex = 0; operandIndex < operandMapCount;
       ++operandIndex) {
    if (failed(mapValueDimsToIterators(
            op->getOperand(operandIndex), indexingMaps[operandIndex], symbols,
            summary.iteratorTypes, space, axes, op)))
      return failure();
  }

  unsigned outputMapBase = linalgOp.getNumDpsInputs();
  unsigned resultMapCount = std::min<unsigned>(
      op->getNumResults(),
      outputMapBase < indexingMaps.size() ? indexingMaps.size() - outputMapBase
                                          : 0);
  for (unsigned resultIndex = 0; resultIndex < resultMapCount; ++resultIndex) {
    if (failed(mapValueDimsToIterators(
            op->getResult(resultIndex),
            indexingMaps[outputMapBase + resultIndex], symbols,
            summary.iteratorTypes, space, axes, op)))
      return failure();
  }

  if (llvm::any_of(axes, [](const OpAxisRef &axis) { return axis.hasAxis(); }))
    opAxisMap.try_emplace(op, std::move(axes));
  return success();
}

LogicalResult buildFunctionAxisSpace(func::FuncOp func,
                                     DependencyAnalysisResult &result) {
  FailureOr<symbol::SymbolConstraintTable> symbols =
      symbol::parseSymbolConstraintAttr(func);
  if (failed(symbols))
    return failure();
  if (symbols->classes.empty())
    return success();

  FunctionAxisSpace space;
  space.func = func.getOperation();
  for (auto [classIndex, klass] : llvm::enumerate(symbols->classes)) {
    LogicalAxis axis;
    axis.func = func.getOperation();
    axis.axisId = static_cast<unsigned>(classIndex);
    axis.symbolName = klass.symName.getValue().str();
    axis.memberCount = static_cast<unsigned>(klass.members.size());
    space.axes.push_back(std::move(axis));
  }

  for (Operation *op : result.index.orderedOps) {
    if (op->getParentOfType<func::FuncOp>() != func)
      continue;
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp)
      continue;

    auto summaryIt = result.summaries.find(op);
    if (summaryIt == result.summaries.end())
      continue;
    if (failed(buildLinalgOpAxisMap(linalgOp, summaryIt->second, *symbols,
                                    space, result.opAxisMap)))
      return failure();
  }

  result.axisSpaces.push_back(std::move(space));
  return success();
}

LogicalResult buildAxisSpaces(ModuleOp module,
                              DependencyAnalysisResult &result) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (failed(buildFunctionAxisSpace(func, result)))
      return failure();
  }
  return success();
}

} // namespace

FailureOr<DependencyAnalysisResult>
DependencyAnalyzer::analyze(ModuleOp module) const {
  DependencyAnalysisResult result;
  KernelizeOpModelRegistry registry;
  registerDefaultKernelizeOpModels(registry);

  DenseMap<Operation *, KernelizeOpSemanticInfo> resolved;
  WalkResult resolveResult = module.walk([&](Operation *op) -> WalkResult {
    FailureOr<KernelizeOpSemanticInfo> info = registry.resolve(op);
    if (failed(info)) {
      op->emitError() << "failed to resolve Kernelize op semantics";
      return WalkResult::interrupt();
    }
    resolved.try_emplace(op, std::move(*info));
    return WalkResult::advance();
  });
  if (resolveResult.wasInterrupted())
    return failure();

  WalkResult unsupportedResult = module.walk([&](Operation *op) -> WalkResult {
    auto it = resolved.find(op);
    if (it == resolved.end())
      return WalkResult::advance();

    const KernelizeOpSemanticInfo &info = it->second;
    if (info.participation == KernelizeParticipationKind::Unsupported &&
        !info.unsupportedReason.empty()) {
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

    SmallVector<Value, 4> dependencyOperands;
    collectDependencyOperands(op, dependencyOperands);
    for (Value operand : dependencyOperands) {
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

  if (failed(buildAxisSpaces(module, result)))
    return failure();

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
       << "\" seed_policy = \""
       << stringifyKernelizeSeedPolicy(summary.seedPolicy)
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
  if (!result.axisSpaces.empty()) {
    os << "GlobalAxisSpace\n";
    for (const FunctionAxisSpace &space : result.axisSpaces) {
      auto func = cast<func::FuncOp>(space.func);
      os << "  func = \"" << func.getName() << "\" axes = "
         << space.axes.size() << "\n";
      for (const LogicalAxis &axis : space.axes) {
        os << "  axis_id = " << axis.axisId << " sym = \""
           << axis.symbolName << "\" kind = \""
           << stringifyIteratorKind(axis.kind) << "\" members = "
           << axis.memberCount;
        if (axis.hasMixedIteratorKinds)
          os << " mixed_iterator_kinds = true";
        os << "\n";
      }
    }
  }
  if (!result.opAxisMap.empty()) {
    os << "OpAxisMap\n";
    for (Operation *op : result.index.orderedOps) {
      auto axisIt = result.opAxisMap.find(op);
      if (axisIt == result.opAxisMap.end())
        continue;

      const OpSemanticSummary &summary = result.summaries.lookup(op);
      os << "  op_id = " << summary.opId.value << " axes = [";
      llvm::interleaveComma(axisIt->second, os, [&](const OpAxisRef &axis) {
        if (!axis.hasAxis()) {
          os << "_";
          return;
        }
        os << axis.axisId << ":\"" << axis.symbolName << "\"";
      });
      os << "] iterator_kinds = [";
      llvm::interleaveComma(axisIt->second, os, [&](const OpAxisRef &axis) {
        os << stringifyIteratorKind(axis.iteratorKind);
      });
      os << "]\n";
    }
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

} // namespace mlir::ascend::kernelize
