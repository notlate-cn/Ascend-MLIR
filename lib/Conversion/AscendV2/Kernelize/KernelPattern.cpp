//===- KernelPattern.cpp - Ascend V2 kernel pattern model ----------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Kernelize/KernelPattern.h"

#include "Conversion/AscendV2/Kernelize/CandidateClosure.h"
#include "Conversion/AscendV2/Kernelize/KernelizeTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

using namespace mlir;

namespace mlir::afir::ascend::v2::kernelize {
namespace {

struct CandidateBuildRecord {
  KernelPatternCandidate candidate;
  unsigned sourceId = 0;
};

unsigned getOpId(Operation *op, const ProducerConsumerIndex &index) {
  auto it = index.opIds.find(op);
  if (it == index.opIds.end())
    return std::numeric_limits<unsigned>::max();
  return it->second.value;
}

unsigned getFirstPrimaryOpId(const KernelPatternCandidate &candidate,
                             const ProducerConsumerIndex &index) {
  if (candidate.primaryOps.empty())
    return std::numeric_limits<unsigned>::max();
  return getOpId(candidate.primaryOps.front(), index);
}

unsigned getMinOpId(ArrayRef<Operation *> ops,
                    const ProducerConsumerIndex &index) {
  unsigned minId = std::numeric_limits<unsigned>::max();
  for (Operation *op : ops)
    minId = std::min(minId, getOpId(op, index));
  return minId;
}

void appendUniqueOp(SmallVectorImpl<Operation *> &ops, Operation *op) {
  if (!llvm::is_contained(ops, op))
    ops.push_back(op);
}

void appendOps(SmallVectorImpl<Operation *> &ops,
               ArrayRef<Operation *> newOps) {
  for (Operation *op : newOps)
    appendUniqueOp(ops, op);
}

void sortUniqueOpsByOpId(SmallVectorImpl<Operation *> &ops,
                         const ProducerConsumerIndex &index) {
  llvm::sort(ops, [&](Operation *lhs, Operation *rhs) {
    return getOpId(lhs, index) < getOpId(rhs, index);
  });
  ops.erase(std::unique(ops.begin(), ops.end()), ops.end());
}

void sortUniqueIds(SmallVectorImpl<unsigned> &ids) {
  llvm::sort(ids);
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
}

void appendEdge(SmallVectorImpl<KernelPatternEdge> &edges,
                DenseSet<KernelPatternEdgeKey> &seenEdges, unsigned from,
                unsigned to,
                KernelPatternEdgeKind kind, Value carriedValue = {}) {
  if (from == to)
    return;

  KernelPatternEdgeKey key{from, to, kind};
  if (!seenEdges.insert(key).second)
    return;

  KernelPatternEdge edge;
  edge.from = from;
  edge.to = to;
  edge.kind = kind;
  edge.carriedValue = carriedValue;
  edges.push_back(edge);
}

Value findCarriedValue(Operation *producer, Operation *consumer) {
  for (Value operand : consumer->getOperands()) {
    if (operand.getDefiningOp() == producer)
      return operand;
  }
  return {};
}

void sortEdges(SmallVectorImpl<KernelPatternEdge> &edges) {
  llvm::sort(edges, [](const KernelPatternEdge &lhs,
                       const KernelPatternEdge &rhs) {
    return std::make_tuple(lhs.from, lhs.to, static_cast<unsigned>(lhs.kind)) <
           std::make_tuple(rhs.from, rhs.to, static_cast<unsigned>(rhs.kind));
  });
}

KernelPatternCandidate buildCandidateFromFusion(
    const FusionCandidate &candidate) {
  KernelPatternCandidate node;
  node.sourceKind = candidate.kind;
  node.internalOps.append(candidate.internalOps.begin(),
                          candidate.internalOps.end());
  node.primaryOps.append(candidate.primaryOps.begin(),
                         candidate.primaryOps.end());
  node.closure = candidate.closure;
  node.scheduleContract = candidate.scheduleContract;
  node.benefitScore = candidate.benefitScore;
  return node;
}

KernelPatternCandidate buildCandidateFromMerged(
    const MergedCandidate &candidate) {
  KernelPatternCandidate node;
  node.sourceKind = CandidateKind::Merged;
  node.internalOps.append(candidate.internalOps.begin(),
                          candidate.internalOps.end());
  node.primaryOps.append(candidate.primaryOps.begin(),
                         candidate.primaryOps.end());
  node.closure = candidate.closure;
  node.scheduleContract = candidate.scheduleContract;
  node.benefitScore = candidate.benefitScore;
  return node;
}

KernelPatternCandidate buildCandidateFromHorizontal(
    const HorizontalFusionCandidate &candidate,
    const DenseMap<unsigned, const FusionCandidate *> &fusionSources,
    const DenseMap<unsigned, const MergedCandidate *> &mergedSources,
    unsigned mergedIdOffset,
    const DependencyAnalysisResult &deps) {
  KernelPatternCandidate node;
  node.sourceKind = CandidateKind::HorizontalFusion;
  node.benefitScore = candidate.benefitScore;

  for (unsigned sourceId : candidate.siblingCandidateIds) {
    if (const FusionCandidate *source = fusionSources.lookup(sourceId)) {
      appendOps(node.internalOps, source->internalOps);
      appendOps(node.primaryOps, source->primaryOps);
      continue;
    }

    if (sourceId < mergedIdOffset)
      continue;

    const MergedCandidate *source =
        mergedSources.lookup(sourceId - mergedIdOffset);
    if (!source)
      continue;
    appendOps(node.internalOps, source->internalOps);
    appendOps(node.primaryOps, source->primaryOps);
  }

  sortUniqueOpsByOpId(node.internalOps, deps.index);
  sortUniqueOpsByOpId(node.primaryOps, deps.index);
  node.closure = computeCandidateClosure(node.internalOps, deps.index);
  if (!candidate.perGroupContracts.empty())
    node.scheduleContract = candidate.perGroupContracts.front();
  return node;
}

ScheduleContract buildFallbackContract(StringRef roleName) {
  ScheduleContract contract;
  if (roleName == "cube")
    contract.templateFamilies.push_back("cube");
  else if (roleName == "reduction")
    contract.templateFamilies.push_back("reduction");
  else if (roleName == "vector")
    contract.templateFamilies.push_back("vector");
  return contract;
}

bool hasSchedulableRole(Operation *op, ScheduleContract &contract) {
  auto role = op->getAttrOfType<StringAttr>(kOpRoleAttr);
  if (!role || role.getValue() == "unsupported")
    return false;

  contract = buildFallbackContract(role.getValue());
  return !contract.templateFamilies.empty();
}

KernelPattern buildPatternFromCandidate(const KernelPatternCandidate &candidate) {
  KernelPattern pattern;
  pattern.internalOps.append(candidate.internalOps.begin(),
                             candidate.internalOps.end());
  pattern.primaryOps.append(candidate.primaryOps.begin(),
                            candidate.primaryOps.end());
  pattern.scheduleContract = candidate.scheduleContract;
  return pattern;
}

KernelPattern buildFallbackPattern(Operation *op,
                                   const DependencyAnalysisResult &deps,
                                   ScheduleContract contract) {
  KernelPattern pattern;
  pattern.internalOps.push_back(op);
  pattern.primaryOps.push_back(op);
  pattern.scheduleContract = std::move(contract);
  (void)computeCandidateClosure(pattern.internalOps, deps.index);
  return pattern;
}

bool overlapsSelected(ArrayRef<Operation *> ops,
                      const DenseSet<Operation *> &selectedOps) {
  for (Operation *op : ops) {
    if (selectedOps.contains(op))
      return true;
  }
  return false;
}

void markSelected(ArrayRef<Operation *> ops, DenseSet<Operation *> &selectedOps) {
  for (Operation *op : ops)
    selectedOps.insert(op);
}

void assignFinalPatternIds(SmallVectorImpl<KernelPattern> &patterns) {
  for (auto [patternId, pattern] : llvm::enumerate(patterns)) {
    pattern.patternId = static_cast<unsigned>(patternId);
    pattern.kernelName =
        (llvm::Twine("kernel_") + llvm::Twine(patternId)).str();
  }
}

void printOpIdList(raw_ostream &os, ArrayRef<Operation *> ops,
                   const ProducerConsumerIndex &index) {
  os << "[";
  llvm::interleaveComma(ops, os, [&](Operation *op) {
    os << index.opIds.lookup(op).value;
  });
  os << "]";
}

} // namespace

KernelPatternGraph KernelPatternBuilder::build(
    ArrayRef<FusionCandidate> fusionCandidates,
    ArrayRef<MergedCandidate> mergedCandidates,
    ArrayRef<HorizontalFusionCandidate> horizontalCandidates,
    const DependencyAnalysisResult &deps) const {
  SmallVector<CandidateBuildRecord, 0> records;
  DenseMap<unsigned, const FusionCandidate *> fusionSources;
  DenseMap<unsigned, const MergedCandidate *> mergedSources;

  for (const FusionCandidate &candidate : fusionCandidates) {
    fusionSources.try_emplace(candidate.candidateId, &candidate);
    if (!candidate.legal)
      continue;
    CandidateBuildRecord record;
    record.candidate = buildCandidateFromFusion(candidate);
    record.sourceId = candidate.candidateId;
    records.push_back(std::move(record));
  }

  for (const MergedCandidate &candidate : mergedCandidates) {
    mergedSources.try_emplace(candidate.mergedCandidateId, &candidate);
    if (!candidate.legal)
      continue;
    CandidateBuildRecord record;
    record.candidate = buildCandidateFromMerged(candidate);
    record.sourceId = candidate.mergedCandidateId;
    records.push_back(std::move(record));
  }

  for (const HorizontalFusionCandidate &candidate : horizontalCandidates) {
    if (!candidate.legal)
      continue;
    CandidateBuildRecord record;
    record.candidate = buildCandidateFromHorizontal(
        candidate, fusionSources, mergedSources,
        static_cast<unsigned>(fusionCandidates.size()), deps);
    if (!record.candidate.closure.isClosed ||
        record.candidate.internalOps.empty())
      continue;
    record.sourceId = candidate.horizontalCandidateId;
    records.push_back(std::move(record));
  }

  llvm::sort(records, [&](const CandidateBuildRecord &lhs,
                          const CandidateBuildRecord &rhs) {
    if (lhs.candidate.benefitScore != rhs.candidate.benefitScore)
      return lhs.candidate.benefitScore > rhs.candidate.benefitScore;

    unsigned lhsPrimary = getFirstPrimaryOpId(lhs.candidate, deps.index);
    unsigned rhsPrimary = getFirstPrimaryOpId(rhs.candidate, deps.index);
    if (lhsPrimary != rhsPrimary)
      return lhsPrimary < rhsPrimary;

    if (lhs.candidate.sourceKind != rhs.candidate.sourceKind)
      return static_cast<unsigned>(lhs.candidate.sourceKind) <
             static_cast<unsigned>(rhs.candidate.sourceKind);

    return lhs.sourceId < rhs.sourceId;
  });

  KernelPatternGraph graph;
  for (auto [candidateId, record] : llvm::enumerate(records)) {
    record.candidate.candidateId = static_cast<unsigned>(candidateId);
    graph.nodes.push_back(std::move(record.candidate));
  }

  for (const KernelPatternCandidate &candidate : graph.nodes) {
    for (Operation *op : candidate.internalOps)
      graph.coveringMap[op].push_back(candidate.candidateId);
  }
  for (auto &entry : graph.coveringMap)
    sortUniqueIds(entry.second);

  DenseSet<KernelPatternEdgeKey> seenEdges;
  for (const auto &entry : graph.coveringMap) {
    ArrayRef<unsigned> coveringIds = entry.second;
    for (auto [idx, from] : llvm::enumerate(coveringIds)) {
      for (unsigned to : coveringIds.drop_front(idx + 1))
        appendEdge(graph.edges, seenEdges, from, to,
                   KernelPatternEdgeKind::Overlap);
    }
  }

  for (Operation *producer : deps.index.orderedOps) {
    auto producerCoveringIt = graph.coveringMap.find(producer);
    auto consumersIt = deps.index.consumers.find(producer);
    if (producerCoveringIt == graph.coveringMap.end() ||
        consumersIt == deps.index.consumers.end())
      continue;

    for (Operation *consumer : consumersIt->second) {
      auto consumerCoveringIt = graph.coveringMap.find(consumer);
      if (consumerCoveringIt == graph.coveringMap.end())
        continue;

      Value carriedValue = findCarriedValue(producer, consumer);
      for (unsigned from : producerCoveringIt->second) {
        for (unsigned to : consumerCoveringIt->second)
          appendEdge(graph.edges, seenEdges, from, to,
                     KernelPatternEdgeKind::DataDependency, carriedValue);
      }
    }
  }

  sortEdges(graph.edges);
  return graph;
}

SmallVector<KernelPattern>
KernelPartitioner::partition(const KernelPatternGraph &graph,
                             const DependencyAnalysisResult &deps) const {
  SmallVector<KernelPattern> patterns;
  DenseSet<Operation *> selectedOps;

  for (const KernelPatternCandidate &candidate : graph.nodes) {
    if (candidate.internalOps.empty() ||
        overlapsSelected(candidate.internalOps, selectedOps))
      continue;

    patterns.push_back(buildPatternFromCandidate(candidate));
    markSelected(candidate.internalOps, selectedOps);
  }

  for (Operation *op : deps.index.orderedOps) {
    if (selectedOps.contains(op))
      continue;

    ScheduleContract contract;
    if (!hasSchedulableRole(op, contract))
      continue;

    patterns.push_back(buildFallbackPattern(op, deps, std::move(contract)));
    selectedOps.insert(op);
  }

  llvm::sort(patterns, [&](const KernelPattern &lhs,
                           const KernelPattern &rhs) {
    return getMinOpId(lhs.internalOps, deps.index) <
           getMinOpId(rhs.internalOps, deps.index);
  });
  assignFinalPatternIds(patterns);
  return patterns;
}

void attachKernelPatternAttributes(ModuleOp module,
                                   ArrayRef<KernelPattern> patterns) {
  MLIRContext *context = module.getContext();
  for (const KernelPattern &pattern : patterns) {
    StringAttr kernelAttr = StringAttr::get(context, pattern.kernelName);
    DenseSet<Operation *> primarySet;
    for (Operation *op : pattern.primaryOps)
      primarySet.insert(op);

    for (Operation *op : pattern.internalOps) {
      op->setAttr(kKernelAttr, kernelAttr);
      if (primarySet.contains(op))
        op->setAttr(kPrimaryAttr, BoolAttr::get(context, true));
    }
  }
}

void emitKernelPatternGraphReport(raw_ostream &os,
                                  const KernelPatternGraph &graph,
                                  const ProducerConsumerIndex &index) {
  os << "KernelPatternGraph\n";
  for (const KernelPatternCandidate &candidate : graph.nodes) {
    os << "  pattern_candidate_id = " << candidate.candidateId
       << " source = \"" << stringifyCandidateKind(candidate.sourceKind)
       << "\" internal_ops = ";
    printOpIdList(os, candidate.internalOps, index);
    os << " primary_ops = ";
    printOpIdList(os, candidate.primaryOps, index);
    os << " benefit = " << candidate.benefitScore << "\n";
  }

  for (const KernelPatternEdge &edge : graph.edges) {
    os << "  edge = " << edge.from << " -> " << edge.to << " kind = \""
       << stringifyKernelPatternEdgeKind(edge.kind) << "\"\n";
  }
}

void emitKernelPartitionReport(raw_ostream &os,
                               ArrayRef<KernelPattern> patterns,
                               const ProducerConsumerIndex &index) {
  os << "KernelPartition\n";
  for (const KernelPattern &pattern : patterns) {
    os << "  kernel_pattern = \"" << pattern.kernelName
       << "\" internal_ops = ";
    printOpIdList(os, pattern.internalOps, index);
    os << " primary_ops = ";
    printOpIdList(os, pattern.primaryOps, index);
    os << "\n";
  }
}

} // namespace mlir::afir::ascend::v2::kernelize
