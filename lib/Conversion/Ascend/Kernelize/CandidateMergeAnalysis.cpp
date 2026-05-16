//===- CandidateMergeAnalysis.cpp - Ascend candidate merge ------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "CandidateMergeAnalysis.h"

#include "CandidateClosure.h"
#include "KernelizeFamilyResolver.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

using namespace mlir;

namespace mlir::afir::ascend::kernelize {
namespace {

using CandidatePair = std::pair<unsigned, unsigned>;

struct MergeSource {
  SmallVector<unsigned, 4> sourceCandidateIds;
  SmallVector<Operation *, 8> internalOps;
  SmallVector<Operation *, 4> primaryOps;
  SmallVector<KernelizePrimitiveKind, 4> primitiveCombo;
  ScheduleContract scheduleContract;
  int64_t benefitScore = 0;
};

uint64_t packPair(unsigned lhs, unsigned rhs) {
  return (static_cast<uint64_t>(lhs) << 32) | rhs;
}

void sortUniqueCandidateIds(SmallVectorImpl<unsigned> &ids) {
  llvm::sort(ids);
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
}

void sortUniqueOpsByOpId(SmallVectorImpl<Operation *> &ops,
                         const ProducerConsumerIndex &index) {
  llvm::sort(ops, [&](Operation *lhs, Operation *rhs) {
    return index.opIds.lookup(lhs).value < index.opIds.lookup(rhs).value;
  });
  ops.erase(std::unique(ops.begin(), ops.end()), ops.end());
}


DenseMap<Operation *, SmallVector<unsigned>>
buildCoveringSourceIds(ArrayRef<MergeSource> sources) {
  DenseMap<Operation *, SmallVector<unsigned>> coveringCandidateIds;
  for (auto [sourceId, source] : llvm::enumerate(sources)) {
    for (Operation *op : source.internalOps)
      coveringCandidateIds[op].push_back(static_cast<unsigned>(sourceId));
  }

  for (auto &entry : coveringCandidateIds)
    sortUniqueCandidateIds(entry.second);
  return coveringCandidateIds;
}

SmallVector<CandidatePair>
collectAdjacentCandidatePairs(
    const ProducerConsumerIndex &index,
    const DenseMap<Operation *, SmallVector<unsigned>> &coveringCandidateIds) {
  SmallVector<CandidatePair> pairs;
  DenseSet<uint64_t> seenPairs;

  for (Operation *producer : index.orderedOps) {
    auto producerCoveringIt = coveringCandidateIds.find(producer);
    auto consumersIt = index.consumers.find(producer);
    if (producerCoveringIt == coveringCandidateIds.end() ||
        consumersIt == index.consumers.end())
      continue;

    for (Operation *consumer : consumersIt->second) {
      auto consumerCoveringIt = coveringCandidateIds.find(consumer);
      if (consumerCoveringIt == coveringCandidateIds.end())
        continue;

      for (unsigned producerCandidateId : producerCoveringIt->second) {
        for (unsigned consumerCandidateId : consumerCoveringIt->second) {
          if (producerCandidateId == consumerCandidateId)
            continue;

          uint64_t packed =
              packPair(producerCandidateId, consumerCandidateId);
          if (!seenPairs.insert(packed).second)
            continue;
          pairs.push_back({producerCandidateId, consumerCandidateId});
        }
      }
    }
  }

  llvm::sort(pairs);
  return pairs;
}

std::string buildSourceKey(const MergeSource &source,
                           const ProducerConsumerIndex &index) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << "ops:";
  for (Operation *op : source.internalOps)
    os << index.opIds.lookup(op).value << ",";
  os << "|primary:";
  for (Operation *op : source.primaryOps)
    os << index.opIds.lookup(op).value << ",";
  os << "|families:";
  for (const std::string &family : source.scheduleContract.templateFamilies)
    os << family << ",";
  return os.str();
}

void appendUniqueCandidateIds(SmallVectorImpl<unsigned> &ids,
                              ArrayRef<unsigned> newIds) {
  ids.append(newIds.begin(), newIds.end());
  sortUniqueCandidateIds(ids);
}

MergeSource buildMergeSource(const FusionCandidate &candidate) {
  MergeSource source;
  source.sourceCandidateIds.push_back(candidate.candidateId);
  source.internalOps.append(candidate.internalOps.begin(),
                            candidate.internalOps.end());
  source.primaryOps.append(candidate.primaryOps.begin(),
                           candidate.primaryOps.end());
  source.primitiveCombo.push_back(candidate.primitive);
  source.scheduleContract = candidate.scheduleContract;
  source.benefitScore = candidate.benefitScore;
  return source;
}

MergeSource buildMergeSource(const MergedCandidate &candidate) {
  MergeSource source;
  source.sourceCandidateIds.append(candidate.sourceCandidateIds.begin(),
                                   candidate.sourceCandidateIds.end());
  source.internalOps.append(candidate.internalOps.begin(),
                            candidate.internalOps.end());
  source.primaryOps.append(candidate.primaryOps.begin(),
                           candidate.primaryOps.end());
  source.primitiveCombo.append(candidate.primitiveCombo.begin(),
                               candidate.primitiveCombo.end());
  source.scheduleContract = candidate.scheduleContract;
  source.benefitScore = candidate.benefitScore;
  return source;
}

MergedCandidate buildMergedCandidate(const MergeSource &lhs,
                                     const MergeSource &rhs,
                                     const DependencyAnalysisResult &deps,
                                     const KernelizeConfig &config) {
  MergedCandidate merged;
  appendUniqueCandidateIds(merged.sourceCandidateIds, lhs.sourceCandidateIds);
  appendUniqueCandidateIds(merged.sourceCandidateIds, rhs.sourceCandidateIds);

  for (const MergeSource *source : {&lhs, &rhs}) {
    merged.primaryOps.append(source->primaryOps.begin(),
                             source->primaryOps.end());
    merged.internalOps.append(source->internalOps.begin(),
                              source->internalOps.end());
    merged.primitiveCombo.append(source->primitiveCombo.begin(),
                                 source->primitiveCombo.end());
    merged.benefitScore += source->benefitScore;
  }

  sortUniqueOpsByOpId(merged.primaryOps, deps.index);
  sortUniqueOpsByOpId(merged.internalOps, deps.index);

  if (merged.internalOps.size() == lhs.internalOps.size() ||
      merged.internalOps.size() == rhs.internalOps.size()) {
    merged.rejectionReason = "SubsumedCandidate";
    return merged;
  }

  if (merged.internalOps.size() > config.maxOpsPerCandidate) {
    merged.rejectionReason = "TooManyInternalOps";
    return merged;
  }

  if (merged.primaryOps.size() > config.maxPrimaryRolesPerCandidate) {
    merged.rejectionReason = "TooManyPrimaryRoles";
    return merged;
  }

  KernelizeFamilyResolution resolution = resolveKernelizeTemplateFamilies(
      lhs.scheduleContract.templateFamilies,
      rhs.scheduleContract.templateFamilies, merged.primitiveCombo);
  merged.scheduleContract.templateFamilies =
      std::move(resolution.templateFamilies);
  merged.familyResolverName = std::move(resolution.resolverName);
  if (merged.scheduleContract.templateFamilies.empty()) {
    merged.rejectionReason = "TemplateFamilyUnavailable";
    return merged;
  }

  merged.closure = computeCandidateClosure(merged.internalOps, deps.index);
  if (!merged.closure.isClosed) {
    merged.rejectionReason = merged.closure.failureReason;
    return merged;
  }

  merged.legal = true;
  return merged;
}

void printCandidateIdList(raw_ostream &os, ArrayRef<unsigned> ids) {
  os << "[";
  llvm::interleaveComma(ids, os);
  os << "]";
}

void printOpIdList(raw_ostream &os, ArrayRef<Operation *> ops,
                   const ProducerConsumerIndex &index) {
  os << "[";
  llvm::interleaveComma(ops, os, [&](Operation *op) {
    os << index.opIds.lookup(op).value;
  });
  os << "]";
}

void printStringList(raw_ostream &os, ArrayRef<std::string> strings) {
  os << "[";
  llvm::interleaveComma(strings, os, [&](const std::string &value) {
    os << "\"" << value << "\"";
  });
  os << "]";
}

void printPrimitiveList(raw_ostream &os,
                        ArrayRef<KernelizePrimitiveKind> primitives) {
  os << "[";
  llvm::interleaveComma(primitives, os, [&](KernelizePrimitiveKind primitive) {
    os << "\"" << stringifyKernelizePrimitiveKind(primitive) << "\"";
  });
  os << "]";
}

} // namespace

SmallVector<MergedCandidate>
CandidateMergeAnalyzer::analyze(ArrayRef<FusionCandidate> candidates,
                                const DependencyAnalysisResult &deps,
                                const KernelizeConfig &config) const {
  SmallVector<MergeSource, 8> sources;
  llvm::StringSet<> seenSourceKeys;
  for (const FusionCandidate &candidate : candidates) {
    if (!candidate.legal || candidate.kind != CandidateKind::Fusion)
      continue;
    sources.push_back(buildMergeSource(candidate));
  }

  for (const MergeSource &source : sources)
    seenSourceKeys.insert(buildSourceKey(source, deps.index));

  SmallVector<MergedCandidate> mergedCandidates;
  bool changed = true;
  while (changed) {
    changed = false;
    DenseMap<Operation *, SmallVector<unsigned>> coveringCandidateIds =
        buildCoveringSourceIds(sources);
    SmallVector<CandidatePair> pairs =
        collectAdjacentCandidatePairs(deps.index, coveringCandidateIds);

    SmallVector<MergeSource, 8> newSources;
    for (auto [lhsId, rhsId] : pairs) {
      if (lhsId >= sources.size() || rhsId >= sources.size())
        continue;

      MergedCandidate merged =
          buildMergedCandidate(sources[lhsId], sources[rhsId], deps, config);
      if (!merged.legal)
        continue;

      MergeSource mergedSource = buildMergeSource(merged);
      std::string sourceKey = buildSourceKey(mergedSource, deps.index);
      if (!seenSourceKeys.insert(sourceKey).second)
        continue;

      merged.mergedCandidateId = static_cast<unsigned>(mergedCandidates.size());
      mergedCandidates.push_back(std::move(merged));
      newSources.push_back(std::move(mergedSource));
      changed = true;
    }
    sources.append(newSources.begin(), newSources.end());
  }

  return mergedCandidates;
}

void emitCandidateMergeReport(raw_ostream &os,
                              ArrayRef<MergedCandidate> merged,
                              const ProducerConsumerIndex &index) {
  os << "CandidateMergeAnalysis\n";
  for (const MergedCandidate &candidate : merged) {
    os << "  merged_candidate_id = " << candidate.mergedCandidateId
       << " source_candidates = ";
    printCandidateIdList(os, candidate.sourceCandidateIds);
    os << " primary_ops = ";
    printOpIdList(os, candidate.primaryOps, index);
    os << " internal_ops = ";
    printOpIdList(os, candidate.internalOps, index);
    os << " primitive_combo = ";
    printPrimitiveList(os, candidate.primitiveCombo);
    os << " closed = " << (candidate.closure.isClosed ? "true" : "false")
       << " benefit = " << candidate.benefitScore << " families = ";
    printStringList(os, candidate.scheduleContract.templateFamilies);
    os << " family_resolver = \"" << candidate.familyResolverName << "\"";
    os << "\n";
  }
}

} // namespace mlir::afir::ascend::kernelize
