//===- CandidateMergeAnalysis.cpp - Ascend candidate merge ------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/CandidateMergeAnalysis.h"

#include "Conversion/Ascend/Kernelize/CandidateClosure.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

using namespace mlir;

namespace mlir::afir::ascend::kernelize {
namespace {

using CandidatePair = std::pair<unsigned, unsigned>;

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

bool containsFamily(ArrayRef<std::string> families, StringRef family) {
  for (const std::string &value : families) {
    if (value == family)
      return true;
  }
  return false;
}

std::optional<StringRef> resolveTableFamily(ArrayRef<std::string> lhsFamilies,
                                            ArrayRef<std::string> rhsFamilies) {
  if (containsFamily(lhsFamilies, "vector") &&
      containsFamily(rhsFamilies, "reduction"))
    return StringRef("reduction");

  if (containsFamily(lhsFamilies, "cube") &&
      containsFamily(rhsFamilies, "vector"))
    return StringRef("cube");

  if (containsFamily(lhsFamilies, "vector") &&
      containsFamily(rhsFamilies, "vector"))
    return StringRef("vector");

  return std::nullopt;
}

SmallVector<std::string, 2>
resolveTemplateFamilies(const FusionCandidate &lhs,
                        const FusionCandidate &rhs) {
  ArrayRef<std::string> lhsFamilies = lhs.scheduleContract.templateFamilies;
  ArrayRef<std::string> rhsFamilies = rhs.scheduleContract.templateFamilies;
  SmallVector<std::string, 2> resolved;

  if (std::optional<StringRef> tableFamily =
          resolveTableFamily(lhsFamilies, rhsFamilies)) {
    resolved.push_back(tableFamily->str());
    return resolved;
  }

  for (const std::string &lhsFamily : lhsFamilies) {
    if (!containsFamily(rhsFamilies, lhsFamily) ||
        containsFamily(resolved, lhsFamily))
      continue;
    resolved.push_back(lhsFamily);
  }
  return resolved;
}

DenseMap<Operation *, SmallVector<unsigned>>
buildCoveringCandidateIds(ArrayRef<const FusionCandidate *> legalCandidates) {
  DenseMap<Operation *, SmallVector<unsigned>> coveringCandidateIds;
  for (const FusionCandidate *candidate : legalCandidates) {
    for (Operation *op : candidate->internalOps)
      coveringCandidateIds[op].push_back(candidate->candidateId);
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

MergedCandidate buildMergedCandidate(const FusionCandidate &lhs,
                                     const FusionCandidate &rhs,
                                     const DependencyAnalysisResult &deps,
                                     const KernelizeConfig &config) {
  MergedCandidate merged;
  merged.sourceCandidateIds.push_back(lhs.candidateId);
  merged.sourceCandidateIds.push_back(rhs.candidateId);

  for (const FusionCandidate *source : {&lhs, &rhs}) {
    merged.primaryOps.append(source->primaryOps.begin(),
                             source->primaryOps.end());
    merged.internalOps.append(source->internalOps.begin(),
                              source->internalOps.end());
    merged.primitiveCombo.push_back(source->primitive);
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

  merged.scheduleContract.templateFamilies = resolveTemplateFamilies(lhs, rhs);
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

} // namespace

SmallVector<MergedCandidate>
CandidateMergeAnalyzer::analyze(ArrayRef<FusionCandidate> candidates,
                                const DependencyAnalysisResult &deps,
                                const KernelizeConfig &config) const {
  SmallVector<const FusionCandidate *> legalCandidates;
  DenseMap<unsigned, const FusionCandidate *> candidatesById;
  for (const FusionCandidate &candidate : candidates) {
    if (!candidate.legal || candidate.kind != CandidateKind::Fusion)
      continue;
    legalCandidates.push_back(&candidate);
    candidatesById.try_emplace(candidate.candidateId, &candidate);
  }

  DenseMap<Operation *, SmallVector<unsigned>> coveringCandidateIds =
      buildCoveringCandidateIds(legalCandidates);
  SmallVector<CandidatePair> pairs =
      collectAdjacentCandidatePairs(deps.index, coveringCandidateIds);

  SmallVector<MergedCandidate> mergedCandidates;
  for (auto [lhsId, rhsId] : pairs) {
    const FusionCandidate *lhs = candidatesById.lookup(lhsId);
    const FusionCandidate *rhs = candidatesById.lookup(rhsId);
    if (!lhs || !rhs)
      continue;

    MergedCandidate merged =
        buildMergedCandidate(*lhs, *rhs, deps, config);
    if (!merged.legal)
      continue;

    merged.mergedCandidateId =
        static_cast<unsigned>(mergedCandidates.size());
    mergedCandidates.push_back(std::move(merged));
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
    printStringList(os, candidate.primitiveCombo);
    os << " closed = " << (candidate.closure.isClosed ? "true" : "false")
       << " benefit = " << candidate.benefitScore << " families = ";
    printStringList(os, candidate.scheduleContract.templateFamilies);
    os << "\n";
  }
}

} // namespace mlir::afir::ascend::kernelize
