//===- HorizontalFusionAnalysis.cpp - Ascend V2 horizontal fusion --------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Kernelize/HorizontalFusionAnalysis.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <utility>

using namespace mlir;

namespace mlir::afir::ascend::v2::kernelize {
namespace {

struct HorizontalSource {
  unsigned sourceId = 0;
  ArrayRef<Operation *> internalOps;
  SmallVector<Value> shareableInputs;
  ScheduleContract scheduleContract;
};

struct HorizontalGroup {
  SmallVector<unsigned> sourceIndices;
  SmallVector<Value> sharedInputs;
};

void appendUniqueValue(SmallVectorImpl<Value> &values, Value value) {
  if (!llvm::is_contained(values, value))
    values.push_back(value);
}

void appendUniqueSourceIndex(SmallVectorImpl<unsigned> &sourceIndices,
                             unsigned sourceIndex) {
  if (!llvm::is_contained(sourceIndices, sourceIndex))
    sourceIndices.push_back(sourceIndex);
}

bool hasSameSourceIndices(ArrayRef<unsigned> lhs, ArrayRef<unsigned> rhs) {
  return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(),
                                                rhs.begin());
}

void sortSourceIndicesBySourceId(SmallVectorImpl<unsigned> &sourceIndices,
                                 ArrayRef<HorizontalSource> sources) {
  llvm::sort(sourceIndices, [&](unsigned lhs, unsigned rhs) {
    return sources[lhs].sourceId < sources[rhs].sourceId;
  });
}

bool hasOverlappingInternalOps(const HorizontalSource &lhs,
                               const HorizontalSource &rhs) {
  DenseSet<Operation *> lhsOps;
  for (Operation *op : lhs.internalOps)
    lhsOps.insert(op);

  for (Operation *op : rhs.internalOps) {
    if (lhsOps.contains(op))
      return true;
  }
  return false;
}

SmallVector<Value> collectShareableInputs(ArrayRef<Operation *> internalOps,
                                          const CandidateClosure &closure) {
  DenseSet<Value> externalInputSet;
  for (Value input : closure.externalInputs)
    externalInputSet.insert(input);

  SmallVector<Value> shareableInputs;
  for (Operation *op : internalOps) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp)
      continue;

    for (OpOperand *inputOperand : linalgOp.getDpsInputOperands()) {
      Value input = inputOperand->get();
      if (externalInputSet.contains(input))
        appendUniqueValue(shareableInputs, input);
    }
  }
  return shareableInputs;
}

bool reachesAnyInternalOp(ArrayRef<Operation *> starts,
                          ArrayRef<Operation *> targets) {
  DenseSet<Operation *> targetSet;
  for (Operation *op : targets)
    targetSet.insert(op);

  DenseSet<Operation *> visitedOps;
  DenseSet<Value> visitedValues;
  SmallVector<Value> worklist;
  for (Operation *op : starts) {
    for (Value result : op->getResults())
      worklist.push_back(result);
  }

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visitedValues.insert(value).second)
      continue;

    for (Operation *user : value.getUsers()) {
      if (targetSet.contains(user))
        return true;

      if (!visitedOps.insert(user).second)
        continue;
      for (Value result : user->getResults())
        worklist.push_back(result);
    }
  }
  return false;
}

bool areIndependent(const HorizontalSource &lhs, const HorizontalSource &rhs) {
  if (hasOverlappingInternalOps(lhs, rhs))
    return false;
  return !reachesAnyInternalOp(lhs.internalOps, rhs.internalOps) &&
         !reachesAnyInternalOp(rhs.internalOps, lhs.internalOps);
}

bool isEligibleGroup(ArrayRef<unsigned> sourceIndices,
                     ArrayRef<HorizontalSource> sources,
                     const KernelizeConfig &config) {
  if (sourceIndices.size() < 2 ||
      sourceIndices.size() > config.maxHorizontalFusionGroupSize)
    return false;

  for (auto [idx, lhsIndex] : llvm::enumerate(sourceIndices)) {
    const HorizontalSource &lhs = sources[lhsIndex];
    for (unsigned rhsIndex : sourceIndices.drop_front(idx + 1)) {
      const HorizontalSource &rhs = sources[rhsIndex];
      if (!areIndependent(lhs, rhs))
        return false;
    }
  }
  return true;
}

void appendGroup(SmallVectorImpl<HorizontalGroup> &groups,
                 ArrayRef<unsigned> sourceIndices, Value sharedInput) {
  for (HorizontalGroup &group : groups) {
    if (!hasSameSourceIndices(group.sourceIndices, sourceIndices))
      continue;
    appendUniqueValue(group.sharedInputs, sharedInput);
    return;
  }

  HorizontalGroup group;
  group.sourceIndices.append(sourceIndices.begin(), sourceIndices.end());
  group.sharedInputs.push_back(sharedInput);
  groups.push_back(std::move(group));
}

SmallVector<HorizontalSource>
collectSources(ArrayRef<FusionCandidate> fusionCandidates,
               ArrayRef<MergedCandidate> mergedCandidates) {
  SmallVector<HorizontalSource> sources;

  for (const FusionCandidate &candidate : fusionCandidates) {
    if (!candidate.legal || candidate.primaryOps.size() != 1 ||
        !candidate.closure.isClosed)
      continue;

    sources.push_back(HorizontalSource{
        candidate.candidateId, candidate.internalOps,
        collectShareableInputs(candidate.internalOps, candidate.closure),
        candidate.scheduleContract});
  }

  unsigned mergedIdOffset = static_cast<unsigned>(fusionCandidates.size());
  for (const MergedCandidate &candidate : mergedCandidates) {
    if (!candidate.legal || !candidate.closure.isClosed)
      continue;

    sources.push_back(HorizontalSource{
        mergedIdOffset + candidate.mergedCandidateId, candidate.internalOps,
        collectShareableInputs(candidate.internalOps, candidate.closure),
        candidate.scheduleContract});
  }

  return sources;
}

SmallVector<Value> collectExternalInputsInStableOrder(
    ArrayRef<HorizontalSource> sources,
    DenseMap<Value, SmallVector<unsigned>> &sourcesByInput) {
  SmallVector<Value> externalInputs;
  for (auto [sourceIndex, source] : llvm::enumerate(sources)) {
    for (Value input : source.shareableInputs) {
      appendUniqueValue(externalInputs, input);
      appendUniqueSourceIndex(sourcesByInput[input],
                              static_cast<unsigned>(sourceIndex));
    }
  }
  return externalInputs;
}

HorizontalFusionCandidate
buildHorizontalCandidate(const HorizontalGroup &group,
                         ArrayRef<HorizontalSource> sources) {
  HorizontalFusionCandidate candidate;
  candidate.sharedInputs.append(group.sharedInputs.begin(),
                                group.sharedInputs.end());

  for (unsigned sourceIndex : group.sourceIndices) {
    const HorizontalSource &source = sources[sourceIndex];
    candidate.siblingCandidateIds.push_back(source.sourceId);
    candidate.perGroupContracts.push_back(source.scheduleContract);
  }

  candidate.benefitScore =
      15 * static_cast<int64_t>(candidate.siblingCandidateIds.size() - 1);
  candidate.legal = true;
  return candidate;
}

void printCandidateIdList(raw_ostream &os, ArrayRef<unsigned> ids) {
  os << "[";
  llvm::interleaveComma(ids, os);
  os << "]";
}

} // namespace

SmallVector<HorizontalFusionCandidate>
HorizontalFusionAnalyzer::analyze(
    ArrayRef<FusionCandidate> fusionCandidates,
    ArrayRef<MergedCandidate> mergedCandidates,
    const DependencyAnalysisResult &,
    const KernelizeConfig &config) const {
  SmallVector<HorizontalSource> sources =
      collectSources(fusionCandidates, mergedCandidates);

  DenseMap<Value, SmallVector<unsigned>> sourcesByInput;
  SmallVector<Value> externalInputs =
      collectExternalInputsInStableOrder(sources, sourcesByInput);

  SmallVector<HorizontalGroup> groups;
  for (Value input : externalInputs) {
    SmallVector<unsigned> sourceIndices = sourcesByInput.lookup(input);
    sortSourceIndicesBySourceId(sourceIndices, sources);
    if (!isEligibleGroup(sourceIndices, sources, config))
      continue;
    appendGroup(groups, sourceIndices, input);
  }

  SmallVector<HorizontalFusionCandidate> horizontalCandidates;
  for (const HorizontalGroup &group : groups) {
    HorizontalFusionCandidate candidate =
        buildHorizontalCandidate(group, sources);
    candidate.horizontalCandidateId =
        static_cast<unsigned>(horizontalCandidates.size());
    horizontalCandidates.push_back(std::move(candidate));
  }

  return horizontalCandidates;
}

void emitHorizontalFusionReport(
    raw_ostream &os, ArrayRef<HorizontalFusionCandidate> horizontal) {
  os << "HorizontalFusionAnalysis\n";
  for (const HorizontalFusionCandidate &candidate : horizontal) {
    os << "  horizontal_candidate_id = "
       << candidate.horizontalCandidateId << " sibling_candidates = ";
    printCandidateIdList(os, candidate.siblingCandidateIds);
    os << " shared_inputs = " << candidate.sharedInputs.size()
       << " per_group_contracts = " << candidate.perGroupContracts.size()
       << " benefit = " << candidate.benefitScore << "\n";
  }
}

} // namespace mlir::afir::ascend::v2::kernelize
