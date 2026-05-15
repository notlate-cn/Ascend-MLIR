//===- FusionCandidateAnalysis.cpp - Ascend fusion candidates ---------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "FusionCandidateAnalysis.h"

#include "CandidateClosure.h"
#include "KernelizeTypes.h"
#include "OpRoleClassification.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <utility>

using namespace mlir;

namespace mlir::afir::ascend::kernelize {
namespace {

bool hasRole(ArrayRef<OpRole> roles, OpRole role) {
  return llvm::is_contained(roles, role);
}

ArrayRef<OpRole> getRoles(const OpRoleMap &roleMap, Operation *op) {
  auto it = roleMap.find(op);
  if (it == roleMap.end())
    return {};
  return it->second;
}

bool hasAnalyzedProducerCount(const ProducerConsumerIndex &index,
                              Operation *op, unsigned count) {
  auto it = index.producers.find(op);
  return (it == index.producers.end() ? 0 : it->second.size()) == count;
}

bool hasAnalyzedConsumerCount(const ProducerConsumerIndex &index,
                              Operation *op, unsigned count) {
  auto it = index.consumers.find(op);
  return (it == index.consumers.end() ? 0 : it->second.size()) == count;
}

ArrayRef<Operation *> getConsumers(const ProducerConsumerIndex &index,
                                   Operation *op) {
  auto it = index.consumers.find(op);
  if (it == index.consumers.end())
    return {};
  return it->second;
}

ArrayRef<Operation *> getProducers(const ProducerConsumerIndex &index,
                                   Operation *op) {
  auto it = index.producers.find(op);
  if (it == index.producers.end())
    return {};
  return it->second;
}

void sortByOpId(SmallVectorImpl<Operation *> &ops,
                const ProducerConsumerIndex &index) {
  llvm::sort(ops, [&](Operation *lhs, Operation *rhs) {
    return index.opIds.lookup(lhs).value < index.opIds.lookup(rhs).value;
  });
}

bool isVectorInjective(ArrayRef<OpRole> roles) {
  return hasRole(roles, OpRole::Vector) && hasRole(roles, OpRole::Injective);
}

bool isElementwiseChainOp(ArrayRef<OpRole> roles) {
  return isVectorInjective(roles) && !hasRole(roles, OpRole::Branch) &&
         !hasRole(roles, OpRole::LayoutTransform) &&
         !hasRole(roles, OpRole::Reduction) && !hasRole(roles, OpRole::Cube);
}

bool appendUniqueFamily(ScheduleContract &contract, StringRef family) {
  if (llvm::is_contained(contract.templateFamilies, family))
    return false;
  contract.templateFamilies.push_back(family.str());
  return true;
}

bool appendPrimaryFamily(ArrayRef<OpRole> roles, ScheduleContract &contract) {
  if (hasRole(roles, OpRole::Cube)) {
    appendUniqueFamily(contract, kOpRoleCube);
    return true;
  }
  if (hasRole(roles, OpRole::Reduction)) {
    appendUniqueFamily(contract, kOpRoleReduction);
    return true;
  }
  if (hasRole(roles, OpRole::Vector)) {
    appendUniqueFamily(contract, kOpRoleVector);
    return true;
  }
  return false;
}

bool populatePrimaryFamilies(ArrayRef<Operation *> primaryOps,
                             const OpRoleMap &roleMap,
                             ScheduleContract &contract) {
  bool foundFamily = false;
  for (Operation *primaryOp : primaryOps)
    foundFamily |= appendPrimaryFamily(getRoles(roleMap, primaryOp), contract);
  return foundFamily;
}

bool isFallbackEligible(ArrayRef<OpRole> roles) {
  if (roles.empty())
    return false;
  return !hasRole(roles, OpRole::Unsupported);
}

bool legalizeCandidate(FusionCandidate &candidate,
                       const DependencyAnalysisResult &deps,
                       const OpRoleMap &roleMap,
                       const KernelizeConfig &config) {
  if (candidate.internalOps.size() > config.maxOpsPerCandidate) {
    candidate.rejectionReason = "TooManyInternalOps";
    return false;
  }

  if (candidate.primaryOps.size() > config.maxPrimaryRolesPerCandidate) {
    candidate.rejectionReason = "TooManyPrimaryRoles";
    return false;
  }

  candidate.closure = computeCandidateClosure(candidate.internalOps, deps.index);
  if (!candidate.closure.isClosed) {
    candidate.rejectionReason = candidate.closure.failureReason;
    return false;
  }

  if (candidate.primaryOps.empty() ||
      !populatePrimaryFamilies(candidate.primaryOps, roleMap,
                               candidate.scheduleContract)) {
    candidate.rejectionReason = "TemplateFamilyUnavailable";
    return false;
  }

  candidate.legal = true;
  return true;
}

void appendLegalCandidate(SmallVectorImpl<FusionCandidate> &candidates,
                          FusionCandidate candidate,
                          const DependencyAnalysisResult &deps,
                          const OpRoleMap &roleMap,
                          const KernelizeConfig &config) {
  if (legalizeCandidate(candidate, deps, roleMap, config))
    candidates.push_back(std::move(candidate));
}

FusionCandidate buildElementwiseChainCandidate(
    Operation *seed, const DependencyAnalysisResult &deps,
    const OpRoleMap &roleMap) {
  FusionCandidate candidate;
  candidate.kind = CandidateKind::Fusion;
  candidate.primitive = KernelizePrimitiveKind::ElementwiseChain;
  candidate.primaryOps.push_back(seed);
  candidate.internalOps.push_back(seed);

  Operation *current = seed;
  while (hasAnalyzedConsumerCount(deps.index, current, 1)) {
    Operation *consumer = getConsumers(deps.index, current).front();
    if (!isElementwiseChainOp(getRoles(roleMap, consumer)) ||
        !hasAnalyzedProducerCount(deps.index, consumer, 1))
      break;

    candidate.internalOps.push_back(consumer);
    current = consumer;
  }

  sortByOpId(candidate.internalOps, deps.index);
  candidate.benefitScore =
      10 * static_cast<int64_t>(candidate.internalOps.size() - 1);
  return candidate;
}

FusionCandidate buildConsumerIntoPrimaryCandidate(
    Operation *seed, const DependencyAnalysisResult &deps,
    const OpRoleMap &roleMap) {
  FusionCandidate candidate;
  candidate.kind = CandidateKind::Fusion;
  candidate.primitive = KernelizePrimitiveKind::ConsumerIntoPrimary;
  candidate.internalOps.push_back(seed);

  ArrayRef<OpRole> seedRoles = getRoles(roleMap, seed);
  bool reductionSeed = hasRole(seedRoles, OpRole::Reduction) &&
                       !hasRole(seedRoles, OpRole::Cube);
  SmallVector<Operation *> absorbedConsumers;
  unsigned absorbedConsumerCount = 0;
  for (Operation *consumer : getConsumers(deps.index, seed)) {
    if (!isVectorInjective(getRoles(roleMap, consumer)) ||
        !hasAnalyzedProducerCount(deps.index, consumer, 1))
      continue;
    candidate.internalOps.push_back(consumer);
    absorbedConsumers.push_back(consumer);
    ++absorbedConsumerCount;
  }

  if (reductionSeed && !absorbedConsumers.empty())
    candidate.primaryOps.append(absorbedConsumers.begin(),
                                absorbedConsumers.end());
  else
    candidate.primaryOps.push_back(seed);

  sortByOpId(candidate.internalOps, deps.index);
  sortByOpId(candidate.primaryOps, deps.index);
  candidate.benefitScore = 20 + 5 * static_cast<int64_t>(absorbedConsumerCount);
  return candidate;
}

FusionCandidate buildReductionInliningCandidate(
    Operation *seed, const DependencyAnalysisResult &deps,
    const OpRoleMap &roleMap) {
  FusionCandidate candidate;
  candidate.kind = CandidateKind::Fusion;
  candidate.primitive = KernelizePrimitiveKind::ReductionInlining;
  candidate.primaryOps.push_back(seed);
  candidate.internalOps.push_back(seed);

  Operation *producer = getProducers(deps.index, seed).front();
  candidate.internalOps.push_back(producer);
  sortByOpId(candidate.internalOps, deps.index);
  candidate.benefitScore = 10;
  return candidate;
}

FusionCandidate buildFallbackSingleOpCandidate(Operation *seed) {
  FusionCandidate candidate;
  candidate.kind = CandidateKind::FallbackSingleOp;
  candidate.primitive = KernelizePrimitiveKind::FallbackSingleOp;
  candidate.primaryOps.push_back(seed);
  candidate.internalOps.push_back(seed);
  candidate.benefitScore = 1;
  return candidate;
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

SmallVector<FusionCandidate>
FusionCandidateAnalyzer::analyze(const DependencyAnalysisResult &deps,
                                 const OpRoleMap &roleMap,
                                 const KernelizeConfig &config) const {
  SmallVector<FusionCandidate> candidates;

  for (Operation *seed : deps.index.orderedOps) {
    if (!isElementwiseChainOp(getRoles(roleMap, seed)))
      continue;

    FusionCandidate candidate =
        buildElementwiseChainCandidate(seed, deps, roleMap);
    if (candidate.internalOps.size() > 1)
      appendLegalCandidate(candidates, std::move(candidate), deps, roleMap,
                           config);
  }

  for (Operation *seed : deps.index.orderedOps) {
    ArrayRef<OpRole> roles = getRoles(roleMap, seed);
    if (!hasRole(roles, OpRole::Cube) && !hasRole(roles, OpRole::Reduction))
      continue;

    FusionCandidate candidate =
        buildConsumerIntoPrimaryCandidate(seed, deps, roleMap);
    if (candidate.internalOps.size() > 1)
      appendLegalCandidate(candidates, std::move(candidate), deps, roleMap,
                           config);
  }

  for (Operation *seed : deps.index.orderedOps) {
    ArrayRef<OpRole> roles = getRoles(roleMap, seed);
    if (!hasRole(roles, OpRole::Reduction) ||
        !hasAnalyzedProducerCount(deps.index, seed, 1))
      continue;

    Operation *producer = getProducers(deps.index, seed).front();
    ArrayRef<OpRole> producerRoles = getRoles(roleMap, producer);
    if (!isVectorInjective(producerRoles) ||
        !hasAnalyzedConsumerCount(deps.index, producer, 1) ||
        hasRole(producerRoles, OpRole::Branch))
      continue;

    appendLegalCandidate(candidates,
                         buildReductionInliningCandidate(seed, deps, roleMap),
                         deps, roleMap, config);
  }

  for (Operation *seed : deps.index.orderedOps) {
    if (!isFallbackEligible(getRoles(roleMap, seed)))
      continue;

    appendLegalCandidate(candidates, buildFallbackSingleOpCandidate(seed), deps,
                         roleMap, config);
  }

  for (auto [candidateId, candidate] : llvm::enumerate(candidates))
    candidate.candidateId = static_cast<unsigned>(candidateId);

  return candidates;
}

void emitFusionCandidateReport(raw_ostream &os,
                               ArrayRef<FusionCandidate> candidates,
                               const ProducerConsumerIndex &index) {
  os << "FusionCandidateAnalysis\n";
  for (const FusionCandidate &candidate : candidates) {
    os << "  candidate_id = " << candidate.candidateId << " kind = \""
       << stringifyCandidateKind(candidate.kind) << "\" primitive = \""
       << stringifyKernelizePrimitiveKind(candidate.primitive)
       << "\" primary_ops = ";
    printOpIdList(os, candidate.primaryOps, index);
    os << " internal_ops = ";
    printOpIdList(os, candidate.internalOps, index);
    os << " closed = " << (candidate.closure.isClosed ? "true" : "false")
       << " benefit = " << candidate.benefitScore << " families = ";
    printStringList(os, candidate.scheduleContract.templateFamilies);
    os << "\n";
  }
}

} // namespace mlir::afir::ascend::kernelize
