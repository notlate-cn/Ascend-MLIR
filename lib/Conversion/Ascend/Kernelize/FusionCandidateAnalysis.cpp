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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
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

KernelizeSeedPolicy getSeedPolicy(Operation *op,
                                  const DependencyAnalysisResult &deps) {
  auto it = deps.summaries.find(op);
  if (it == deps.summaries.end())
    return KernelizeSeedPolicy::NeverSeed;
  return it->second.seedPolicy;
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

bool isHandwrittenPrimaryCandidate(ArrayRef<OpRole> roles) {
  return hasRole(roles, OpRole::Cube) || hasRole(roles, OpRole::Vector) ||
         hasRole(roles, OpRole::Reduction);
}

bool isAttentionLikeHandwrittenOp(ArrayRef<OpRole> roles) {
  return hasRole(roles, OpRole::Cube) || hasRole(roles, OpRole::Reduction) ||
         isVectorInjective(roles);
}

unsigned getHandwrittenPrimaryPriority(ArrayRef<OpRole> roles) {
  if (hasRole(roles, OpRole::Cube))
    return 0;
  if (hasRole(roles, OpRole::Vector))
    return 1;
  if (hasRole(roles, OpRole::Reduction))
    return 2;
  return 3;
}

std::optional<int64_t> getHandwrittenGroup(Operation *op) {
  auto group = op->getAttrOfType<IntegerAttr>(kHandwrittenGroupAttr);
  if (!group)
    return std::nullopt;
  return group.getInt();
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

  bool reductionSeed =
      getSeedPolicy(seed, deps) == KernelizeSeedPolicy::NonSeedWhenFused;
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

FusionCandidate buildHandwrittenPatternCandidate(
    ArrayRef<Operation *> groupOps, const DependencyAnalysisResult &deps,
    const OpRoleMap &roleMap) {
  FusionCandidate candidate;
  candidate.kind = CandidateKind::HandwrittenPattern;
  candidate.primitive = KernelizePrimitiveKind::HandwrittenPattern;
  candidate.internalOps.append(groupOps.begin(), groupOps.end());

  Operation *primary = nullptr;
  unsigned primaryPriority = 3;
  for (Operation *op : groupOps) {
    ArrayRef<OpRole> roles = getRoles(roleMap, op);
    if (!isHandwrittenPrimaryCandidate(roles))
      continue;
    unsigned priority = getHandwrittenPrimaryPriority(roles);
    if (primary && priority >= primaryPriority)
      continue;
    primary = op;
    primaryPriority = priority;
  }
  if (primary)
    candidate.primaryOps.push_back(primary);

  sortByOpId(candidate.internalOps, deps.index);
  sortByOpId(candidate.primaryOps, deps.index);
  candidate.benefitScore =
      1000 + 20 * static_cast<int64_t>(candidate.internalOps.size());
  return candidate;
}

std::optional<SmallVector<Operation *, 8>>
collectAttentionLikeHandwrittenPattern(Operation *seed,
                                       const DependencyAnalysisResult &deps,
                                       const OpRoleMap &roleMap,
                                       const KernelizeConfig &config) {
  if (!hasRole(getRoles(roleMap, seed), OpRole::Cube))
    return std::nullopt;

  SmallVector<Operation *, 8> groupOps;
  llvm::DenseSet<Operation *> seen;
  SmallVector<Operation *, 4> seedReductionConsumers;
  for (Operation *consumer : getConsumers(deps.index, seed))
    if (hasRole(getRoles(roleMap, consumer), OpRole::Reduction))
      seedReductionConsumers.push_back(consumer);
  if (seedReductionConsumers.size() != 1)
    return std::nullopt;

  Operation *current = seedReductionConsumers.front();
  bool hasVector = false;
  unsigned cubeCount = 1;
  seen.insert(seed);
  seen.insert(current);
  groupOps.push_back(seed);
  groupOps.push_back(current);

  while (true) {
    SmallVector<Operation *, 4> eligibleConsumers;
    for (Operation *consumer : getConsumers(deps.index, current)) {
      if (seen.contains(consumer))
        return std::nullopt;
      if (isAttentionLikeHandwrittenOp(getRoles(roleMap, consumer)))
        eligibleConsumers.push_back(consumer);
    }

    if (eligibleConsumers.empty())
      break;
    if (eligibleConsumers.size() != 1)
      return std::nullopt;

    Operation *next = eligibleConsumers.front();
    ArrayRef<OpRole> nextRoles = getRoles(roleMap, next);
    bool isCube = hasRole(nextRoles, OpRole::Cube);
    if (!isCube && !hasRole(nextRoles, OpRole::Reduction) &&
        !isVectorInjective(nextRoles))
      return std::nullopt;

    seen.insert(next);
    groupOps.push_back(next);
    if (isVectorInjective(nextRoles))
      hasVector = true;
    if (groupOps.size() > config.maxOpsPerCandidate)
      return std::nullopt;

    if (isCube) {
      ++cubeCount;
      if (cubeCount != 2)
        return std::nullopt;
      break;
    }

    current = next;
  }

  if (cubeCount < 2 || !hasVector)
    return std::nullopt;
  sortByOpId(groupOps, deps.index);
  return groupOps;
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
    bool consumerIntoPrimarySeed =
        hasRole(roles, OpRole::Cube) ||
        getSeedPolicy(seed, deps) == KernelizeSeedPolicy::NonSeedWhenFused;
    if (!consumerIntoPrimarySeed)
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

  llvm::DenseSet<Operation *> attentionGroupedOps;
  for (Operation *seed : deps.index.orderedOps) {
    if (attentionGroupedOps.contains(seed))
      continue;
    std::optional<SmallVector<Operation *, 8>> groupOps =
        collectAttentionLikeHandwrittenPattern(seed, deps, roleMap, config);
    if (!groupOps)
      continue;
    for (Operation *op : *groupOps)
      attentionGroupedOps.insert(op);
    appendLegalCandidate(candidates,
                         buildHandwrittenPatternCandidate(*groupOps, deps,
                                                          roleMap),
                         deps, roleMap, config);
  }

  DenseMap<int64_t, SmallVector<Operation *, 4>> handwrittenGroups;
  for (Operation *op : deps.index.orderedOps) {
    std::optional<int64_t> group = getHandwrittenGroup(op);
    if (!group)
      continue;
    handwrittenGroups[*group].push_back(op);
  }
  SmallVector<int64_t, 4> handwrittenGroupIds;
  for (const auto &entry : handwrittenGroups)
    handwrittenGroupIds.push_back(entry.first);
  llvm::sort(handwrittenGroupIds);
  for (int64_t groupId : handwrittenGroupIds) {
    SmallVector<Operation *, 4> &groupOps = handwrittenGroups[groupId];
    if (groupOps.size() < 2)
      continue;
    sortByOpId(groupOps, deps.index);
    appendLegalCandidate(candidates,
                         buildHandwrittenPatternCandidate(groupOps, deps,
                                                          roleMap),
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
