//===- TargetCostModel.cpp - Ascend target cost model --------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/Ascend/TargetCostModel.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <tuple>

using namespace mlir;

namespace mlir::ascend {
namespace {

std::string makeRateKey(StringRef section, StringRef name) {
  std::string key = section.str();
  key += ".";
  key += name.str();
  return key;
}

int sectionPriority(StringRef section) {
  if (section == "AICoreMemoryRates")
    return 0;
  if (section == "VectorCoreMemoryRates")
    return 1;
  return 2;
}

int64_t startupCyclesFor(PathKind kind) {
  switch (kind) {
  case PathKind::DirectCopy:
    return 1;
  case PathKind::Load2D:
    return 2;
  case PathKind::Load2DTranspose:
    return 3;
  case PathKind::FixPipe:
    return 4;
  case PathKind::QueueTransfer:
    return 0;
  }
  llvm_unreachable("unknown path kind");
}

bool overlapsCompute(PathKind kind) {
  return kind == PathKind::QueueTransfer;
}

SmallVector<StringRef> getRateCandidates(const PathEdge &edge,
                                         PathKind kind) {
  if (edge.srcPlace == MemoryPlace::A1 && edge.dstPlace == MemoryPlace::A2)
    return {"l1_to_l0_a_rate"};
  if (edge.srcPlace == MemoryPlace::B1 && edge.dstPlace == MemoryPlace::B2)
    return {"l1_to_l0_b_rate"};
  if (edge.srcPlace == MemoryPlace::CO1 && edge.dstPlace == MemoryPlace::VECIN)
    return {"l0_c_to_ub_rate"};
  if (edge.srcPlace == MemoryPlace::VECOUT && edge.dstPlace == MemoryPlace::GM)
    return {"ub_to_ddr_rate"};
  if (edge.srcPlace == MemoryPlace::GM)
    return {"ddr_read_rate", "ddr_rate"};
  if (edge.dstPlace == MemoryPlace::GM)
    return {"ddr_write_rate", "ub_to_ddr_rate", "ddr_rate"};

  switch (kind) {
  case PathKind::DirectCopy:
    return {"ddr_rate"};
  case PathKind::Load2D:
  case PathKind::Load2DTranspose:
    return {"ddr_read_rate", "ddr_rate"};
  case PathKind::FixPipe:
    return {"ddr_write_rate", "ub_to_ddr_rate", "ddr_rate"};
  case PathKind::QueueTransfer:
    return {"l0_c_to_ub_rate"};
  }
  llvm_unreachable("unknown path kind");
}

SmallVector<StringRef> getSectionCandidates(const PathEdge &edge) {
  if ((edge.srcPlace == MemoryPlace::GM && edge.dstPlace == MemoryPlace::VECIN) ||
      (edge.srcPlace == MemoryPlace::VECOUT && edge.dstPlace == MemoryPlace::GM))
    return {"VectorCoreMemoryRates", "AICoreMemoryRates"};
  return {"AICoreMemoryRates", "VectorCoreMemoryRates"};
}

StringRef stringifyPathKind(PathKind kind) {
  switch (kind) {
  case PathKind::DirectCopy:
    return "DirectCopy";
  case PathKind::Load2D:
    return "Load2D";
  case PathKind::Load2DTranspose:
    return "Load2DTranspose";
  case PathKind::FixPipe:
    return "FixPipe";
  case PathKind::QueueTransfer:
    return "QueueTransfer";
  }
  llvm_unreachable("unknown path kind");
}

void printPath(raw_ostream &os, const PathEdge &edge) {
  os << stringifyMemoryPlace(edge.srcPlace) << " -> "
     << stringifyMemoryPlace(edge.dstPlace);
}

void printRateCandidates(raw_ostream &os, ArrayRef<StringRef> sections,
                         ArrayRef<StringRef> names) {
  bool first = true;
  for (StringRef section : sections) {
    for (StringRef name : names) {
      if (!first)
        os << ", ";
      first = false;
      os << section << "." << name;
    }
  }
}

} // namespace

FailureOr<int64_t>
TargetCostModel::getMemoryRate(StringRef section, StringRef name) const {
  auto it = ratesByQualifiedName.find(makeRateKey(section, name));
  if (it == ratesByQualifiedName.end())
    return failure();
  return it->second;
}

FailureOr<int64_t>
TargetCostModel::getPreferredMemoryRate(StringRef name) const {
  auto it = preferredRatesByName.find(name);
  if (it == preferredRatesByName.end())
    return failure();
  return it->second;
}

FailureOr<PathCost>
TargetCostModel::getPathCost(const PathEdge &edge) const {
  auto it = pathCosts.find(edge);
  if (it == pathCosts.end())
    return failure();
  return it->second;
}

FailureOr<int64_t>
TargetCostModel::estimateTransferCycles(const PathEdge &edge,
                                        int64_t bytes) const {
  if (bytes < 0)
    return failure();
  FailureOr<PathCost> cost = getPathCost(edge);
  if (failed(cost) || cost->bytesPerCycle <= 0)
    return failure();
  int64_t payloadCycles =
      bytes == 0 ? 0 : (bytes + cost->bytesPerCycle - 1) / cost->bytesPerCycle;
  return cost->startupCycles + payloadCycles;
}

FailureOr<TargetCostModel>
TargetCostModelBuilder::build(const TargetProfile &profile,
                              const TargetMemoryModel &memoryModel,
                              raw_ostream &os) const {
  TargetCostModel model;
  llvm::StringMap<int> preferredPriorities;

  SmallVector<const TargetMemoryRateInfo *> rates;
  rates.reserve(profile.memoryRates.size());
  for (const TargetMemoryRateInfo &rate : profile.memoryRates)
    rates.push_back(&rate);
  llvm::sort(rates, [](const TargetMemoryRateInfo *lhs,
                       const TargetMemoryRateInfo *rhs) {
    if (sectionPriority(lhs->section) != sectionPriority(rhs->section))
      return sectionPriority(lhs->section) < sectionPriority(rhs->section);
    return std::tie(lhs->section, lhs->name) <
           std::tie(rhs->section, rhs->name);
  });

  for (const TargetMemoryRateInfo *rate : rates) {
    if (rate->bytesPerCycle <= 0)
      continue;

    model.ratesByQualifiedName[makeRateKey(rate->section, rate->name)] =
        rate->bytesPerCycle;
    int priority = sectionPriority(rate->section);
    auto preferredIt = preferredPriorities.find(rate->name);
    if (preferredIt == preferredPriorities.end() ||
        priority < preferredIt->second) {
      preferredPriorities[rate->name] = priority;
      model.preferredRatesByName[rate->name] = rate->bytesPerCycle;
    }
  }

  for (MemoryPlace src : memoryModel.getMemoryPlaces()) {
    for (MemoryPlace dst : memoryModel.getMemoryPlaces()) {
      for (const PathEdge &edge : memoryModel.findDirectPaths(src, dst)) {
        FailureOr<PathKind> kind = memoryModel.getPathKind(edge);
        if (failed(kind))
          continue;

        SmallVector<StringRef> rateCandidates = getRateCandidates(edge, *kind);
        SmallVector<StringRef> sectionCandidates = getSectionCandidates(edge);
        std::optional<StringRef> selectedRateName;
        std::optional<StringRef> selectedRateSection;
        int64_t selectedRate = 0;
        for (StringRef section : sectionCandidates) {
          for (StringRef candidate : rateCandidates) {
            FailureOr<int64_t> rate = model.getMemoryRate(section, candidate);
            if (succeeded(rate)) {
              selectedRateSection = section;
              selectedRateName = candidate;
              selectedRate = *rate;
              break;
            }
          }
          if (selectedRateName)
            break;
        }

        if (!selectedRateName) {
          os << "TargetCostModel verification failed: missing memory rate from "
             << "candidates [";
          printRateCandidates(os, sectionCandidates, rateCandidates);
          os << "]";
          os << " for path ";
          printPath(os, edge);
          os << " kind " << stringifyPathKind(*kind) << "\n";
          return failure();
        }

        model.pathCosts[edge] =
            PathCost{startupCyclesFor(*kind), selectedRate,
                     overlapsCompute(*kind), selectedRateSection->str(),
                     selectedRateName->str()};
      }
    }
  }

  return model;
}

} // namespace mlir::ascend
