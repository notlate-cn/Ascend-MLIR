//===- TargetMemoryModel.cpp - Ascend target memory model -----------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/Ascend/TargetMemoryModel.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::ascend {
namespace {

constexpr MemoryPlace requiredMemoryPlaces[] = {
    MemoryPlace::GM,     MemoryPlace::A1,     MemoryPlace::A2,
    MemoryPlace::B1,     MemoryPlace::B2,     MemoryPlace::CO1,
    MemoryPlace::VECIN,  MemoryPlace::VECOUT, MemoryPlace::VECCALC};

} // namespace

ArrayRef<MemoryPlace> TargetMemoryModel::getMemoryPlaces() const {
  return memoryPlaces;
}

bool TargetMemoryModel::supportsMemoryPlace(MemoryPlace place) const {
  return llvm::is_contained(memoryPlaces, place);
}

FailureOr<CapacityRule>
TargetMemoryModel::getCapacity(MemoryPlace place) const {
  auto it = capacity.find(place);
  if (it == capacity.end())
    return failure();
  return it->second;
}

FailureOr<AlignmentRule>
TargetMemoryModel::getAlignment(MemoryPlace place) const {
  auto it = alignment.find(place);
  if (it == alignment.end())
    return failure();
  return it->second;
}

bool TargetMemoryModel::isPlaceVisibleTo(MemoryPlace place,
                                         ExecutionUnit unit) const {
  auto it = visibilityRules.find(place);
  if (it == visibilityRules.end())
    return false;

  switch (unit) {
  case ExecutionUnit::DMA:
    return it->second.dma;
  case ExecutionUnit::Cube:
    return it->second.cube;
  case ExecutionUnit::Vector:
    return it->second.vector;
  }
  llvm_unreachable("unknown execution unit");
}

SmallVector<PathEdge>
TargetMemoryModel::findDirectPaths(MemoryPlace src, MemoryPlace dst) const {
  SmallVector<PathEdge> matches;
  auto it = pathGraph.find(src);
  if (it == pathGraph.end())
    return matches;

  for (const PathEdge &edge : it->second) {
    if (edge.dstPlace == dst)
      matches.push_back(edge);
  }
  return matches;
}

FailureOr<PathKind> TargetMemoryModel::getPathKind(const PathEdge &edge) const {
  auto it = pathKinds.find(edge);
  if (it == pathKinds.end())
    return failure();
  return it->second;
}

FailureOr<TargetMemoryModel>
TargetMemoryModelBuilder::build(const TargetProfile &profile,
                                raw_ostream &os) const {
  TargetMemoryModel model;

  auto addVisibility = [&](MemoryPlace place, VisibilityRule rule) {
    model.visibilityRules[place] = rule;
  };

  auto addDirectPath = [&](MemoryPlace src, MemoryPlace dst, PathKind kind) {
    PathEdge edge{src, dst, 0};
    model.pathGraph[src].push_back(edge);
    model.pathKinds[edge] = kind;
  };

  for (MemoryPlace place : requiredMemoryPlaces) {
    auto capacityIt = profile.capacityBytes.find(place);
    if (capacityIt == profile.capacityBytes.end() || capacityIt->second <= 0) {
      os << "TargetMemoryModel verification failed: missing required memory "
            "place "
         << stringifyMemoryPlace(place) << "\n";
      return failure();
    }

    model.memoryPlaces.push_back(place);
    model.capacity[place] = {capacityIt->second, capacityIt->second, false};
    model.alignment[place] = AlignmentRule{};
  }

  addVisibility(MemoryPlace::GM, {/*dma=*/true, /*cube=*/false,
                                  /*vector=*/false, /*abiVisible=*/true});
  for (MemoryPlace place : {MemoryPlace::A1, MemoryPlace::A2, MemoryPlace::B1,
                            MemoryPlace::B2, MemoryPlace::CO1}) {
    addVisibility(place, {/*dma=*/true, /*cube=*/true, /*vector=*/false,
                          /*abiVisible=*/false});
  }
  for (MemoryPlace place :
       {MemoryPlace::VECIN, MemoryPlace::VECOUT, MemoryPlace::VECCALC}) {
    addVisibility(place, {/*dma=*/true, /*cube=*/false, /*vector=*/true,
                          /*abiVisible=*/false});
  }

  addDirectPath(MemoryPlace::GM, MemoryPlace::A1, PathKind::Load2D);
  addDirectPath(MemoryPlace::GM, MemoryPlace::B1, PathKind::Load2D);
  addDirectPath(MemoryPlace::A1, MemoryPlace::A2, PathKind::DirectCopy);
  addDirectPath(MemoryPlace::B1, MemoryPlace::B2, PathKind::DirectCopy);
  addDirectPath(MemoryPlace::CO1, MemoryPlace::VECIN,
                PathKind::QueueTransfer);
  addDirectPath(MemoryPlace::GM, MemoryPlace::VECIN, PathKind::DirectCopy);
  addDirectPath(MemoryPlace::VECOUT, MemoryPlace::GM, PathKind::DirectCopy);
  if (profile.hardware.supportFixpipe)
    addDirectPath(MemoryPlace::CO1, MemoryPlace::GM, PathKind::FixPipe);

  // Intrinsic-backed path validation belongs to TargetIntrinsicModel and the
  // profile verifier. This MVP only validates logical places and direct edges.
  return model;
}

} // namespace mlir::ascend
