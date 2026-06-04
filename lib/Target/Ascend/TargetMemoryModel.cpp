//===- TargetMemoryModel.cpp - Ascend target memory model -----------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/Ascend/TargetMemoryModel.h"

#include "llvm/ADT/STLExtras.h"
#include <algorithm>

using namespace mlir;

namespace mlir::ascend {
namespace {

constexpr MemoryPlace requiredMemoryPlaces[] = {
    MemoryPlace::GM,     MemoryPlace::A1,     MemoryPlace::A2,
    MemoryPlace::B1,     MemoryPlace::B2,     MemoryPlace::CO1,
    MemoryPlace::VECIN,  MemoryPlace::VECOUT, MemoryPlace::VECCALC};

bool isLoad2DIntrinsic(StringRef name) {
  return name == "Intrinsic_data_move_out2l1" ||
         name == "Intrinsic_data_move_out2l0a" ||
         name == "Intrinsic_data_move_out2l0b";
}

bool isLoad2DTransposeIntrinsic(StringRef name) {
  return name.contains("_transpose_");
}

bool isFixPipePathIntrinsic(StringRef name) {
  return name.starts_with("Intrinsic_fix_pipe_l");
}

bool isDirectCopyIntrinsic(StringRef name) {
  return name.starts_with("Intrinsic_data_move_") &&
         !isLoad2DIntrinsic(name) && !isLoad2DTransposeIntrinsic(name);
}

bool matchesPathKind(StringRef intrinsicName, PathKind kind) {
  switch (kind) {
  case PathKind::DirectCopy:
    return isDirectCopyIntrinsic(intrinsicName);
  case PathKind::Load2D:
    return isLoad2DIntrinsic(intrinsicName);
  case PathKind::Load2DTranspose:
    return isLoad2DTransposeIntrinsic(intrinsicName);
  case PathKind::FixPipe:
    return isFixPipePathIntrinsic(intrinsicName);
  case PathKind::QueueTransfer:
    return false;
  }
  llvm_unreachable("unknown path kind");
}

void appendUnique(SmallVectorImpl<std::string> &values, StringRef value) {
  if (llvm::none_of(values, [&](StringRef existing) {
        return existing == value;
      }))
    values.push_back(value.str());
}

SmallVector<std::string> collectDTypesForKind(const TargetProfile &profile,
                                              PathKind kind) {
  SmallVector<std::string> dtypes;
  if (kind == PathKind::QueueTransfer) {
    dtypes.push_back("*");
    return dtypes;
  }

  for (const TargetIntrinsicInfo &intrinsic : profile.intrinsics) {
    if (!matchesPathKind(intrinsic.name, kind))
      continue;
    for (StringRef dtype : intrinsic.dtypes)
      appendUnique(dtypes, dtype);
  }

  if (dtypes.empty())
    dtypes.push_back("*");
  llvm::sort(dtypes);
  dtypes.erase(std::unique(dtypes.begin(), dtypes.end()), dtypes.end());
  return dtypes;
}

SmallVector<PathConstraint> buildPathConstraints(const TargetProfile &profile,
                                                 PathKind kind) {
  PathConstraint constraint;
  constraint.dtypes = collectDTypesForKind(profile, kind);

  switch (kind) {
  case PathKind::DirectCopy:
  case PathKind::FixPipe:
  case PathKind::QueueTransfer:
    constraint.minRank = 1;
    break;
  case PathKind::Load2D:
    constraint.minRank = 2;
    constraint.requires2DLoad = true;
    break;
  case PathKind::Load2DTranspose:
    constraint.minRank = 2;
    constraint.requires2DLoad = true;
    constraint.allowsTranspose = true;
    break;
  }

  return {constraint};
}

bool hasIntrinsicForKind(const TargetProfile &profile, PathKind kind) {
  for (const TargetIntrinsicInfo &intrinsic : profile.intrinsics)
    if (matchesPathKind(intrinsic.name, kind))
      return true;
  return false;
}

bool containsPlace(ArrayRef<MemoryPlace> places, MemoryPlace place) {
  return llvm::is_contained(places, place);
}

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

SmallVector<PathRoute>
TargetMemoryModel::findPaths(MemoryPlace src, MemoryPlace dst) const {
  SmallVector<PathRoute> routes;
  SmallVector<PathEdge> activeEdges;
  SmallVector<MemoryPlace> activePlaces;

  auto search = [&](auto &&self, MemoryPlace current) -> void {
    if (current == dst) {
      routes.push_back(PathRoute{activeEdges});
      return;
    }
    if (activeEdges.size() >= memoryPlaces.size())
      return;

    auto it = pathGraph.find(current);
    if (it == pathGraph.end())
      return;

    for (const PathEdge &edge : it->second) {
      if (containsPlace(activePlaces, edge.dstPlace))
        continue;
      activeEdges.push_back(edge);
      activePlaces.push_back(edge.dstPlace);
      self(self, edge.dstPlace);
      activePlaces.pop_back();
      activeEdges.pop_back();
    }
  };

  activePlaces.push_back(src);
  search(search, src);
  return routes;
}

FailureOr<PathKind> TargetMemoryModel::getPathKind(const PathEdge &edge) const {
  auto it = pathKinds.find(edge);
  if (it == pathKinds.end())
    return failure();
  return it->second;
}

FailureOr<SmallVector<PathConstraint>>
TargetMemoryModel::getPathConstraints(const PathEdge &edge) const {
  auto it = pathConstraints.find(edge);
  if (it == pathConstraints.end())
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
    unsigned pathVariant = 0;
    for (const PathEdge &existing : model.pathGraph[src])
      if (existing.dstPlace == dst)
        ++pathVariant;
    PathEdge edge{src, dst, pathVariant};
    model.pathGraph[src].push_back(edge);
    model.pathKinds[edge] = kind;
    model.pathConstraints[edge] = buildPathConstraints(profile, kind);
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
  if (hasIntrinsicForKind(profile, PathKind::Load2DTranspose))
    addDirectPath(MemoryPlace::GM, MemoryPlace::B1,
                  PathKind::Load2DTranspose);
  addDirectPath(MemoryPlace::A1, MemoryPlace::A2, PathKind::DirectCopy);
  addDirectPath(MemoryPlace::B1, MemoryPlace::B2, PathKind::DirectCopy);
  addDirectPath(MemoryPlace::CO1, MemoryPlace::VECIN,
                PathKind::QueueTransfer);
  addDirectPath(MemoryPlace::GM, MemoryPlace::VECIN, PathKind::DirectCopy);
  addDirectPath(MemoryPlace::VECOUT, MemoryPlace::GM, PathKind::DirectCopy);
  if (profile.hardware.supportFixpipe)
    addDirectPath(MemoryPlace::CO1, MemoryPlace::GM, PathKind::FixPipe);

  // Intrinsic-backed closure is verified after the intrinsic and cost models
  // are built, because optional path variants depend on the complete profile.
  return model;
}

} // namespace mlir::ascend
