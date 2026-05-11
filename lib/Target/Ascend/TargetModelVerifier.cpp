//===- TargetModelVerifier.cpp - Ascend target model verifier -------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/Ascend/TargetModelVerifier.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::ascend {
namespace {

constexpr MemoryPlace requiredMemoryPlaces[] = {
    MemoryPlace::GM,     MemoryPlace::A1,     MemoryPlace::A2,
    MemoryPlace::B1,     MemoryPlace::B2,     MemoryPlace::CO1,
    MemoryPlace::VECIN,  MemoryPlace::VECOUT, MemoryPlace::VECCALC};

struct RequiredDirectPath {
  MemoryPlace src;
  MemoryPlace dst;
  PathKind kind;
};

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

void printPath(raw_ostream &os, MemoryPlace src, MemoryPlace dst) {
  os << stringifyMemoryPlace(src) << " -> " << stringifyMemoryPlace(dst);
}

void printPath(raw_ostream &os, const PathEdge &edge) {
  printPath(os, edge.srcPlace, edge.dstPlace);
}

void emitFailure(raw_ostream &os, StringRef code) {
  os << "TargetModelVerifier failed: " << code;
}

bool requiresIntrinsic(PathKind kind) {
  return kind != PathKind::QueueTransfer;
}

bool hasIntrinsicForPathKind(const TargetIntrinsicModel &intrinsicModel,
                             PathKind kind) {
  return !intrinsicModel.getIntrinsicsForPathKind(kind).empty();
}

LogicalResult verifyRequiredPlaces(const TargetMemoryModel &memoryModel,
                                   raw_ostream &os) {
  for (MemoryPlace place : requiredMemoryPlaces) {
    if (!memoryModel.supportsMemoryPlace(place)) {
      emitFailure(os, "TargetProfileMissingField");
      os << ": missing memory place " << stringifyMemoryPlace(place) << "\n";
      return failure();
    }

    FailureOr<CapacityRule> capacity = memoryModel.getCapacity(place);
    if (failed(capacity) || capacity->staticCapacityBytes <= 0) {
      emitFailure(os, "TargetProfileMissingField");
      os << ": missing positive capacity for memory place "
         << stringifyMemoryPlace(place) << "\n";
      return failure();
    }
  }
  return success();
}

bool hasDirectPath(const TargetMemoryModel &memoryModel,
                   const RequiredDirectPath &required) {
  for (const PathEdge &edge :
       memoryModel.findDirectPaths(required.src, required.dst)) {
    FailureOr<PathKind> kind = memoryModel.getPathKind(edge);
    if (succeeded(kind) && *kind == required.kind)
      return true;
  }
  return false;
}

LogicalResult verifyRequiredPath(const TargetMemoryModel &memoryModel,
                                 const RequiredDirectPath &required,
                                 raw_ostream &os) {
  if (hasDirectPath(memoryModel, required))
    return success();

  emitFailure(os, "TargetPathMissing");
  os << ": missing required path ";
  printPath(os, required.src, required.dst);
  os << " kind " << stringifyPathKind(required.kind) << "\n";
  return failure();
}

LogicalResult verifyRequiredPaths(const TargetProfile &profile,
                                  const TargetMemoryModel &memoryModel,
                                  raw_ostream &os) {
  constexpr RequiredDirectPath requiredPaths[] = {
      {MemoryPlace::GM, MemoryPlace::A1, PathKind::Load2D},
      {MemoryPlace::GM, MemoryPlace::B1, PathKind::Load2D},
      {MemoryPlace::A1, MemoryPlace::A2, PathKind::DirectCopy},
      {MemoryPlace::B1, MemoryPlace::B2, PathKind::DirectCopy},
      {MemoryPlace::CO1, MemoryPlace::VECIN, PathKind::QueueTransfer},
      {MemoryPlace::GM, MemoryPlace::VECIN, PathKind::DirectCopy},
      {MemoryPlace::VECOUT, MemoryPlace::GM, PathKind::DirectCopy}};

  for (const RequiredDirectPath &required : requiredPaths)
    if (failed(verifyRequiredPath(memoryModel, required, os)))
      return failure();

  if (profile.hardware.supportFixpipe) {
    RequiredDirectPath fixpipe{MemoryPlace::CO1, MemoryPlace::GM,
                               PathKind::FixPipe};
    if (failed(verifyRequiredPath(memoryModel, fixpipe, os)))
      return failure();
  }

  return success();
}

LogicalResult
verifyFixpipeConsistency(const TargetProfile &profile,
                         const TargetIntrinsicModel &intrinsicModel,
                         const TargetMemoryModel &memoryModel,
                         raw_ostream &os) {
  bool hasFixpipeIntrinsic =
      hasIntrinsicForPathKind(intrinsicModel, PathKind::FixPipe);
  bool hasFixpipePath =
      hasDirectPath(memoryModel,
                    {MemoryPlace::CO1, MemoryPlace::GM, PathKind::FixPipe});

  if (profile.hardware.supportFixpipe && !hasFixpipeIntrinsic) {
    emitFailure(os, "TargetPathIntrinsicMissing");
    os << ": support_fixpipe=true but no FixPipe intrinsic is available\n";
    return failure();
  }
  if (profile.hardware.supportFixpipe && !hasFixpipePath) {
    emitFailure(os, "TargetPathMissing");
    os << ": support_fixpipe=true but no CO1 -> GM FixPipe path is available\n";
    return failure();
  }
  if (!profile.hardware.supportFixpipe && hasFixpipeIntrinsic) {
    emitFailure(os, "TargetProfileMissingField");
    os << ": support_fixpipe=false but FixPipe intrinsic is available\n";
    return failure();
  }

  return success();
}

LogicalResult verifyPathClosure(const TargetMemoryModel &memoryModel,
                                const TargetIntrinsicModel &intrinsicModel,
                                const TargetCostModel &costModel,
                                raw_ostream &os) {
  for (MemoryPlace src : memoryModel.getMemoryPlaces()) {
    for (MemoryPlace dst : memoryModel.getMemoryPlaces()) {
      for (const PathEdge &edge : memoryModel.findDirectPaths(src, dst)) {
        if (!memoryModel.supportsMemoryPlace(edge.srcPlace) ||
            !memoryModel.supportsMemoryPlace(edge.dstPlace)) {
          emitFailure(os, "TargetProfileMissingField");
          os << ": path references unsupported memory place ";
          printPath(os, edge);
          os << "\n";
          return failure();
        }

        FailureOr<PathKind> kind = memoryModel.getPathKind(edge);
        if (failed(kind)) {
          emitFailure(os, "TargetPathMissing");
          os << ": missing path kind for ";
          printPath(os, edge);
          os << "\n";
          return failure();
        }

        if (requiresIntrinsic(*kind) &&
            !hasIntrinsicForPathKind(intrinsicModel, *kind)) {
          emitFailure(os, "TargetPathIntrinsicMissing");
          os << ": missing " << stringifyPathKind(*kind)
             << " intrinsic for path ";
          printPath(os, edge);
          os << "\n";
          return failure();
        }

        FailureOr<PathCost> cost = costModel.getPathCost(edge);
        if (failed(cost)) {
          emitFailure(os, "TargetPathCostMissing");
          os << ": missing path cost for ";
          printPath(os, edge);
          os << " kind " << stringifyPathKind(*kind) << "\n";
          return failure();
        }
      }
    }
  }
  return success();
}

} // namespace

LogicalResult
TargetModelVerifier::verify(const TargetProfile &profile,
                            const TargetMemoryModel &memoryModel,
                            const TargetIntrinsicModel &intrinsicModel,
                            const TargetCostModel &costModel,
                            raw_ostream &os) const {
  if (failed(verifyTargetProfile(profile, os)))
    return failure();
  if (failed(verifyRequiredPlaces(memoryModel, os)))
    return failure();
  if (failed(verifyRequiredPaths(profile, memoryModel, os)))
    return failure();
  if (failed(verifyFixpipeConsistency(profile, intrinsicModel, memoryModel, os)))
    return failure();
  if (failed(verifyPathClosure(memoryModel, intrinsicModel, costModel, os)))
    return failure();
  return success();
}

} // namespace mlir::ascend
