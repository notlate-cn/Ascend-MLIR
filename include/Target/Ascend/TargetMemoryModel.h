//===- TargetMemoryModel.h - Ascend target memory model --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_TARGET_ASCEND_TARGET_MEMORY_MODEL_H
#define ASCEND_MLIR_TARGET_ASCEND_TARGET_MEMORY_MODEL_H

#include "Target/Ascend/TargetProfile.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>

namespace mlir::ascend {

enum class PathKind { DirectCopy, Load2D, Load2DTranspose, FixPipe,
                      QueueTransfer };

struct PathEdge {
  MemoryPlace srcPlace;
  MemoryPlace dstPlace;
  unsigned pathVariant = 0;

  bool operator==(const PathEdge &other) const {
    return srcPlace == other.srcPlace && dstPlace == other.dstPlace &&
           pathVariant == other.pathVariant;
  }
};

struct PathConstraint {
  SmallVector<std::string> dtypes;
  int64_t minRank = 1;
  int64_t maxRank = 0;
  bool requires2DLoad = false;
  bool allowsTranspose = false;
};

struct PathRoute {
  SmallVector<PathEdge> edges;
};

struct CapacityRule {
  int64_t staticCapacityBytes = 0;
  int64_t availableCapacityBytes = 0;
  bool partitionedByUnit = false;
};

struct AlignmentRule {
  int64_t addressAlignmentBytes = 32;
  int64_t strideAlignmentBytes = 32;
  int64_t tileAlignmentElements = 1;
  bool requiresPowerOfTwo = false;
};

struct VisibilityRule {
  bool dma = false;
  bool cube = false;
  bool vector = false;
  bool abiVisible = false;
};

class TargetMemoryModel {
public:
  ArrayRef<MemoryPlace> getMemoryPlaces() const;
  bool supportsMemoryPlace(MemoryPlace place) const;
  FailureOr<CapacityRule> getCapacity(MemoryPlace place) const;
  FailureOr<AlignmentRule> getAlignment(MemoryPlace place) const;
  bool isPlaceVisibleTo(MemoryPlace place, ExecutionUnit unit) const;

  SmallVector<PathEdge> findDirectPaths(MemoryPlace src,
                                        MemoryPlace dst) const;
  SmallVector<PathRoute> findPaths(MemoryPlace src, MemoryPlace dst) const;
  FailureOr<PathKind> getPathKind(const PathEdge &edge) const;
  FailureOr<SmallVector<PathConstraint>>
  getPathConstraints(const PathEdge &edge) const;

private:
  friend class TargetMemoryModelBuilder;
  SmallVector<MemoryPlace> memoryPlaces;
  DenseMap<MemoryPlace, CapacityRule> capacity;
  DenseMap<MemoryPlace, AlignmentRule> alignment;
  DenseMap<MemoryPlace, VisibilityRule> visibilityRules;
  DenseMap<MemoryPlace, SmallVector<PathEdge>> pathGraph;
  DenseMap<PathEdge, PathKind> pathKinds;
  DenseMap<PathEdge, SmallVector<PathConstraint>> pathConstraints;
};

class TargetMemoryModelBuilder {
public:
  FailureOr<TargetMemoryModel> build(const TargetProfile &profile,
                                     raw_ostream &os) const;
};

} // namespace mlir::ascend

namespace llvm {

template <> struct DenseMapInfo<mlir::ascend::PathEdge> {
  static mlir::ascend::PathEdge getEmptyKey() {
    return {static_cast<mlir::ascend::MemoryPlace>(-1),
            static_cast<mlir::ascend::MemoryPlace>(-1), ~0u};
  }

  static mlir::ascend::PathEdge getTombstoneKey() {
    return {static_cast<mlir::ascend::MemoryPlace>(-2),
            static_cast<mlir::ascend::MemoryPlace>(-2), ~0u - 1};
  }

  static unsigned getHashValue(const mlir::ascend::PathEdge &edge) {
    return static_cast<unsigned>(
        hash_combine(static_cast<int>(edge.srcPlace),
                     static_cast<int>(edge.dstPlace), edge.pathVariant));
  }

  static bool isEqual(const mlir::ascend::PathEdge &lhs,
                      const mlir::ascend::PathEdge &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

#endif // ASCEND_MLIR_TARGET_ASCEND_TARGET_MEMORY_MODEL_H
