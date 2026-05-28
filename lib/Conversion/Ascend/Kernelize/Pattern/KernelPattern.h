//===- KernelPattern.h - Ascend kernel pattern model -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELPATTERN_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELPATTERN_H

#include "Conversion/Ascend/Kernelize/Candidate/CandidateClosure.h"
#include "Conversion/Ascend/Kernelize/Candidate/CandidateMergeAnalysis.h"
#include "Conversion/Ascend/Kernelize/Analysis/DependencyAnalysis.h"
#include "Conversion/Ascend/Kernelize/Candidate/FusionCandidateAnalysis.h"
#include "Conversion/Ascend/Kernelize/Candidate/HorizontalFusionAnalysis.h"
#include "Conversion/Ascend/Kernelize/KernelizeTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace mlir::ascend::kernelize {

struct KernelPatternCandidate {
  unsigned candidateId = 0;
  CandidateKind sourceKind = CandidateKind::Fusion;
  SmallVector<Operation *> internalOps;
  SmallVector<Operation *> primaryOps;
  CandidateClosure closure;
  ScheduleContract scheduleContract;
  int64_t benefitScore = 0;
};

struct KernelPatternEdge {
  unsigned from = 0;
  unsigned to = 0;
  KernelPatternEdgeKind kind = KernelPatternEdgeKind::DataDependency;
  Value carriedValue;
};

struct KernelPatternEdgeKey {
  unsigned from = 0;
  unsigned to = 0;
  KernelPatternEdgeKind kind = KernelPatternEdgeKind::DataDependency;

  bool operator==(const KernelPatternEdgeKey &other) const {
    return from == other.from && to == other.to && kind == other.kind;
  }
};

struct KernelPattern {
  unsigned patternId = 0;
  std::string kernelName;
  SmallVector<Operation *> internalOps;
  SmallVector<Operation *> primaryOps;
  ScheduleContract scheduleContract;
};

} // namespace mlir::ascend::kernelize

namespace llvm {
template <>
struct CalculateSmallVectorDefaultInlinedElements<
    mlir::ascend::kernelize::KernelPatternCandidate> {
  static constexpr size_t value = 0;
};

template <>
struct CalculateSmallVectorDefaultInlinedElements<
    mlir::ascend::kernelize::KernelPattern> {
  static constexpr size_t value = 0;
};

template <>
struct DenseMapInfo<mlir::ascend::kernelize::KernelPatternEdgeKey> {
  using Key = mlir::ascend::kernelize::KernelPatternEdgeKey;

  // Candidate ids are dense zero-based indices; reserve the max sentinels for
  // DenseMap bookkeeping.
  static inline Key getEmptyKey() {
    return {std::numeric_limits<unsigned>::max(),
            std::numeric_limits<unsigned>::max(),
            mlir::ascend::kernelize::KernelPatternEdgeKind::
                DataDependency};
  }

  static inline Key getTombstoneKey() {
    return {std::numeric_limits<unsigned>::max() - 1,
            std::numeric_limits<unsigned>::max(),
            mlir::ascend::kernelize::KernelPatternEdgeKind::
                DataDependency};
  }

  static unsigned getHashValue(const Key &key) {
    return static_cast<unsigned>(
        llvm::hash_combine(key.from, key.to, static_cast<unsigned>(key.kind)));
  }

  static bool isEqual(const Key &lhs, const Key &rhs) { return lhs == rhs; }
};
} // namespace llvm

namespace mlir::ascend::kernelize {

struct KernelPatternGraph {
  SmallVector<KernelPatternCandidate> nodes;
  DenseMap<Operation *, SmallVector<unsigned>> coveringMap;
  SmallVector<KernelPatternEdge> edges;
};

class KernelPatternBuilder {
public:
  KernelPatternGraph
  build(ArrayRef<FusionCandidate> fusionCandidates,
        ArrayRef<MergedCandidate> mergedCandidates,
        ArrayRef<HorizontalFusionCandidate> horizontalCandidates,
        const DependencyAnalysisResult &deps) const;
};

class KernelPartitioner {
public:
  SmallVector<KernelPattern> partition(const KernelPatternGraph &graph,
                                       const DependencyAnalysisResult &deps)
      const;
};

void attachKernelPatternAttributes(ModuleOp module,
                                   ArrayRef<KernelPattern> patterns);
void emitKernelPatternGraphReport(raw_ostream &os,
                                  const KernelPatternGraph &graph,
                                  const ProducerConsumerIndex &index);
void emitKernelPartitionReport(raw_ostream &os,
                               ArrayRef<KernelPattern> patterns,
                               const ProducerConsumerIndex &index);

} // namespace mlir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELPATTERN_H
