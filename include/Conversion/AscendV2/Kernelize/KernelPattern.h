//===- KernelPattern.h - Ascend V2 kernel pattern model -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_KERNELPATTERN_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_KERNELPATTERN_H

#include "Conversion/AscendV2/Kernelize/CandidateClosure.h"
#include "Conversion/AscendV2/Kernelize/CandidateMergeAnalysis.h"
#include "Conversion/AscendV2/Kernelize/DependencyAnalysis.h"
#include "Conversion/AscendV2/Kernelize/FusionCandidateAnalysis.h"
#include "Conversion/AscendV2/Kernelize/HorizontalFusionAnalysis.h"
#include "Conversion/AscendV2/Kernelize/KernelizeTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace mlir::afir::ascend::v2::kernelize {

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

struct KernelPattern {
  unsigned patternId = 0;
  std::string kernelName;
  SmallVector<Operation *> internalOps;
  SmallVector<Operation *> primaryOps;
  ScheduleContract scheduleContract;
};

} // namespace mlir::afir::ascend::v2::kernelize

namespace llvm {
template <>
struct CalculateSmallVectorDefaultInlinedElements<
    mlir::afir::ascend::v2::kernelize::KernelPatternCandidate> {
  static constexpr size_t value = 0;
};

template <>
struct CalculateSmallVectorDefaultInlinedElements<
    mlir::afir::ascend::v2::kernelize::KernelPattern> {
  static constexpr size_t value = 0;
};
} // namespace llvm

namespace mlir::afir::ascend::v2::kernelize {

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

} // namespace mlir::afir::ascend::v2::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_KERNELPATTERN_H
