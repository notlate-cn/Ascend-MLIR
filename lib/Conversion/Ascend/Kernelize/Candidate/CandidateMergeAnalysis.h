//===- CandidateMergeAnalysis.h - Ascend candidate merge ----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_CANDIDATEMERGEANALYSIS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_CANDIDATEMERGEANALYSIS_H

#include "Conversion/Ascend/Kernelize/Candidate/CandidateClosure.h"
#include "Conversion/Ascend/Kernelize/Analysis/DependencyAnalysis.h"
#include "Conversion/Ascend/Kernelize/Candidate/FusionCandidateAnalysis.h"
#include "Conversion/Ascend/Kernelize/KernelizeTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace mlir::afir::ascend::kernelize {

struct MergedCandidate {
  unsigned mergedCandidateId = 0;
  SmallVector<unsigned> sourceCandidateIds;
  SmallVector<Operation *> primaryOps;
  SmallVector<Operation *> internalOps;
  SmallVector<KernelizePrimitiveKind> primitiveCombo;
  CandidateClosure closure;
  ScheduleContract scheduleContract;
  std::string familyResolverName = "kernelize_trait_resolver";
  int64_t benefitScore = 0;
  bool legal = false;
  std::string rejectionReason;
};

} // namespace mlir::afir::ascend::kernelize

namespace llvm {
template <>
struct CalculateSmallVectorDefaultInlinedElements<
    mlir::afir::ascend::kernelize::MergedCandidate> {
  static constexpr size_t value = 0;
};
} // namespace llvm

namespace mlir::afir::ascend::kernelize {

class CandidateMergeAnalyzer {
public:
  SmallVector<MergedCandidate> analyze(ArrayRef<FusionCandidate> candidates,
                                       const DependencyAnalysisResult &deps,
                                       const KernelizeConfig &config) const;
};

void emitCandidateMergeReport(raw_ostream &os,
                              ArrayRef<MergedCandidate> merged,
                              const ProducerConsumerIndex &index);

} // namespace mlir::afir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_CANDIDATEMERGEANALYSIS_H
