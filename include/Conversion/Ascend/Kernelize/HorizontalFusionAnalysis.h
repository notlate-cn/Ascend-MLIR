//===- HorizontalFusionAnalysis.h - Ascend horizontal fusion -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_HORIZONTALFUSIONANALYSIS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_HORIZONTALFUSIONANALYSIS_H

#include "Conversion/Ascend/Kernelize/CandidateMergeAnalysis.h"
#include "Conversion/Ascend/Kernelize/DependencyAnalysis.h"
#include "Conversion/Ascend/Kernelize/FusionCandidateAnalysis.h"
#include "Conversion/Ascend/Kernelize/KernelizeTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace mlir::afir::ascend::kernelize {

struct HorizontalFusionCandidate {
  unsigned horizontalCandidateId = 0;
  SmallVector<unsigned> siblingCandidateIds;
  SmallVector<Value> sharedInputs;
  SmallVector<ScheduleContract> perGroupContracts;
  int64_t benefitScore = 0;
  bool legal = false;
  std::string rejectionReason;
};

} // namespace mlir::afir::ascend::kernelize

namespace llvm {
template <>
struct CalculateSmallVectorDefaultInlinedElements<
    mlir::afir::ascend::kernelize::HorizontalFusionCandidate> {
  static constexpr size_t value = 0;
};
} // namespace llvm

namespace mlir::afir::ascend::kernelize {

class HorizontalFusionAnalyzer {
public:
  SmallVector<HorizontalFusionCandidate>
  analyze(ArrayRef<FusionCandidate> fusionCandidates,
          ArrayRef<MergedCandidate> mergedCandidates,
          const DependencyAnalysisResult &deps,
          const KernelizeConfig &config) const;
};

void emitHorizontalFusionReport(
    raw_ostream &os, ArrayRef<HorizontalFusionCandidate> horizontal);

} // namespace mlir::afir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_HORIZONTALFUSIONANALYSIS_H
