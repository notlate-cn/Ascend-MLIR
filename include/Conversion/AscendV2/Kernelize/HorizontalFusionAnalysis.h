//===- HorizontalFusionAnalysis.h - Ascend V2 horizontal fusion -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_HORIZONTALFUSIONANALYSIS_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_HORIZONTALFUSIONANALYSIS_H

#include "Conversion/AscendV2/Kernelize/CandidateMergeAnalysis.h"
#include "Conversion/AscendV2/Kernelize/DependencyAnalysis.h"
#include "Conversion/AscendV2/Kernelize/FusionCandidateAnalysis.h"
#include "Conversion/AscendV2/Kernelize/KernelizeTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

namespace mlir::afir::ascend::v2::kernelize {

struct HorizontalFusionCandidate {
  unsigned horizontalCandidateId = 0;
  SmallVector<unsigned> siblingCandidateIds;
  SmallVector<Value> sharedInputs;
  SmallVector<ScheduleContract> perGroupContracts;
  int64_t benefitScore = 0;
  bool legal = false;
  std::string rejectionReason;
};

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

} // namespace mlir::afir::ascend::v2::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_HORIZONTALFUSIONANALYSIS_H
