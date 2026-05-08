//===- FusionCandidateAnalysis.h - Ascend V2 fusion candidates -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_FUSIONCANDIDATEANALYSIS_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_FUSIONCANDIDATEANALYSIS_H

#include "Conversion/AscendV2/Kernelize/CandidateClosure.h"
#include "Conversion/AscendV2/Kernelize/KernelizeTypes.h"
#include "Conversion/AscendV2/Kernelize/OpRoleClassification.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace mlir::afir::ascend::v2::kernelize {

struct ScheduleContract {
  SmallVector<StringRef, 0> templateFamilies;
};

struct FusionCandidate {
  unsigned candidateId = 0;
  CandidateKind kind = CandidateKind::Fusion;
  std::string primitive;
  SmallVector<Operation *, 0> primaryOps;
  SmallVector<Operation *, 0> internalOps;
  CandidateClosure closure;
  ScheduleContract scheduleContract;
  int64_t benefitScore = 0;
  bool legal = false;
  std::string rejectionReason;
};

} // namespace mlir::afir::ascend::v2::kernelize

namespace llvm {
template <>
struct CalculateSmallVectorDefaultInlinedElements<
    mlir::afir::ascend::v2::kernelize::FusionCandidate> {
  static constexpr size_t value = 0;
};
} // namespace llvm

namespace mlir::afir::ascend::v2::kernelize {

class FusionCandidateAnalyzer {
public:
  SmallVector<FusionCandidate> analyze(const DependencyAnalysisResult &deps,
                                       const OpRoleMap &roleMap,
                                       const KernelizeConfig &config) const;
};

void emitFusionCandidateReport(raw_ostream &os,
                               ArrayRef<FusionCandidate> candidates,
                               const ProducerConsumerIndex &index);

} // namespace mlir::afir::ascend::v2::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_FUSIONCANDIDATEANALYSIS_H
