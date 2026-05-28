//===- FusionCandidateAnalysis.h - Ascend fusion candidates -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_FUSIONCANDIDATEANALYSIS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_FUSIONCANDIDATEANALYSIS_H

#include "Conversion/Ascend/Kernelize/Candidate/CandidateClosure.h"
#include "Conversion/Ascend/Kernelize/KernelizeTypes.h"
#include "Conversion/Ascend/Kernelize/Analysis/OpRoleClassification.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace mlir::ascend::kernelize {

struct ScheduleContract {
  SmallVector<std::string, 2> templateFamilies;
  std::string handwrittenKind;
};

struct FusionCandidate {
  unsigned candidateId = 0;
  CandidateKind kind = CandidateKind::Fusion;
  KernelizePrimitiveKind primitive = KernelizePrimitiveKind::Unknown;
  SmallVector<Operation *, 0> primaryOps;
  SmallVector<Operation *, 0> internalOps;
  CandidateClosure closure;
  ScheduleContract scheduleContract;
  int64_t benefitScore = 0;
  bool legal = false;
  std::string rejectionReason;
};

} // namespace mlir::ascend::kernelize

namespace llvm {
template <>
struct CalculateSmallVectorDefaultInlinedElements<
    mlir::ascend::kernelize::FusionCandidate> {
  static constexpr size_t value = 0;
};
} // namespace llvm

namespace mlir::ascend::kernelize {

class FusionCandidateAnalyzer {
public:
  SmallVector<FusionCandidate> analyze(const DependencyAnalysisResult &deps,
                                       const OpRoleMap &roleMap,
                                       const KernelizeConfig &config) const;
};

void emitFusionCandidateReport(raw_ostream &os,
                               ArrayRef<FusionCandidate> candidates,
                               const ProducerConsumerIndex &index);

} // namespace mlir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_FUSIONCANDIDATEANALYSIS_H
