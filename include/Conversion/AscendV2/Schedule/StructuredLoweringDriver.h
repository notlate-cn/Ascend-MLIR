//===- StructuredLoweringDriver.h - Ascend V2 structured lowering -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_STRUCTUREDLOWERINGDRIVER_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_STRUCTUREDLOWERINGDRIVER_H

#include "Conversion/AscendV2/Schedule/ScheduleTypes.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir::afir::ascend::v2::schedule {

struct StructuredLoweringReport {
  std::string kernelId;
  std::string skeleton;
  unsigned verifiedOps = 0;
};

LogicalResult
applyStructuredLoweringMarkers(const KernelPatternView &pattern,
                               const ScheduleProblem &scheduleProblem,
                               const ScheduleDecisionSet &decisionSet);

LogicalResult applyStructuredLoweringMarkers(
    const KernelPatternView &pattern, const ScheduleProblem &scheduleProblem,
    const ScheduleDecisionSet &decisionSet, StructuredLoweringReport &report);

void printStructuredLoweringReport(const StructuredLoweringReport &report,
                                   llvm::raw_ostream &os);

} // namespace mlir::afir::ascend::v2::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_STRUCTUREDLOWERINGDRIVER_H
