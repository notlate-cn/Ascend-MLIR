//===- ScheduleContractDriver.h - Ascend schedule contract -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULECONTRACTDRIVER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULECONTRACTDRIVER_H

#include "ScheduleTypes.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir::ascend::schedule {

struct ScheduleContractReport {
  std::string kernelId;
  std::string contract;
  unsigned verifiedOps = 0;
};

LogicalResult
applyScheduleContractMarkers(const KernelPatternView &pattern,
                             const ScheduleProblem &scheduleProblem,
                             const ScheduleDecisionSet &decisionSet);

LogicalResult applyScheduleContractMarkers(
    const KernelPatternView &pattern, const ScheduleProblem &scheduleProblem,
    const ScheduleDecisionSet &decisionSet, ScheduleContractReport &report);

void printScheduleContractReport(const ScheduleContractReport &report,
                                 llvm::raw_ostream &os);

} // namespace mlir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULECONTRACTDRIVER_H
