//===- ScheduleDecision.h - Ascend schedule decisions ------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULEDECISION_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULEDECISION_H

#include "ScheduleTypes.h"

#include "llvm/ADT/ArrayRef.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir::ascend::schedule {

ScheduleDecisionSet buildScheduleDecisionSet(
    const ScheduleProblem &problem, llvm::ArrayRef<ScheduleInstance> instances,
    const ScheduleSearchOptions &options = ScheduleSearchOptions());

void printScheduleDecisionSetReport(const ScheduleDecisionSet &decisionSet,
                                    llvm::raw_ostream &os);

} // namespace mlir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULEDECISION_H
