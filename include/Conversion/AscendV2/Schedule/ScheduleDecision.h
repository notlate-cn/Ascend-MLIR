//===- ScheduleDecision.h - Ascend V2 schedule decisions ------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_SCHEDULEDECISION_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_SCHEDULEDECISION_H

#include "Conversion/AscendV2/Schedule/ScheduleTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir::afir::ascend::v2::schedule {

ScheduleDecisionSet buildScheduleDecisionSet(
    llvm::StringRef kernelId, llvm::ArrayRef<ScheduleInstance> instances);

void printScheduleDecisionSetReport(const ScheduleDecisionSet &decisionSet,
                                    llvm::raw_ostream &os);

} // namespace mlir::afir::ascend::v2::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_SCHEDULEDECISION_H
