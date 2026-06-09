//===- ScheduleProblemBuilder.h - Ascend schedule problem ---*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULEPROBLEMBUILDER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULEPROBLEMBUILDER_H

#include "ScheduleTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::ascend::schedule {

class ScheduleSymbolAxisSpaceCache;

FailureOr<ScheduleProblem>
buildScheduleProblem(const KernelPatternView &pattern,
                     const CoalescedAxisInfo &axes,
                     ScheduleSymbolAxisSpaceCache *symbolAxisCache = nullptr);

void printScheduleProblemReport(const ScheduleProblem &problem,
                                llvm::raw_ostream &os);

} // namespace mlir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULEPROBLEMBUILDER_H
