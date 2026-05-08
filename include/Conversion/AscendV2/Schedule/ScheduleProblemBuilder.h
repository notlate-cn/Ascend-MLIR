//===- ScheduleProblemBuilder.h - Ascend V2 schedule problem ---*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_SCHEDULEPROBLEMBUILDER_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_SCHEDULEPROBLEMBUILDER_H

#include "Conversion/AscendV2/Schedule/ScheduleTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::afir::ascend::v2::schedule {

FailureOr<ScheduleProblem>
buildScheduleProblem(const KernelPatternView &pattern,
                     const CoalescedAxisInfo &axes);

void printScheduleProblemReport(const ScheduleProblem &problem,
                                llvm::raw_ostream &os);

} // namespace mlir::afir::ascend::v2::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_SCHEDULEPROBLEMBUILDER_H
