//===- AxisCoalescer.h - Ascend logical axis coalescing -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_AXISCOALESCER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_AXISCOALESCER_H

#include "ScheduleTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::afir::ascend::schedule {

FailureOr<CoalescedAxisInfo> coalesceAxes(const KernelPatternView &pattern);

void printAxisCoalescingReport(StringRef kernelId,
                               const CoalescedAxisInfo &info,
                               llvm::raw_ostream &os);

} // namespace mlir::afir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_AXISCOALESCER_H
