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

namespace mlir::ascend::schedule {

class ScheduleSymbolAxisSpaceCache;

FailureOr<CoalescedAxisInfo>
coalesceAxes(const KernelPatternView &pattern,
             ScheduleSymbolAxisSpaceCache *symbolAxisCache = nullptr);

void printAxisCoalescingReport(StringRef kernelId,
                               const CoalescedAxisInfo &info,
                               llvm::raw_ostream &os);

} // namespace mlir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_AXISCOALESCER_H
