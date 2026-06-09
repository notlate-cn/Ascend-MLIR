//===- ScheduleAxisContract.h - Ascend schedule axis contract -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULEAXISCONTRACT_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULEAXISCONTRACT_H

#include "KernelPatternView.h"
#include "ScheduleTypes.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::ascend::schedule {

class ScheduleSymbolAxisSpaceCache;

FailureOr<ScheduleAxisContract>
buildScheduleAxisContract(const KernelPatternView &pattern,
                          const CoalescedAxisInfo &axes,
                          ScheduleSymbolAxisSpaceCache *symbolAxisCache = nullptr);

void printScheduleAxisList(ArrayRef<SymbolicAxisRef> axes,
                           llvm::raw_ostream &os);

} // namespace mlir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULEAXISCONTRACT_H
