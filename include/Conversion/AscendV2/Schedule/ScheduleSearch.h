//===- ScheduleSearch.h - Ascend V2 schedule search -----------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_SCHEDULESEARCH_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_SCHEDULESEARCH_H

#include "Conversion/AscendV2/Schedule/ScheduleTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir::afir::ascend::v2::schedule {

struct ScheduleSearchResult {
  unsigned generatedCount = 0;
  SmallVector<ScheduleInstance, 4> keptInstances;
};

ScheduleSearchResult searchScheduleInstancesWithStats(
    const ScheduleProblem &problem, ArrayRef<ScheduleTemplate> templates,
    const ScheduleSearchOptions &options = ScheduleSearchOptions());

SmallVector<ScheduleInstance, 4> searchScheduleInstances(
    const ScheduleProblem &problem, ArrayRef<ScheduleTemplate> templates,
    const ScheduleSearchOptions &options = ScheduleSearchOptions());

void printScheduleSearchReport(StringRef kernelId, unsigned generatedCount,
                               const ScheduleSearchOptions &options,
                               ArrayRef<ScheduleInstance> keptInstances,
                               llvm::raw_ostream &os);

} // namespace mlir::afir::ascend::v2::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_SCHEDULESEARCH_H
