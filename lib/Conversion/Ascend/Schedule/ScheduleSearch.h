//===- ScheduleSearch.h - Ascend schedule search -----------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULESEARCH_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULESEARCH_H

#include "ScheduleTypes.h"
#include "ScheduleTemplateImplementation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir::ascend::schedule {

struct ScheduleSearchResult {
  unsigned generatedCount = 0;
  unsigned prunedByGuardBudget = 0;
  SmallVector<ScheduleInstance, 4> keptInstances;
};

ScheduleSearchResult searchScheduleInstancesWithStats(
    const ScheduleProblem &problem,
    ArrayRef<const ScheduleTemplateImplementation *> templates,
    const ScheduleSearchOptions &options = ScheduleSearchOptions());

SmallVector<ScheduleInstance, 4> searchScheduleInstances(
    const ScheduleProblem &problem,
    ArrayRef<const ScheduleTemplateImplementation *> templates,
    const ScheduleSearchOptions &options = ScheduleSearchOptions());

void printScheduleSearchReport(StringRef kernelId, unsigned generatedCount,
                               const ScheduleSearchOptions &options,
                               ArrayRef<ScheduleInstance> keptInstances,
                               llvm::raw_ostream &os);

void printScheduleGuardsReport(const ScheduleProblem &problem,
                               const ScheduleSearchResult &result,
                               llvm::raw_ostream &os);

} // namespace mlir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULESEARCH_H
