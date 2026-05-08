//===- ScheduleCache.h - Ascend V2 schedule cache model -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_SCHEDULECACHE_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_SCHEDULECACHE_H

#include "Conversion/AscendV2/Schedule/ScheduleTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSet.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir::afir::ascend::v2::schedule {

class ScheduleCacheModel {
public:
  ScheduleCacheReport
  recordScheduleDecisionSet(const ScheduleProblem &problem,
                            const ScheduleDecisionSet &decisionSet);

  void recordGuardBudgetPruned(unsigned count);
  void recordNegativeCacheEntry();

  const ScheduleCacheReport &getReport() const { return report; }
  llvm::ArrayRef<ShapeBucketKey> getShapeBucketKeys() const {
    return shapeBucketKeys;
  }
  llvm::ArrayRef<TuningResultKey> getTuningResultKeys() const {
    return tuningResultKeys;
  }

private:
  ScheduleCacheReport report;
  llvm::StringSet<> shapeBucketSignatures;
  llvm::StringSet<> tuningResultSignatures;
  SmallVector<ShapeBucketKey, 8> shapeBucketKeys;
  SmallVector<TuningResultKey, 8> tuningResultKeys;
};

void printScheduleCacheReport(const ScheduleCacheReport &report,
                              llvm::raw_ostream &os);

void printScheduleCacheReport(const ScheduleCacheModel &cacheModel,
                              llvm::raw_ostream &os);

} // namespace mlir::afir::ascend::v2::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_SCHEDULECACHE_H
