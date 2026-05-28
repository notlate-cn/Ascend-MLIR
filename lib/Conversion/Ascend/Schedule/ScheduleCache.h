//===- ScheduleCache.h - Ascend schedule cache model -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULECACHE_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULECACHE_H

#include "ScheduleTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir::ascend::schedule {

class ScheduleCacheModel {
public:
  void seedPersistentTuningSignatures(ArrayRef<std::string> signatures);

  ScheduleCacheReport
  recordScheduleDecisionSet(const ScheduleProblem &problem,
                            const ScheduleDecisionSet &decisionSet);

  void recordGuardBudgetPruned(unsigned count);
  void recordNegativeCacheEntry();
  void recordNegativeCacheHit();

  const ScheduleCacheReport &getReport() const { return report; }
  llvm::ArrayRef<ShapeBucketKey> getShapeBucketKeys() const {
    return shapeBucketKeys;
  }
  llvm::ArrayRef<TuningResultKey> getTuningResultKeys() const {
    return tuningResultKeys;
  }
  llvm::ArrayRef<std::string> getPersistentTuningSignatures() const {
    return persistentTuningSignatures;
  }

private:
  ScheduleCacheReport report;
  llvm::StringSet<> shapeBucketSignatures;
  llvm::StringSet<> tuningResultSignatures;
  llvm::StringSet<> seededPersistentTuningSignatures;
  SmallVector<std::string, 8> persistentTuningSignatures;
  SmallVector<ShapeBucketKey, 8> shapeBucketKeys;
  SmallVector<TuningResultKey, 8> tuningResultKeys;
};

std::string serializeScheduleDims(ArrayRef<int64_t> dims);
std::string serializeTuningResultKey(const TuningResultKey &key);
std::string getTuningResultSignature(const TuningResultKey &key);

void printScheduleCacheReport(const ScheduleCacheReport &report,
                              llvm::raw_ostream &os);

void printScheduleCacheReport(const ScheduleCacheModel &cacheModel,
                              llvm::raw_ostream &os);

} // namespace mlir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULECACHE_H
