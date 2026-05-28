//===- ScheduleTuningDB.h - Ascend schedule tuning DB -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULETUNINGDB_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULETUNINGDB_H

#include "ScheduleTypes.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <optional>

namespace mlir::ascend::schedule {

inline constexpr unsigned kScheduleTuningDBSchemaVersion = 1;

struct ScheduleTuningRecord {
  std::string target;
  std::string policy;
  std::string signature;
  std::string family;
  std::string templateName;
  std::string resultShape;
  std::string tileShape;
  std::optional<int64_t> score;
  std::optional<int64_t> cycleCount;
  std::string profilePath;
  std::string source;
};

struct ScheduleNegativeRecord {
  std::string target;
  std::string policy;
  std::string signature;
  std::string reason;
  std::optional<int64_t> score;
  std::string profilePath;
  std::string source;
};

struct ScheduleTuningDatabase {
  SmallVector<ScheduleTuningRecord, 8> records;
  SmallVector<ScheduleNegativeRecord, 4> negativeRecords;
};

FailureOr<ScheduleTuningDatabase> loadScheduleTuningDBFile(llvm::StringRef path);
LogicalResult writeScheduleTuningDBFile(llvm::StringRef path,
                                        const ScheduleTuningDatabase &db);

SmallVector<std::string, 8>
collectMatchingTuningSignatures(const ScheduleTuningDatabase &db,
                                llvm::StringRef target,
                                llvm::StringRef policy);

LogicalResult appendTuningResultRecords(ScheduleTuningDatabase &db,
                                        llvm::StringRef target,
                                        llvm::StringRef policy,
                                        ArrayRef<TuningResultKey> keys);

} // namespace mlir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULETUNINGDB_H
