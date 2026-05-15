//===- SchedulePersistentCacheIO.h - Schedule cache file IO -----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_PERSISTENT_CACHE_IO_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_PERSISTENT_CACHE_IO_H

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::afir::ascend::schedule {

FailureOr<llvm::SmallVector<std::string, 8>>
loadPersistentTuningCacheFile(llvm::StringRef path);

LogicalResult writePersistentTuningCacheFile(
    llvm::StringRef path, llvm::ArrayRef<std::string> signatures);

} // namespace mlir::afir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_PERSISTENT_CACHE_IO_H
