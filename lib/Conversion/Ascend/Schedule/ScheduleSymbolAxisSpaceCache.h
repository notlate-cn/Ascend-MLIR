//===- ScheduleSymbolAxisSpaceCache.h - Shared symbol axes ------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SYMBOLAXISSPACECACHE_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SYMBOLAXISSPACECACHE_H

#include "Conversion/Ascend/Kernelize/Analysis/SymbolAxisSpace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::ascend::schedule {

class ScheduleSymbolAxisSpaceCache {
public:
  FailureOr<const ::mlir::ascend::kernelize::SymbolAxisSpace *>
  get(func::FuncOp func);

  unsigned getRequestCount() const { return requestCount; }
  unsigned getBuildCount() const { return buildCount; }
  unsigned getHitCount() const { return hitCount; }

private:
  llvm::DenseMap<Operation *, ::mlir::ascend::kernelize::SymbolAxisSpace>
      cachedByFunc;
  unsigned requestCount = 0;
  unsigned buildCount = 0;
  unsigned hitCount = 0;
};

void printSymbolAxisSpaceCacheReport(
    const ScheduleSymbolAxisSpaceCache &cache, llvm::raw_ostream &os);

} // namespace mlir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SYMBOLAXISSPACECACHE_H
