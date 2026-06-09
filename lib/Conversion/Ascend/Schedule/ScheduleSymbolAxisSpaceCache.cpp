//===- ScheduleSymbolAxisSpaceCache.cpp - Shared symbol axes --------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "ScheduleSymbolAxisSpaceCache.h"

using namespace mlir;
using namespace mlir::ascend::schedule;

FailureOr<const ::mlir::ascend::kernelize::SymbolAxisSpace *>
ScheduleSymbolAxisSpaceCache::get(func::FuncOp func) {
  ++requestCount;
  Operation *funcOp = func.getOperation();
  auto it = cachedByFunc.find(funcOp);
  if (it != cachedByFunc.end()) {
    ++hitCount;
    return &it->second;
  }

  FailureOr<::mlir::ascend::kernelize::SymbolAxisSpace> built =
      ::mlir::ascend::kernelize::buildSymbolAxisSpace(func);
  if (failed(built))
    return failure();

  ++buildCount;
  auto inserted = cachedByFunc.try_emplace(funcOp, std::move(*built));
  return &inserted.first->second;
}

void mlir::ascend::schedule::printSymbolAxisSpaceCacheReport(
    const ScheduleSymbolAxisSpaceCache &cache, llvm::raw_ostream &os) {
  os << "SymbolAxisSpaceCache:\n";
  os << "  requests = " << cache.getRequestCount() << "\n";
  os << "  builds = " << cache.getBuildCount() << "\n";
  os << "  hits = " << cache.getHitCount() << "\n";
}
