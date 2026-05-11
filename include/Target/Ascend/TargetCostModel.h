//===- TargetCostModel.h - Ascend target cost model ------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_TARGET_ASCEND_TARGET_COST_MODEL_H
#define ASCEND_MLIR_TARGET_ASCEND_TARGET_COST_MODEL_H

#include "Target/Ascend/TargetMemoryModel.h"
#include "Target/Ascend/TargetProfile.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>

namespace mlir::ascend {

struct PathCost {
  int64_t startupCycles = 0;
  int64_t bytesPerCycle = 0;
  bool overlapsCompute = false;
  std::string rateSection;
  std::string rateName;
};

class TargetCostModel {
public:
  FailureOr<int64_t> getMemoryRate(StringRef section, StringRef name) const;
  FailureOr<int64_t> getPreferredMemoryRate(StringRef name) const;
  FailureOr<PathCost> getPathCost(const PathEdge &edge) const;
  FailureOr<int64_t> estimateTransferCycles(const PathEdge &edge,
                                            int64_t bytes) const;

private:
  friend class TargetCostModelBuilder;
  llvm::StringMap<int64_t> ratesByQualifiedName;
  llvm::StringMap<int64_t> preferredRatesByName;
  DenseMap<PathEdge, PathCost> pathCosts;
};

class TargetCostModelBuilder {
public:
  FailureOr<TargetCostModel> build(const TargetProfile &profile,
                                   const TargetMemoryModel &memoryModel,
                                   raw_ostream &os) const;
};

} // namespace mlir::ascend

#endif // ASCEND_MLIR_TARGET_ASCEND_TARGET_COST_MODEL_H
