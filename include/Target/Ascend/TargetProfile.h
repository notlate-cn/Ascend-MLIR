//===- TargetProfile.h - Ascend target profile -----------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_TARGET_ASCEND_TARGET_PROFILE_H
#define ASCEND_MLIR_TARGET_ASCEND_TARGET_PROFILE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>
#include <utility>

namespace mlir::ascend {

enum class MemoryPlace { GM, L2, L1, L0A, L0B, L0C, UB };

struct TargetIdentity {
  std::string socVersion;
  std::string shortSocVersion;
  std::string npuArch;
};

struct TargetHardwareInfo {
  int64_t aiCoreCount = 0;
  int64_t cubeCoreCount = 0;
  int64_t vectorCoreCount = 0;
  int64_t l1SizeBytes = 0;
  int64_t ubSizeBytes = 0;
  bool supportBF16 = false;
  bool supportFixpipe = false;
};

struct TargetIntrinsicInfo {
  std::string name;
  SmallVector<std::string> dtypes;
};

struct TargetProfile {
  TargetIdentity identity;
  TargetHardwareInfo hardware;
  DenseMap<MemoryPlace, int64_t> capacityBytes;
  SmallVector<std::pair<MemoryPlace, MemoryPlace>> movementPaths;
  SmallVector<TargetIntrinsicInfo> intrinsics;
};

LogicalResult verifyTargetProfile(const TargetProfile &profile,
                                  raw_ostream &os);
StringRef stringifyMemoryPlace(MemoryPlace place);

} // namespace mlir::ascend

#endif // ASCEND_MLIR_TARGET_ASCEND_TARGET_PROFILE_H
