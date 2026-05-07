//===- CannTargetProfileLoader.h - CANN target profile loader ---*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_TARGET_ASCEND_CANN_TARGET_PROFILE_LOADER_H
#define ASCEND_MLIR_TARGET_ASCEND_CANN_TARGET_PROFILE_LOADER_H

#include "Target/Ascend/TargetProfile.h"
#include "mlir/Support/LLVM.h"

namespace mlir::ascend {

class CannTargetProfileLoader {
public:
  static FailureOr<TargetProfile> load(StringRef cannRoot, StringRef socVersion);
};

} // namespace mlir::ascend

#endif // ASCEND_MLIR_TARGET_ASCEND_CANN_TARGET_PROFILE_LOADER_H
