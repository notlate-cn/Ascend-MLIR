//===- TargetModelVerifier.h - Ascend target model verifier ----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_TARGET_ASCEND_TARGET_MODEL_VERIFIER_H
#define ASCEND_MLIR_TARGET_ASCEND_TARGET_MODEL_VERIFIER_H

#include "Target/Ascend/TargetCostModel.h"
#include "Target/Ascend/TargetIntrinsicModel.h"
#include "Target/Ascend/TargetMemoryModel.h"
#include "Target/Ascend/TargetProfile.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::ascend {

class TargetModelVerifier {
public:
  LogicalResult verify(const TargetProfile &profile,
                       const TargetMemoryModel &memoryModel,
                       const TargetIntrinsicModel &intrinsicModel,
                       const TargetCostModel &costModel,
                       raw_ostream &os) const;
};

} // namespace mlir::ascend

#endif // ASCEND_MLIR_TARGET_ASCEND_TARGET_MODEL_VERIFIER_H
