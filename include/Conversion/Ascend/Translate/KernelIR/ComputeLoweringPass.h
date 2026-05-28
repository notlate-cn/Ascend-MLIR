//===- ComputeLoweringPass.h - Ascend compute lowering pass -----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_COMPUTE_LOWERING_PASS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_COMPUTE_LOWERING_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::ascend {

std::unique_ptr<Pass> createAscendComputeLowerPass();

} // namespace mlir::ascend

#endif // ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_COMPUTE_LOWERING_PASS_H
