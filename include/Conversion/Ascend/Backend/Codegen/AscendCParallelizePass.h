//===- AscendCParallelizePass.h - AscendC parallelize pass -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_CODEGEN_ASCENDC_PARALLELIZE_PASS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_CODEGEN_ASCENDC_PARALLELIZE_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
}

namespace mlir::afir {

std::unique_ptr<Pass> createAscendCParallelizePass();

}  // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_CODEGEN_ASCENDC_PARALLELIZE_PASS_H
