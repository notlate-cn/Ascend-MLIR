//===- KernelizePass.h - Ascend kernelize pass ---------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_PASS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
}

namespace mlir::ascend {

std::unique_ptr<Pass> createAscendKernelizePass();

} // namespace mlir::ascend

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_PASS_H
