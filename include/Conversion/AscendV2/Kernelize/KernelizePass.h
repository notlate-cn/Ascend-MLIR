//===- KernelizePass.h - Ascend V2 kernelize pass ---------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_PASS_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
}

namespace mlir::afir {

std::unique_ptr<Pass> createAscendKernelizePass();

} // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_PASS_H
