//===- RealizePass.h - Ascend realize pass ------------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_PASS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
}

namespace mlir::afir {

std::unique_ptr<Pass> createAscendRealizePass();

} // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_PASS_H
