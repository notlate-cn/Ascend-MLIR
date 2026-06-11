//===- EliminateCfAssertPass.h - Remove cf.assert ops -----------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ELIMINATECFASSERT_H
#define ASCEND_MLIR_CONVERSION_ELIMINATECFASSERT_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
}

namespace mlir::afir {

std::unique_ptr<Pass> createEliminateCfAssertPass();

}  // namespace mlir::afir

#endif