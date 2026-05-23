//===- MarkStructuredOpsPass.h - Mark structured linalg ops -----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_MARK_STRUCTURED_OPS_PASS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_MARK_STRUCTURED_OPS_PASS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::afir {

std::unique_ptr<Pass> createMarkStructuredOpsPass();
std::unique_ptr<Pass> createAnnotateMixMatmulSemanticsPass();

}  // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_MARK_STRUCTURED_OPS_PASS_H
