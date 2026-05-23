//===- FuseGatherElementwisePass.h - Fuse elementwise into gather *- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_FUSE_GATHER_ELEMENTWISE_PASS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_FUSE_GATHER_ELEMENTWISE_PASS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::afir {

std::unique_ptr<Pass> createFuseGatherElementwisePass();

}  // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_FUSE_GATHER_ELEMENTWISE_PASS_H
