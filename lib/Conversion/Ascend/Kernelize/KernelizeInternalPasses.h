//===- KernelizeInternalPasses.h - Ascend kernelize internals ---*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_LIB_CONVERSION_ASCEND_KERNELIZE_INTERNAL_PASSES_H
#define ASCEND_MLIR_LIB_CONVERSION_ASCEND_KERNELIZE_INTERNAL_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::afir {

LogicalResult markStructuredOps(func::FuncOp funcOp);
LogicalResult fuseGatherElementwise(func::FuncOp funcOp);
LogicalResult annotateMixMatmulSemantics(func::FuncOp funcOp);

} // namespace mlir::afir

#endif // ASCEND_MLIR_LIB_CONVERSION_ASCEND_KERNELIZE_INTERNAL_PASSES_H
