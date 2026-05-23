//===- CodegenPasses.h - Ascend backend codegen internals -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_LIB_CONVERSION_ASCEND_BACKEND_CODEGEN_PASSES_H
#define ASCEND_MLIR_LIB_CONVERSION_ASCEND_BACKEND_CODEGEN_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>

namespace mlir::afir {

LogicalResult annotateAscendKernelKind(func::FuncOp funcOp);

std::unique_ptr<Pass> createAscendCodegenAnnotateKernelKindPass();
std::unique_ptr<Pass> createAscendCodegenCanonicalizeCannSignaturePass();
std::unique_ptr<Pass> createAscendCodegenParallelizePass();
std::unique_ptr<Pass> createAscendCodegenPrepareForEmitPass();

} // namespace mlir::afir

#endif // ASCEND_MLIR_LIB_CONVERSION_ASCEND_BACKEND_CODEGEN_PASSES_H
