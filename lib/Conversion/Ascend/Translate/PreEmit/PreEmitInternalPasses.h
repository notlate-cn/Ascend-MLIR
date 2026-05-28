//===- PreEmitInternalPasses.h - Ascend pre-emit internal passes -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_LIB_CONVERSION_ASCEND_TRANSLATE_PREEMIT_INTERNAL_PASSES_H
#define ASCEND_MLIR_LIB_CONVERSION_ASCEND_TRANSLATE_PREEMIT_INTERNAL_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>

namespace mlir::ascend {

LogicalResult annotateAscendKernelKind(func::FuncOp funcOp);

std::unique_ptr<Pass> createAscendPreEmitAnnotateKernelKindPass();
std::unique_ptr<Pass> createAscendPreEmitCanonicalizeCannSignaturePass();
std::unique_ptr<Pass> createAscendPreEmitParallelizePass();
std::unique_ptr<Pass> createAscendPreEmitPrepareForEmitPass();

} // namespace mlir::ascend

#endif // ASCEND_MLIR_LIB_CONVERSION_ASCEND_TRANSLATE_PREEMIT_INTERNAL_PASSES_H
