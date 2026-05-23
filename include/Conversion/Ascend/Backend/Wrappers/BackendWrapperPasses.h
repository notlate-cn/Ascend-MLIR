//===- BackendWrapperPasses.h - Ascend backend wrapper passes ---*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_WRAPPER_PASSES_H
#define ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_WRAPPER_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::afir {

std::unique_ptr<Pass> createAscendParallelizePass();
std::unique_ptr<Pass> createAscendPrepareForEmitPass();
std::unique_ptr<Pass> createAscendCanonicalizeCannSignaturePass();

} // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_WRAPPER_PASSES_H
