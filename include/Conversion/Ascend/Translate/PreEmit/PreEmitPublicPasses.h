//===- PreEmitPublicPasses.h - Ascend pre-emit public passes ---*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_PREEMIT_PREEMIT_PUBLIC_PASSES_H
#define ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_PREEMIT_PREEMIT_PUBLIC_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::ascend {

std::unique_ptr<Pass> createAscendParallelizePass();
std::unique_ptr<Pass> createAscendPrepareForEmitPass();
std::unique_ptr<Pass> createAscendCanonicalizeCannSignaturePass();

} // namespace mlir::ascend

#endif // ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_PREEMIT_PREEMIT_PUBLIC_PASSES_H
