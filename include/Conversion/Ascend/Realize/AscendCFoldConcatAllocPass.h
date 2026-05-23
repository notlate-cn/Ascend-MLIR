//===- AscendCFoldConcatAllocPass.h - Fold concat allocs --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDCFOLDCONCATALLOC_H
#define ASCEND_MLIR_CONVERSION_ASCENDCFOLDCONCATALLOC_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
}

namespace mlir::afir {

std::unique_ptr<Pass> createAscendCFoldConcatAllocPass();

}  // namespace mlir::afir

#endif
