//===- NormalizePass.h - Ascend V2 normalize pass ---------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_NORMALIZE_PASS_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_NORMALIZE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
}

namespace mlir::afir {

std::unique_ptr<Pass> createAscendNormalizePass();

} // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_NORMALIZE_PASS_H
