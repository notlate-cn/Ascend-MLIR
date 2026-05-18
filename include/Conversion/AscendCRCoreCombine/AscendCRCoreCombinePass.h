//===- AscendCRCoreCombinePass.h - RCore cross-core combine -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDCRCORECOMBINE_PASS_H
#define ASCEND_MLIR_CONVERSION_ASCENDCRCORECOMBINE_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
}

namespace mlir::afir {

std::unique_ptr<Pass> createAscendCRCoreCombinePass();

}  // namespace mlir::afir

#endif  // ASCEND_MLIR_CONVERSION_ASCENDCRCORECOMBINE_PASS_H
