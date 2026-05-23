//===- LinalgToAscendCPass.h - Linalg to AscendC lowering -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_LOWERING_LINALG_TO_ASCENDC_PASS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_LOWERING_LINALG_TO_ASCENDC_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::afir {

std::unique_ptr<Pass> createLinalgToAscendCPass();

} // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_LOWERING_LINALG_TO_ASCENDC_PASS_H
