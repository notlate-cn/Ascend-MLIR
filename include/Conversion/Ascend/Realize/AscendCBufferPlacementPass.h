//===- AscendCBufferPlacementPass.h - AscendC buffer placement -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_ASCENDC_BUFFER_PLACEMENT_PASS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_ASCENDC_BUFFER_PLACEMENT_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
}

namespace mlir::afir {

std::unique_ptr<Pass> createAscendCBufferPlacementPass();

}  // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_ASCENDC_BUFFER_PLACEMENT_PASS_H
