//===- AscendCBufferPlacementPass.h - AscendC buffer placement -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDCBUFFERPLACEMENT_ASCENDCBUFFERPLACEMENTPASS_H
#define ASCEND_MLIR_CONVERSION_ASCENDCBUFFERPLACEMENT_ASCENDCBUFFERPLACEMENTPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
}

namespace mlir::afir {

std::unique_ptr<Pass> createAscendCBufferPlacementPass();

}  // namespace mlir::afir

#endif  // ASCEND_MLIR_CONVERSION_ASCENDCBUFFERPLACEMENT_ASCENDCBUFFERPLACEMENTPASS_H
