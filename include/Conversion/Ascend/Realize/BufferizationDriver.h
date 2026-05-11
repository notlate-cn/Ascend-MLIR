//===- BufferizationDriver.h - Ascend realize buffer facts -----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_BUFFERIZATIONDRIVER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_BUFFERIZATIONDRIVER_H

#include "Conversion/Ascend/Realize/RealizeTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::afir::ascend::realize {

class BufferizationDriver {
public:
  FailureOr<SmallVector<BufferizedKernelIR, 4>>
  collectTensorFacts(ModuleOp module) const;
  LogicalResult runOneShotBufferize(ModuleOp module) const;
};

} // namespace mlir::afir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_BUFFERIZATIONDRIVER_H
