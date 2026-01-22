//===- AFIRToASCIRText.h - AFIR to ASCIR text conversion ------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_AFIRTOASCIR_AFIRTOASCIRTEXT_H
#define ASCEND_MLIR_CONVERSION_AFIRTOASCIR_AFIRTOASCIRTEXT_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
}

namespace mlir::afir {

std::unique_ptr<Pass> createConvertAFIRToASCIRTextPass();

}  // namespace mlir::afir

#endif  // ASCEND_MLIR_CONVERSION_AFIRTOASCIR_AFIRTOASCIRTEXT_H
