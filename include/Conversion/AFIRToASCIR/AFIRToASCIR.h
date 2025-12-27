//===- AFIRToASCIR.h - AFIR to ASC-IR conversion --------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CONVERSION_AFIRTOASCIR
#define AFIR_CONVERSION_AFIRTOASCIR

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace afir {

std::unique_ptr<Pass> createConvertAFIRToASCIRPass();

}  // namespace afir
}  // namespace mlir

#endif  // AFIR_CONVERSION_AFIRTOASCIR
