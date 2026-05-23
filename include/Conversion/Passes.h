//===- Passes.h - Conversion passes declarations ----------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CONVERSION_PASSES
#define AFIR_CONVERSION_PASSES

#include "Conversion/AFIRToASCIR/AFIRToASCIR.h"
#include "Conversion/AFIRToASCIRText/AFIRToASCIRText.h"
#include "Conversion/Ascend/Passes.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::afir {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

std::unique_ptr<Pass> createConvertAFIRToASCIRTextPass();

} // namespace mlir::afir

#endif // AFIR_CONVERSION_PASSES
