//===- Passes.h - Conversion passes declarations ----------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CONVERSION_PASSES
#define AFIR_CONVERSION_PASSES

#include "Conversion/AFIRToASCIR/AFIRToASCIR.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace afir {

#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

} // namespace afir
} // namespace mlir

#endif // AFIR_CONVERSION_PASSES
