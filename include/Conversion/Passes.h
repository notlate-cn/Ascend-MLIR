//===- Passes.h - Conversion passes declarations ----------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_PASSES_H
#define MLIR_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace afir {

// StableHLO to AFIR conversion pass
// std::unique_ptr<Pass> createConvertStableHLOToAFIRPass();

// AFIR to ASC-IR conversion pass
std::unique_ptr<Pass> createConvertAFIRToASCIRPass();

#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"
// /// Register all conversion passes
// inline void registerConversionPasses() {
//   // Passes are auto-registered via PassWrapper
//   createConvertAFIRToASCIRPass();
// }

} // namespace afir
} // namespace mlir

#endif // MLIR_CONVERSION_PASSES_H