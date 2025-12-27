//===- Passes.h - Conversion passes declarations ----------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CONVERSION_PASSES
#define AFIR_CONVERSION_PASSES

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

} // namespace afir
} // namespace mlir

#endif // AFIR_CONVERSION_PASSES