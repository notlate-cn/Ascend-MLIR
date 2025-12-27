//===- Passes.h - AFIR dialect passes ---------------------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFIR_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_AFIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace afir {

// Canonicalization pass
std::unique_ptr<Pass> createAFIRCanonicalizePass();

// Shape inference pass
std::unique_ptr<Pass> createAFIRShapeInferencePass();

// Generate pass registration declarations
#define GEN_PASS_REGISTRATION
#include "Dialect/AFIR/Transforms/Passes.h.inc"

}  // namespace afir
}  // namespace mlir

#endif  // MLIR_DIALECT_AFIR_TRANSFORMS_PASSES_H
