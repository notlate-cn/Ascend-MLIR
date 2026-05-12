//===- Passes.h - AFIR dialect passes ---------------------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFIR_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_AFIR_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

std::unique_ptr<Pass> createLinalgInferShapePass();

std::unique_ptr<Pass> createLinalgMarkPass();

std::unique_ptr<Pass> createLinalgAddBroadcastPass();

std::unique_ptr<Pass> createAFIRSymbolizeShapesPass();

std::unique_ptr<Pass> createAFIRVerifySymbolicShapesPass();

namespace afir {

// Canonicalization pass
std::unique_ptr<Pass> createAFIRCanonicalizePass();

// Shape inference pass
std::unique_ptr<Pass> createAFIRShapeInferencePass();

std::unique_ptr<Pass> createAFIRAddAxisPass();

// Generate pass registration declarations
#define GEN_PASS_REGISTRATION
#include "Dialect/AFIR/Transforms/Passes.h.inc"

}  // namespace afir
}  // namespace mlir

#endif  // MLIR_DIALECT_AFIR_TRANSFORMS_PASSES_H
