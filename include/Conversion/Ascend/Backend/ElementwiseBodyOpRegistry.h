//===- ElementwiseBodyOpRegistry.h - Elementwise body op registry --*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_ELEMENTWISEBODYOPREGISTRY_H
#define ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_ELEMENTWISEBODYOPREGISTRY_H

#include "Conversion/Ascend/Backend/BackendSupportMatrix.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

#include <functional>

namespace mlir::afir::ascend::backend {

using UnaryEmitter  = std::function<void(mlir::OpBuilder &, mlir::Location,
                                         mlir::Value dst, mlir::Value src,
                                         mlir::Value count)>;
using BinaryEmitter = std::function<void(mlir::OpBuilder &, mlir::Location,
                                         mlir::Value dst, mlir::Value src0,
                                         mlir::Value src1, mlir::Value count)>;

/// One entry in the elementwise body op registry.
/// Exactly one of unaryEmitter / binaryEmitter is non-null.
struct ElementwiseBodyOpEntry {
  llvm::StringRef dialectOpName; // e.g. "arith.addf", "math.exp"
  ComputeKind kind;
  UnaryEmitter  unaryEmitter;    // set for unary ops (exp, sqrt, neg, ...)
  BinaryEmitter binaryEmitter;   // set for binary ops (add, sub, mul, ...)
};

/// Register one entry. Second call with same dialectOpName is a no-op.
void registerElementwiseBodyOp(ElementwiseBodyOpEntry entry);

/// Register all built-in arith/math entries. Called automatically on first
/// lookup; exposed for explicit initialization in tests.
void registerBuiltinElementwiseBodyOps();

/// Returns nullptr if opName is not registered.
const ElementwiseBodyOpEntry *
lookupElementwiseBodyOp(llvm::StringRef dialectOpName);

} // namespace mlir::afir::ascend::backend

#endif // ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_ELEMENTWISEBODYOPREGISTRY_H
