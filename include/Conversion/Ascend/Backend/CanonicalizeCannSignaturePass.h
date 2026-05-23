//===- CanonicalizeCannSignaturePass.h ----------------------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CONVERSION_CANONICALIZECANNSIGNATURE_PASS_H
#define AFIR_CONVERSION_CANONICALIZECANNSIGNATURE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::afir {
std::unique_ptr<Pass> createCanonicalizeCannSignaturePass();
}  // namespace mlir::afir

#endif // AFIR_CONVERSION_CANONICALIZECANNSIGNATURE_PASS_H
