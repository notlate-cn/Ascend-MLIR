//===- CanonicalizeCannSignaturePass.h ----------------------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_CODEGEN_CANONICALIZE_CANN_SIGNATURE_PASS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_CODEGEN_CANONICALIZE_CANN_SIGNATURE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::afir {
std::unique_ptr<Pass> createCanonicalizeCannSignaturePass();
}  // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_CODEGEN_CANONICALIZE_CANN_SIGNATURE_PASS_H
