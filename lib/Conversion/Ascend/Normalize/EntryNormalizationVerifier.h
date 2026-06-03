//===- EntryNormalizationVerifier.h - Normalize verifier -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_NORMALIZE_ENTRYNORMALIZATIONVERIFIER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_NORMALIZE_ENTRYNORMALIZATIONVERIFIER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::ascend::normalize {
LogicalResult verifyEntryNormalization(ModuleOp module);
}

#endif
