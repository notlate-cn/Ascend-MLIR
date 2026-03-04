//===- AFIRTransformOps.h - AFIR transformation ops ----------------------===//
//
// Part of Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFIR_TRANSFORMOPS_AFIRTRANSFORMOPS_H
#define MLIR_DIALECT_AFIR_TRANSFORMOPS_AFIRTRANSFORMOPS_H

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Dialect/AFIR/TransformOps/AFIRTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace afir {

void registerTransformDialectExtension(DialectRegistry &registry);

} // namespace afir
} // namespace mlir

#endif // MLIR_DIALECT_AFIR_TRANSFORMOPS_AFIRTRANSFORMOPS_H
