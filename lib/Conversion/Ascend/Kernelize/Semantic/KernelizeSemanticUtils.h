//===- KernelizeSemanticUtils.h - Kernelize semantic helpers -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_SEMANTIC_UTILS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_SEMANTIC_UTILS_H

#include "Conversion/Ascend/Kernelize/KernelizeOpInterface.h"

#include "llvm/ADT/StringRef.h"

namespace mlir::ascend::kernelize {

bool matchLinalgSemanticOp(Operation *op);
bool matchArithConstantSemanticOp(Operation *op);
bool matchTensorViewSemanticOp(Operation *op);

LogicalResult populateLinalgSemanticInfo(Operation *op,
                                         KernelizeOpSemanticInfo &info,
                                         llvm::StringRef modelName);
LogicalResult populateArithConstantSemanticInfo(
    Operation *op, KernelizeOpSemanticInfo &info, llvm::StringRef modelName);
LogicalResult populateTensorViewSemanticInfo(Operation *op,
                                             KernelizeOpSemanticInfo &info,
                                             llvm::StringRef modelName);

} // namespace mlir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_SEMANTIC_UTILS_H
