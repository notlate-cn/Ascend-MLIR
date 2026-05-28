//===- LinalgBodyClassifier.h - Ascend linalg body classifier --*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_CAPABILITIES_LINALG_BODY_CLASSIFIER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_CAPABILITIES_LINALG_BODY_CLASSIFIER_H

#include "Conversion/Ascend/Translate/KernelIR/Capabilities/BackendSupportMatrix.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"

namespace mlir::ascend::backend {

bool hasIdentityOutputMaps(linalg::LinalgOp linalgOp);

ComputeKind
classifyLinalgComputeKind(Operation *op,
                          const AscendBackendSupportMatrix &matrix);

ComputeKind classifyBackendReductionBody(linalg::GenericOp generic,
                                        const AscendBackendSupportMatrix &matrix);

bool isSupportedBackendVectorOutput(linalg::LinalgOp linalgOp,
                                   const AscendBackendSupportMatrix &matrix);

bool isSupportedBackendGatherOutput(linalg::LinalgOp linalgOp,
                                   const AscendBackendSupportMatrix &matrix);

bool isSupportedBackendFinalOutput(linalg::LinalgOp linalgOp,
                                  const AscendBackendSupportMatrix &matrix);

} // namespace mlir::ascend::backend

#endif // ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_CAPABILITIES_LINALG_BODY_CLASSIFIER_H
