//===- LinalgBodyClassifier.h - Ascend linalg body classifier --*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_LINALGBODYCLASSIFIER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_LINALGBODYCLASSIFIER_H

#include "Conversion/Ascend/Backend/BackendSupportMatrix.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"

namespace mlir::afir::ascend::backend {

bool hasIdentityOutputMaps(linalg::LinalgOp linalgOp);

ComputeKind
classifyLinalgComputeKind(Operation *op,
                          const AscendBackendSupportMatrix &matrix);

ComputeKind classifyPhase5ReductionBody(linalg::GenericOp generic,
                                        const AscendBackendSupportMatrix &matrix);

bool isSupportedPhase5VectorOutput(linalg::LinalgOp linalgOp,
                                   const AscendBackendSupportMatrix &matrix);

bool isSupportedPhase5GatherOutput(linalg::LinalgOp linalgOp,
                                   const AscendBackendSupportMatrix &matrix);

bool isSupportedPhase5FinalOutput(linalg::LinalgOp linalgOp,
                                  const AscendBackendSupportMatrix &matrix);

} // namespace mlir::afir::ascend::backend

#endif // ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_LINALGBODYCLASSIFIER_H
