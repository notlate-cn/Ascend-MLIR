#ifndef ASCEND_MLIR_CAPI_PASSES_H
#define ASCEND_MLIR_CAPI_PASSES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Register AFIR dialect and conversion passes.
MLIR_CAPI_EXPORTED void mlirRegisterAFIRPasses();

/// Register all Ascend conversion passes.
MLIR_CAPI_EXPORTED void mlirRegisterAscendPasses();

#ifdef __cplusplus
}
#endif

#endif // ASCEND_MLIR_CAPI_PASSES_H
