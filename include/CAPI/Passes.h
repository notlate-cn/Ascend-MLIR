#ifndef STABLEHLO_INTEGRATIONS_C_STABLEHLO_PASSES_H
#define STABLEHLO_INTEGRATIONS_C_STABLEHLO_PASSES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Register all compiler passes of StableHLO.
MLIR_CAPI_EXPORTED void mlirRegisterAFIRPasses();

#ifdef __cplusplus
}
#endif

#endif  // STABLEHLO_INTEGRATIONS_C_STABLEHLO_PASSES_H