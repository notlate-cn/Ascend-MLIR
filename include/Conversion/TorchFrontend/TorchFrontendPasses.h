#ifndef ASCEND_MLIR_CONVERSION_TORCHFRONTEND_PASSES_H
#define ASCEND_MLIR_CONVERSION_TORCHFRONTEND_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::afir {

std::unique_ptr<Pass> createConvertTmTensorAttentionPass();
std::unique_ptr<Pass> createRemoveCfAssertPass();

void registerTorchFrontendPipeline();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Conversion/TorchFrontend/TorchFrontendPasses.h.inc"

} // namespace mlir::afir

#endif