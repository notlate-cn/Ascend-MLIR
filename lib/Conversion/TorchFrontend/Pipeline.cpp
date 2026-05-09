#include "Conversion/TorchFrontend/TorchFrontendPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace mlir::afir {

void registerTorchFrontendPipeline() {
  PassPipelineRegistration<>(
      "torch-normalize",
      "Torch frontend normalization: remove cf.assert + lower tm_tensor ops",
      [](OpPassManager &pm) {
        // Remove shape-guard assertions inserted by torch-mlir.
        pm.addNestedPass<func::FuncOp>(createRemoveCfAssertPass());
        // Lower tm_tensor.attention → linalg.generic {ascendc.unit}.
        pm.addPass(createConvertTmTensorAttentionPass());
      });
}

} // namespace mlir::afir