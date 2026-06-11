#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace mlir::afir {

std::unique_ptr<Pass> createAclnnFinalizeDeclPass();
std::unique_ptr<Pass> createFuseTransposeIntoElementwisePass();
std::unique_ptr<Pass> createRecognizeAttentionPass();
std::unique_ptr<Pass> createRecognizeBatchNormPass();
std::unique_ptr<Pass> createRecognizeEmbeddingPass();
std::unique_ptr<Pass> createRecognizeLayerNormPass();
std::unique_ptr<Pass> createLowerBroadcastExtractPass();

} // namespace mlir::afir
