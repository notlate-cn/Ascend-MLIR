#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace mlir::afir {

std::unique_ptr<Pass> createAclnnFinalizeDeclPass();
std::unique_ptr<Pass> createRecognizeAttentionPass();

} // namespace mlir::afir
