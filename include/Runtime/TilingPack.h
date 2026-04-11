#pragma once

#include "Runtime/TaskGraph.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace mlir::runtime {

llvm::Expected<std::vector<uint8_t>>
packTilingBytes(const std::optional<TilingBinding> &tiling);

} // namespace mlir::runtime
