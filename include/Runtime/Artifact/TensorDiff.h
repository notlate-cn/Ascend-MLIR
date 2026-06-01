#pragma once

#include "llvm/Support/Error.h"

#include <cstddef>
#include <string>

namespace mlir::runtime {

struct TensorDiffRequest {
  std::string manifestPath;
  std::string outputSummaryPath;
};

struct TensorDiffRunResult {
  bool passed = false;
  size_t comparisonCount = 0;
  size_t failedCount = 0;
};

llvm::Expected<TensorDiffRunResult>
emitTensorDiffSummaryFromManifest(const TensorDiffRequest &request);

} // namespace mlir::runtime
