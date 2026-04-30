#pragma once

#include "Runtime/TaskGraph.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace mlir::runtime {

struct OutputComparisonResult {
  bool passed = false;
  double maxAbsDiff = 0.0;
  double meanAbsDiff = 0.0;
  std::string errorMessage;
};

llvm::Expected<OutputComparisonResult>
compareRuntimeOutputs(llvm::ArrayRef<NDArray> actual,
                      llvm::ArrayRef<NDArray> expected, double atol,
                      double rtol);

} // namespace mlir::runtime
