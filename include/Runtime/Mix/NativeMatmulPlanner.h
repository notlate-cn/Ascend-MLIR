#pragma once

#include "Runtime/Mix/MatmulTilingTypes.h"

#include "llvm/Support/Error.h"

namespace mlir::runtime {

struct NativeMatmulPlan {
  MatrixTraverseKind traverse = MatrixTraverseKind::FirstM;
  uint32_t blockDim = 1;
  bool splitKEnabled = false;
  int64_t tileM = 0;
  int64_t tileN = 0;
  int64_t tileK = 0;
};

class NativeMatmulPlanner {
public:
  static bool supports(const MatmulTilingRequest &request);
  static llvm::Expected<NativeMatmulPlan>
  buildPlan(const MatmulTilingRequest &request);
};

} // namespace mlir::runtime
