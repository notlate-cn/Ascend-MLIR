#include "Runtime/Mix/NativeMatmulTilingBackend.h"

#include "llvm/Support/Error.h"

namespace mlir::runtime {

llvm::StringRef NativeMatmulTilingBackend::name() const { return "native"; }

bool NativeMatmulTilingBackend::supports(
    const MatmulTilingRequest &) const {
  // Native matmul tiling is intentionally unavailable in production selection
  // until there is a matching execution-side payload contract.
  return false;
}

llvm::Expected<MatmulTilingResult>
NativeMatmulTilingBackend::generate(const MatmulTilingRequest &request) const {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "native matmul tiling backend is not available for kernel %s",
      request.kernelName.c_str());
}

} // namespace mlir::runtime

