#include "Runtime/Mix/MatmulTilingDispatcher.h"

#include "llvm/Support/Error.h"

namespace mlir::runtime {

namespace {

class MatmulTilingDispatchError final
    : public llvm::ErrorInfo<MatmulTilingDispatchError> {
public:
  explicit MatmulTilingDispatchError(std::string message)
      : message_(std::move(message)) {}

  void log(llvm::raw_ostream &os) const override { os << message_; }

  std::error_code convertToErrorCode() const override {
    return std::make_error_code(std::errc::invalid_argument);
  }

  static char ID;

private:
  std::string message_;
};

char MatmulTilingDispatchError::ID = 0;

static llvm::Error makeUnsupportedRequestError(
    const MatmulTilingRequest &request) {
  return llvm::make_error<MatmulTilingDispatchError>(
      "no matmul tiling backend supports kernel " + request.kernelName);
}

} // namespace

llvm::Expected<MatmulTilingResult>
dispatchMatmulTiling(const MatmulTilingRequest &request,
                     const MatmulTilingBackend &nativeBackend,
                     const MatmulTilingBackend &apiBackend) {
  if (nativeBackend.supports(request)) {
    auto nativeResult = nativeBackend.generate(request);
    if (nativeResult)
      return std::move(nativeResult);
    llvm::consumeError(nativeResult.takeError());
  }

  if (!apiBackend.supports(request)) {
    return makeUnsupportedRequestError(request);
  }

  return apiBackend.generate(request);
}

} // namespace mlir::runtime
