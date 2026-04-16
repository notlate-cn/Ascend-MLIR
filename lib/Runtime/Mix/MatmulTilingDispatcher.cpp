#include "Runtime/Mix/MatmulTilingDispatcher.h"

#include "llvm/Support/Error.h"

namespace mlir::runtime {

namespace {

static std::string errorMessage(llvm::Error err) {
  std::string message;
  llvm::handleAllErrors(std::move(err),
                        [&](const llvm::ErrorInfoBase &info) {
                          message = info.message();
                        });
  return message;
}

static llvm::Error makeUnsupportedRequestError(
    const MatmulTilingRequest &request) {
  return llvm::createStringError(
      "no matmul tiling backend supports kernel %s", request.kernelName.c_str());
}

static llvm::Error makeNativeFailureError(const MatmulTilingBackend &nativeBackend,
                                          const MatmulTilingBackend &apiBackend,
                                          const MatmulTilingRequest &request,
                                          llvm::StringRef nativeFailure,
                                          llvm::StringRef apiFailure = {}) {
  if (apiFailure.empty()) {
    return llvm::createStringError(
        "native matmul tiling backend %s claimed support for kernel %s but failed: %s; "
        "api backend %s does not support the request",
        nativeBackend.name().str().c_str(), request.kernelName.c_str(),
        nativeFailure.str().c_str(), apiBackend.name().str().c_str());
  }
  return llvm::createStringError(
      "native matmul tiling backend %s claimed support for kernel %s but failed: %s; "
      "api backend %s also failed: %s",
      nativeBackend.name().str().c_str(), request.kernelName.c_str(),
      nativeFailure.str().c_str(), apiBackend.name().str().c_str(),
      apiFailure.str().c_str());
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

    std::string nativeFailure = errorMessage(nativeResult.takeError());
    if (!apiBackend.supports(request)) {
      return makeNativeFailureError(nativeBackend, apiBackend, request,
                                    nativeFailure);
    }

    auto apiResult = apiBackend.generate(request);
    if (apiResult)
      return std::move(apiResult);

    std::string apiFailure = errorMessage(apiResult.takeError());
    return makeNativeFailureError(nativeBackend, apiBackend, request,
                                  nativeFailure, apiFailure);
  }

  if (!apiBackend.supports(request)) {
    return makeUnsupportedRequestError(request);
  }

  return apiBackend.generate(request);
}

} // namespace mlir::runtime
