#include "Runtime/Mix/MatmulTilingDispatcher.h"

#include "llvm/Support/Error.h"

namespace mlir::runtime {

namespace {

static llvm::Error makeUnsupportedRequestError(
    const MatmulTilingRequest &request) {
  return llvm::createStringError(
      "no matmul tiling backend supports kernel %s", request.kernelName.c_str());
}

static llvm::Error makeNativeFailureContext(
    const MatmulTilingBackend &nativeBackend, const MatmulTilingBackend &apiBackend,
    const MatmulTilingRequest &request, bool apiSupports) {
  const std::string nativeName = nativeBackend.name().str();
  const std::string apiName = apiBackend.name().str();
  if (apiSupports) {
    return llvm::createStringError(
        "native matmul tiling backend %s claimed support for kernel %s but failed; "
        "api backend %s also failed",
        nativeName.c_str(), request.kernelName.c_str(), apiName.c_str());
  }
  return llvm::createStringError(
      "native matmul tiling backend %s claimed support for kernel %s but failed; "
      "api backend %s does not support the request",
      nativeName.c_str(), request.kernelName.c_str(), apiName.c_str());
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

    llvm::Error nativeFailure = nativeResult.takeError();
    const bool apiSupports = apiBackend.supports(request);
    if (!apiSupports) {
      return llvm::joinErrors(
          std::move(nativeFailure),
          makeNativeFailureContext(nativeBackend, apiBackend, request,
                                   /*apiSupports=*/false));
    }

    auto apiResult = apiBackend.generate(request);
    if (apiResult)
      return std::move(apiResult);

    llvm::Error apiFailure = apiResult.takeError();
    return llvm::joinErrors(
        std::move(nativeFailure),
        llvm::joinErrors(
            std::move(apiFailure),
            makeNativeFailureContext(nativeBackend, apiBackend, request,
                                     /*apiSupports=*/true)));
  }

  if (!apiBackend.supports(request)) {
    return makeUnsupportedRequestError(request);
  }

  return apiBackend.generate(request);
}

} // namespace mlir::runtime
