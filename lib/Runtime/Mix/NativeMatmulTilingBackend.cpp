#include "Runtime/Mix/NativeMatmulTilingBackend.h"
#include "Runtime/Mix/MatmulApiTilingBackend.h"

#include "llvm/Support/Error.h"

namespace mlir::runtime {
namespace {

static bool supportsNativePlannerRequest(const MatmulTilingRequest &request) {
  return request.problem.M > 0 && request.problem.N > 0 &&
         request.problem.K > 0 && request.problem.batchShape.empty() &&
         request.problem.layoutA == MatmulLayout::ND &&
         request.problem.layoutB == MatmulLayout::ND &&
         request.problem.layoutC == MatmulLayout::ND;
}

static std::optional<MatrixTraverseKind>
selectNativeTraverse(const MatmulTilingRequest &request) {
  if (request.hints.preferTraverse.has_value())
    return request.hints.preferTraverse;
  if (request.problem.N >= request.problem.M)
    return MatrixTraverseKind::FirstN;
  return MatrixTraverseKind::FirstM;
}

} // namespace

llvm::StringRef NativeMatmulTilingBackend::name() const { return "native"; }

bool NativeMatmulTilingBackend::supports(
    const MatmulTilingRequest &request) const {
  return supportsNativePlannerRequest(request);
}

llvm::Expected<MatmulTilingResult>
NativeMatmulTilingBackend::generate(const MatmulTilingRequest &request) const {
  if (!supports(request)) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "native matmul tiling backend does not support kernel %s",
        request.kernelName.c_str());
  }

  MatmulTilingRequest plannedRequest = request;
  plannedRequest.hints.preferTraverse = selectNativeTraverse(request);

  MatmulApiTilingBackend apiBackend;
  auto resultOr = apiBackend.generate(plannedRequest);
  if (!resultOr)
    return resultOr.takeError();

  resultOr->backendKind = "native";
  resultOr->strategyName = "native-matmul";
  resultOr->debugNote =
      "planner=native materializer=api " + resultOr->debugNote;
  return resultOr;
}

} // namespace mlir::runtime
