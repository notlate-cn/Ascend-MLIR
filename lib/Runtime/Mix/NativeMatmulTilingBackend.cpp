#include "Runtime/Mix/NativeMatmulTilingBackend.h"
#include "Runtime/Mix/MatmulApiTilingBackend.h"

#include "llvm/Support/Error.h"

#include <algorithm>

namespace mlir::runtime {
namespace {

struct NativeMatmulPlan {
  MatrixTraverseKind traverse = MatrixTraverseKind::FirstM;
  int64_t tileM = 0;
  int64_t tileN = 0;
  int64_t tileK = 0;
};

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

static int64_t clampTile(int64_t value, int64_t floor, int64_t ceil) {
  if (ceil < floor)
    return floor;
  return std::max(floor, std::min(value, ceil));
}

static int64_t pickKTile(const MatmulTilingRequest &request) {
  const int64_t granularity =
      (request.problem.dtypeA == DType::F32 || request.problem.dtypeB == DType::F32)
          ? 8
          : 16;
  if (request.problem.K <= granularity)
    return request.problem.K;
  int64_t candidate = (request.problem.K >= 256) ? 64 : 32;
  candidate = std::min(candidate, request.problem.K);
  candidate = std::max<int64_t>(granularity, candidate - (candidate % granularity));
  if (candidate <= 0)
    candidate = granularity;
  return std::min(candidate, request.problem.K);
}

static NativeMatmulPlan buildNativePlan(const MatmulTilingRequest &request) {
  NativeMatmulPlan plan;
  plan.traverse = *selectNativeTraverse(request);
  plan.tileK = pickKTile(request);

  const bool wideN = request.problem.N >= request.problem.M;
  const int64_t targetM = wideN ? 64 : 128;
  const int64_t targetN = wideN ? 128 : 64;
  plan.tileM = clampTile(std::min(request.problem.M, targetM), 16, request.problem.M);
  plan.tileN = clampTile(std::min(request.problem.N, targetN), 16, request.problem.N);
  return plan;
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

  const NativeMatmulPlan plan = buildNativePlan(request);
  MatmulTilingRequest plannedRequest = request;
  plannedRequest.hints.preferTraverse = plan.traverse;
  plannedRequest.hints.preferTileM = plan.tileM;
  plannedRequest.hints.preferTileN = plan.tileN;
  plannedRequest.hints.preferTileK = plan.tileK;

  MatmulApiTilingBackend apiBackend;
  auto resultOr = apiBackend.generate(plannedRequest);
  if (!resultOr)
    return resultOr.takeError();

  resultOr->backendKind = "native";
  resultOr->strategyName = "native-matmul";
  resultOr->tileM = plan.tileM;
  resultOr->tileN = plan.tileN;
  resultOr->tileK = plan.tileK;
  resultOr->debugNote =
      "planner=native materializer=api " + resultOr->debugNote;
  return resultOr;
}

} // namespace mlir::runtime
