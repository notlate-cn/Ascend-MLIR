#include "Runtime/Mix/NativeMatmulPlanner.h"

#include "llvm/Support/Error.h"

#include <algorithm>

namespace mlir::runtime {
namespace {

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

static int64_t ceilDiv(int64_t lhs, int64_t rhs) {
  return (lhs + rhs - 1) / rhs;
}

static int64_t pickKTile(const MatmulTilingRequest &request) {
  const int64_t granularity =
      (request.problem.dtypeA == DType::F32 ||
       request.problem.dtypeB == DType::F32)
          ? 8
          : 16;
  if (request.problem.K <= granularity)
    return request.problem.K;
  int64_t candidate = (request.problem.K >= 256) ? 64 : 32;
  candidate = std::min(candidate, request.problem.K);
  candidate =
      std::max<int64_t>(granularity, candidate - (candidate % granularity));
  if (candidate <= 0)
    candidate = granularity;
  return std::min(candidate, request.problem.K);
}

} // namespace

bool NativeMatmulPlanner::supports(const MatmulTilingRequest &request) {
  return request.problem.M > 0 && request.problem.N > 0 &&
         request.problem.K > 0 && request.problem.batchShape.empty() &&
         request.problem.layoutA == MatmulLayout::ND &&
         request.problem.layoutB == MatmulLayout::ND &&
         request.problem.layoutC == MatmulLayout::ND;
}

llvm::Expected<NativeMatmulPlan>
NativeMatmulPlanner::buildPlan(const MatmulTilingRequest &request) {
  if (!supports(request)) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "native matmul planner does not support kernel %s",
        request.kernelName.c_str());
  }

  NativeMatmulPlan plan;
  plan.traverse = *selectNativeTraverse(request);
  plan.tileK = pickKTile(request);

  const bool wideN = request.problem.N >= request.problem.M;
  const int64_t targetM = wideN ? 64 : 128;
  const int64_t targetN = wideN ? 128 : 64;
  plan.tileM =
      clampTile(std::min(request.problem.M, targetM), 16, request.problem.M);
  plan.tileN =
      clampTile(std::min(request.problem.N, targetN), 16, request.problem.N);
  plan.splitKEnabled =
      request.hints.preferSplitK.value_or(request.problem.K > plan.tileK);
  if (!plan.splitKEnabled)
    plan.tileK = request.problem.K;
  if (request.hints.preferBlockDim.has_value() &&
      *request.hints.preferBlockDim > 0) {
    plan.blockDim = static_cast<uint32_t>(*request.hints.preferBlockDim);
  } else {
    const int64_t mTiles = ceilDiv(request.problem.M, plan.tileM);
    const int64_t nTiles = ceilDiv(request.problem.N, plan.tileN);
    plan.blockDim =
        static_cast<uint32_t>(std::max<int64_t>(1, mTiles * nTiles));
  }

  return plan;
}

MatmulTilingRequest
NativeMatmulPlanner::applyPlan(const MatmulTilingRequest &request,
                               const NativeMatmulPlan &plan) {
  MatmulTilingRequest plannedRequest = request;
  plannedRequest.hints.preferBlockDim = plan.blockDim;
  plannedRequest.hints.preferSplitK = plan.splitKEnabled;
  plannedRequest.hints.preferTraverse = plan.traverse;
  plannedRequest.hints.preferTileM = plan.tileM;
  plannedRequest.hints.preferTileN = plan.tileN;
  plannedRequest.hints.preferTileK = plan.tileK;
  return plannedRequest;
}

std::string NativeMatmulPlanner::describePlan(const NativeMatmulPlan &plan) {
  const char *traverse =
      plan.traverse == MatrixTraverseKind::FirstN ? "FIRSTN" : "FIRSTM";
  return "planned_block_dim=" + std::to_string(plan.blockDim) +
         " traverse=" + traverse + " split_k=" +
         std::string(plan.splitKEnabled ? "1" : "0") + " fix_split=" +
         std::to_string(plan.tileM) + "x" + std::to_string(plan.tileN) + "x" +
         std::to_string(plan.tileK);
}

std::string NativeMatmulPlanner::describePlanPrefix(
    const NativeMatmulPlan &plan) {
  return "planned_block_dim=" + std::to_string(plan.blockDim);
}

} // namespace mlir::runtime
