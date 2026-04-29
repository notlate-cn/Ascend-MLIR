#include "Runtime/Mix/NativeMatmulTilingBackend.h"
#include "Runtime/Mix/MatmulApiTilingBackend.h"

#include "llvm/Support/Error.h"

namespace mlir::runtime {

llvm::StringRef NativeMatmulTilingBackend::name() const { return "native"; }

bool NativeMatmulTilingBackend::supports(
    const MatmulTilingRequest &request) const {
  return NativeMatmulPlanner::supports(request);
}

llvm::Expected<MatmulTilingResult>
NativeMatmulTilingBackend::generate(const MatmulTilingRequest &request) const {
  if (!supports(request)) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "native matmul tiling backend does not support kernel %s",
        request.kernelName.c_str());
  }

  auto planOr = NativeMatmulPlanner::buildPlan(request);
  if (!planOr)
    return planOr.takeError();
  const NativeMatmulPlan &plan = *planOr;
  MatmulTilingRequest plannedRequest =
      NativeMatmulPlanner::applyPlan(request, plan);

  MatmulApiTilingBackend apiBackend;
  auto resultOr = apiBackend.generate(plannedRequest);
  if (!resultOr)
    return resultOr.takeError();

  resultOr->backendKind = "native";
  resultOr->strategyName = "native-matmul";
  resultOr->plannedBlockDim = plan.blockDim;
  resultOr->splitKEnabled = plan.splitKEnabled;
  resultOr->tileM = plan.tileM;
  resultOr->tileN = plan.tileN;
  resultOr->tileK = plan.tileK;
  resultOr->debugNote = "planner=native materializer=api " +
                        NativeMatmulPlanner::describePlanPrefix(plan) + " " +
                        resultOr->debugNote;
  return resultOr;
}

} // namespace mlir::runtime
