#include "Runtime/Mix/NativeMatmulPlanner.h"

#include "gtest/gtest.h"

using namespace mlir::runtime;

namespace {

MatmulTilingRequest makeRequest() {
  MatmulTilingRequest request;
  request.kernelName = "matmul_bias_relu";
  request.problem.M = 128;
  request.problem.N = 256;
  request.problem.K = 512;
  request.problem.dtypeA = DType::F16;
  request.problem.dtypeB = DType::BF16;
  request.problem.dtypeC = DType::F32;
  request.problem.hasBias = true;
  request.hints.socVersion = "Ascend910B1";
  return request;
}

} // namespace

TEST(NativeMatmulPlannerTest, BuildsDefaultPlanForWideNProblem) {
  MatmulTilingRequest request = makeRequest();

  ASSERT_TRUE(NativeMatmulPlanner::supports(request));
  auto planOr = NativeMatmulPlanner::buildPlan(request);

  ASSERT_TRUE(static_cast<bool>(planOr));
  EXPECT_EQ(planOr->traverse, MatrixTraverseKind::FirstN);
  EXPECT_EQ(planOr->blockDim, 4u);
  EXPECT_TRUE(planOr->splitKEnabled);
  EXPECT_EQ(planOr->tileM, 64);
  EXPECT_EQ(planOr->tileN, 128);
  EXPECT_EQ(planOr->tileK, 64);
}

TEST(NativeMatmulPlannerTest, HonorsExplicitPlannerHints) {
  MatmulTilingRequest request = makeRequest();
  request.hints.preferTraverse = MatrixTraverseKind::FirstM;
  request.hints.preferBlockDim = 7;
  request.hints.preferSplitK = false;

  auto planOr = NativeMatmulPlanner::buildPlan(request);

  ASSERT_TRUE(static_cast<bool>(planOr));
  EXPECT_EQ(planOr->traverse, MatrixTraverseKind::FirstM);
  EXPECT_EQ(planOr->blockDim, 7u);
  EXPECT_FALSE(planOr->splitKEnabled);
  EXPECT_EQ(planOr->tileK, 512);
}

TEST(NativeMatmulPlannerTest, RejectsUnsupportedBatchRequest) {
  MatmulTilingRequest request = makeRequest();
  request.problem.batchShape = {2};

  EXPECT_FALSE(NativeMatmulPlanner::supports(request));
  auto planOr = NativeMatmulPlanner::buildPlan(request);

  ASSERT_FALSE(static_cast<bool>(planOr));
  EXPECT_NE(llvm::toString(planOr.takeError()).find("does not support kernel"),
            std::string::npos);
}
