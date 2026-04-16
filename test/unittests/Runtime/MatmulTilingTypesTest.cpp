#include "Runtime/Mix/MatmulTilingTypes.h"
#include "gtest/gtest.h"

using namespace mlir::runtime;

TEST(MatmulTilingTypesTest, DefaultRequestIsNativeFriendly) {
  MatmulTilingRequest request;

  EXPECT_TRUE(request.kernelName.empty());
  EXPECT_EQ(request.problem.layoutA, MatmulLayout::ND);
  EXPECT_EQ(request.problem.layoutB, MatmulLayout::ND);
  EXPECT_EQ(request.problem.layoutC, MatmulLayout::ND);
  EXPECT_EQ(request.fusion.epilogue, EpilogueKind::None);
  EXPECT_FALSE(request.fusion.preferFuseVectorEpilogue);
  EXPECT_EQ(request.fusion.consumerAlignmentBytes, 0u);
  EXPECT_FALSE(request.hints.preferTraverse.has_value());

  EXPECT_TRUE(request.problem.batchShape.empty());
  EXPECT_FALSE(request.problem.transA);
  EXPECT_FALSE(request.problem.transB);
  EXPECT_FALSE(request.problem.hasBias);
  EXPECT_EQ(request.problem.M, 0);
  EXPECT_EQ(request.problem.N, 0);
  EXPECT_EQ(request.problem.K, 0);
  EXPECT_EQ(request.problem.dtypeA, DType::F32);
  EXPECT_EQ(request.problem.dtypeB, DType::F32);
  EXPECT_EQ(request.problem.dtypeC, DType::F32);
  EXPECT_FALSE(request.problem.biasDType.has_value());
  EXPECT_TRUE(request.hints.socVersion.empty());
  EXPECT_FALSE(request.hints.preferBlockDim.has_value());
  EXPECT_FALSE(request.hints.preferSplitK.has_value());
  EXPECT_FALSE(request.hints.preferTileM.has_value());
  EXPECT_FALSE(request.hints.preferTileN.has_value());
  EXPECT_FALSE(request.hints.preferTileK.has_value());

  MatmulTilingResult result;
  EXPECT_EQ(result.blockDim, 0u);
  EXPECT_TRUE(result.backendKind.empty());
  EXPECT_TRUE(result.strategyName.empty());
  EXPECT_TRUE(result.tilingData.empty());
  EXPECT_FALSE(result.tileM.has_value());
  EXPECT_FALSE(result.tileN.has_value());
  EXPECT_FALSE(result.tileK.has_value());
  EXPECT_TRUE(result.debugNote.empty());
}
