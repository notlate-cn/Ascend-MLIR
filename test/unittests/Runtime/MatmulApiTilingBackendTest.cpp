#include "Runtime/Mix/MatmulApiTilingBackend.h"
#include "Runtime/Mix/MixTilingGenerator.h"

#include "gtest/gtest.h"

using namespace mlir::runtime;

namespace {

MatmulTilingRequest makeSupportedRequest() {
  MatmulTilingRequest request;
  request.kernelName = "matmul_bias_relu";
  request.problem.M = 16;
  request.problem.N = 32;
  request.problem.K = 64;
  request.problem.dtypeA = DType::F16;
  request.problem.dtypeB = DType::BF16;
  request.problem.dtypeC = DType::F32;
  request.problem.biasDType = DType::BF16;
  request.problem.transA = true;
  request.problem.transB = false;
  request.problem.hasBias = true;
  request.hints.socVersion = "Ascend910B1";
  request.hints.preferTraverse = MatrixTraverseKind::FirstN;
  return request;
}

MixTilingRequest makeMixRequestWithBias() {
  MixTilingRequest request;
  request.kernelName = "matmul_bias_relu";
  request.socVersion = "Ascend910B1";
  request.inputs = {
      {DType::F16, {16, 64}},
      {DType::BF16, {64, 32}},
      {DType::BF16, {32}},
  };
  request.outputs = {
      {DType::F32, {16, 32}},
  };
  return request;
}

} // namespace

TEST(MatmulApiTilingBackendTest, SupportsSimpleSubset) {
  MatmulApiTilingBackend backend;
  MatmulTilingRequest request = makeSupportedRequest();

  EXPECT_EQ(backend.name(), "matmul-api");
  EXPECT_TRUE(backend.supports(request));

  request.problem.M = 0;
  EXPECT_FALSE(backend.supports(request));
  request.problem.M = 16;
  request.problem.layoutA = MatmulLayout::NZ;
  EXPECT_FALSE(backend.supports(request));
  request.problem.layoutA = MatmulLayout::ND;
  request.problem.batchShape = {2};
  EXPECT_FALSE(backend.supports(request));
  request.problem.batchShape.clear();
  request.problem.dtypeA = DType::INT8;
  EXPECT_FALSE(backend.supports(request));
}

TEST(MatmulApiTilingBackendTest, GeneratesApiTilingForSimpleRequest) {
  MatmulApiTilingBackend backend;
  MatmulTilingRequest request = makeSupportedRequest();

  auto result = backend.generate(request);

  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result->backendKind, "api");
  EXPECT_EQ(result->strategyName, "matmul-api");
  EXPECT_GT(result->blockDim, 0u);
  EXPECT_FALSE(result->tilingData.empty());
  EXPECT_NE(result->debugNote.find("soc=Ascend910B1"), std::string::npos);
  EXPECT_NE(result->debugNote.find("traverse=FIRSTN"), std::string::npos);
  EXPECT_NE(result->debugNote.find("bias_dtype=BF16"), std::string::npos);
}

TEST(MatmulApiTilingBackendTest, MixConversionPreservesExplicitBiasDType) {
  MixTilingRequest mixRequest = makeMixRequestWithBias();

  MatmulTilingRequest matmulRequest = buildMatmulApiTilingRequest(mixRequest);

  EXPECT_TRUE(matmulRequest.problem.biasDType.has_value());
  EXPECT_EQ(*matmulRequest.problem.biasDType, DType::BF16);
  EXPECT_TRUE(matmulRequest.problem.hasBias);
}
