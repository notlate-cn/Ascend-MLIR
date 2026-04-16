#include "Runtime/Mix/MatmulApiTilingBackend.h"

#include "gtest/gtest.h"

#include <limits>

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

TEST(MatmulApiTilingBackendTest, RejectsUnsupportedExplicitBiasDType) {
  MatmulApiTilingBackend backend;
  MatmulTilingRequest request = makeSupportedRequest();
  request.problem.biasDType = DType::INT8;

  EXPECT_FALSE(backend.supports(request));
  auto result = backend.generate(request);
  ASSERT_FALSE(static_cast<bool>(result));
}

TEST(MatmulApiTilingBackendTest, IgnoresBiasDTypeWhenBiasIsDisabled) {
  MatmulApiTilingBackend backend;
  MatmulTilingRequest request = makeSupportedRequest();
  request.problem.hasBias = false;
  request.problem.biasDType = DType::INT8;

  EXPECT_TRUE(backend.supports(request));
  auto result = backend.generate(request);

  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result->backendKind, "api");
  EXPECT_NE(result->debugNote.find("bias=0"), std::string::npos);
  EXPECT_NE(result->debugNote.find("bias_dtype=none"), std::string::npos);
}

TEST(MatmulApiTilingBackendTest, RejectsOutOfRangeShapeForVendorApi) {
  MatmulApiTilingBackend backend;
  MatmulTilingRequest request = makeSupportedRequest();
  request.problem.M = static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
  EXPECT_FALSE(backend.supports(request));
  EXPECT_FALSE(static_cast<bool>(backend.generate(request)));

  request.problem.M = 16;
  request.problem.N = static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
  EXPECT_FALSE(backend.supports(request));

  request.problem.N = 32;
  request.problem.K = static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
  EXPECT_FALSE(backend.supports(request));
}
