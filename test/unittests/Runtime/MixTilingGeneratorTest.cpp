#include "Runtime/Mix/MixTilingGenerator.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <string>
#include <vector>

using namespace mlir::runtime;

namespace mlir::runtime {
#if defined(MLIR_RUNTIME_ENABLE_MATMUL_MIX_ADAPTER_TEST_HOOKS)
llvm::Expected<MatmulTilingRequest>
buildMatmulApiTilingRequestForTest(const MixTilingRequest &request);
#endif
} // namespace mlir::runtime

namespace {

MixTilingRequest makeRequestWithUnsupportedBiasDType() {
  MixTilingRequest request;
  request.kernelName = "matmul_bias_relu";
  request.socVersion = "Ascend910B1";
  request.inputs = {
      {DType::F16, {16, 64}},
      {DType::BF16, {64, 32}},
      {DType::INT8, {32}},
  };
  request.outputs = {
      {DType::F32, {16, 32}},
  };
  return request;
}

} // namespace

TEST(MixTilingGeneratorTest, RejectsUnsupportedBiasDTypeThroughAdapter) {
  auto result = generateMixTilingInProcess(makeRequestWithUnsupportedBiasDType());

  ASSERT_FALSE(static_cast<bool>(result));
  std::vector<std::string> messages;
  llvm::handleAllErrors(result.takeError(),
                        [&](const llvm::ErrorInfoBase &info) {
                          messages.push_back(info.message());
                        });
  ASSERT_FALSE(messages.empty());
  EXPECT_NE(messages[0].find("unsupported matmul api tiling request"),
            std::string::npos);
}

TEST(MixTilingGeneratorTest, PreservesSupportedExplicitBiasDTypeThroughAdapter) {
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

#if defined(MLIR_RUNTIME_ENABLE_MATMUL_MIX_ADAPTER_TEST_HOOKS)
  auto matmulRequestOr = buildMatmulApiTilingRequestForTest(request);
  ASSERT_TRUE(static_cast<bool>(matmulRequestOr));
  EXPECT_TRUE(matmulRequestOr->problem.biasDType.has_value());
  EXPECT_EQ(*matmulRequestOr->problem.biasDType, DType::BF16);
#endif

  auto result = generateMixTilingInProcess(request);
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_GT(result->blockDim, 0u);
  EXPECT_FALSE(result->tilingData.empty());
}
