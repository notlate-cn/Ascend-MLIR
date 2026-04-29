#include "Runtime/Mix/NativeMatmulTilingBackend.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <string>
#include <vector>

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
  request.problem.hasBias = true;
  request.hints.socVersion = "Ascend910B1";
  return request;
}

static size_t countSubstring(std::string_view text, std::string_view needle) {
  size_t count = 0;
  size_t offset = 0;
  while ((offset = text.find(needle, offset)) != std::string_view::npos) {
    ++count;
    offset += needle.size();
  }
  return count;
}

} // namespace

TEST(NativeMatmulTilingBackendTest, IsNotSelectedForProductionRequests) {
  NativeMatmulTilingBackend backend;
  MatmulTilingRequest request = makeSupportedRequest();

  EXPECT_EQ(backend.name(), "native");
  EXPECT_TRUE(backend.supports(request));

  request.problem.batchShape = {2};
  EXPECT_FALSE(backend.supports(request));
}

TEST(NativeMatmulTilingBackendTest, GeneratesNativePlannedTiling) {
  NativeMatmulTilingBackend backend;
  MatmulTilingRequest request = makeSupportedRequest();
  request.problem.transB = true;

  auto result = backend.generate(request);

  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result->backendKind, "native");
  EXPECT_EQ(result->strategyName, "native-matmul");
  EXPECT_GT(result->blockDim, 0u);
  EXPECT_FALSE(result->tilingData.empty());
  ASSERT_TRUE(result->plannedBlockDim.has_value());
  EXPECT_EQ(*result->plannedBlockDim, 1u);
  ASSERT_TRUE(result->splitKEnabled.has_value());
  EXPECT_TRUE(*result->splitKEnabled);
  EXPECT_TRUE(result->tileM.has_value());
  EXPECT_TRUE(result->tileN.has_value());
  EXPECT_TRUE(result->tileK.has_value());
  EXPECT_EQ(*result->tileM, 16);
  EXPECT_EQ(*result->tileN, 32);
  EXPECT_EQ(*result->tileK, 32);
  EXPECT_NE(result->debugNote.find("planner=native"), std::string::npos);
  EXPECT_NE(result->debugNote.find("materializer=api"), std::string::npos);
  EXPECT_NE(result->debugNote.find("planned_block_dim=1"), std::string::npos);
  EXPECT_NE(result->debugNote.find("split_k=1"), std::string::npos);
  EXPECT_EQ(countSubstring(result->debugNote, "split_k="), 1u);
  EXPECT_NE(result->debugNote.find("fix_split=16x32x32"), std::string::npos);
  EXPECT_EQ(countSubstring(result->debugNote, "fix_split="), 1u);
}
