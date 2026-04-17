#include "Runtime/Mix/MatmulApiTilingBackend.h"
#include "MatmulApiTilingBackendTestHooks.h"

#include "gtest/gtest.h"

#include <limits>

using namespace mlir::runtime;

namespace {

int failGetTiling(matmul_tiling::MatmulApiTiling &,
                  optiling::TCubeTiling &) {
  return -1;
}

class ScopedMatmulApiTilingGetTilingHook {
public:
  explicit ScopedMatmulApiTilingGetTilingHook(MatmulApiTilingGetTilingHook hook) {
    setMatmulApiTilingGetTilingForTest(hook);
  }

  ~ScopedMatmulApiTilingGetTilingHook() {
    setMatmulApiTilingGetTilingForTest(nullptr);
  }
};

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

MatmulTilingRequest makeSupportedBatchRequest() {
  MatmulTilingRequest request;
  request.kernelName = "batch_matmul";
  request.problem.M = 16;
  request.problem.N = 32;
  request.problem.K = 64;
  request.problem.batchShape = {2};
  request.problem.dtypeA = DType::F16;
  request.problem.dtypeB = DType::BF16;
  request.problem.dtypeC = DType::F32;
  request.hints.socVersion = "Ascend910B1";
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
  request.problem.dtypeA = DType::INT8;
  EXPECT_FALSE(backend.supports(request));
}

TEST(MatmulApiTilingBackendTest, SupportsConservativeBatchSubset) {
  MatmulApiTilingBackend backend;
  MatmulTilingRequest request = makeSupportedBatchRequest();

  EXPECT_TRUE(backend.supports(request));

  request.problem.hasBias = true;
  EXPECT_FALSE(backend.supports(request));
  request.problem.hasBias = false;
  request.fusion.epilogue = EpilogueKind::BiasAdd;
  EXPECT_FALSE(backend.supports(request));
  request.fusion.epilogue = EpilogueKind::None;
  request.problem.batchShape = {2, 3};
  EXPECT_FALSE(backend.supports(request));
}

TEST(MatmulApiTilingBackendTest, GeneratesApiTilingForSimpleRequest) {
  MatmulApiTilingBackend backend;
  MatmulTilingRequest request = makeSupportedRequest();
  request.problem.K = 128;
  request.hints.preferTileM = 16;
  request.hints.preferTileN = 32;
  request.hints.preferTileK = 32;
  request.hints.preferBlockDim = 7;
  request.hints.preferSplitK = true;

  auto result = backend.generate(request);

  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result->backendKind, "api");
  EXPECT_EQ(result->strategyName, "matmul-api");
  EXPECT_GT(result->blockDim, 0u);
  EXPECT_FALSE(result->tilingData.empty());
  ASSERT_TRUE(result->plannedBlockDim.has_value());
  EXPECT_EQ(*result->plannedBlockDim, 7u);
  ASSERT_TRUE(result->splitKEnabled.has_value());
  EXPECT_TRUE(*result->splitKEnabled);
  EXPECT_NE(result->debugNote.find("soc=Ascend910B1"), std::string::npos);
  EXPECT_NE(result->debugNote.find("traverse=FIRSTN"), std::string::npos);
  EXPECT_NE(result->debugNote.find("bias_dtype=BF16"), std::string::npos);
  EXPECT_NE(result->debugNote.find("requested_block_dim=7"), std::string::npos);
  EXPECT_NE(result->debugNote.find("split_k=1"), std::string::npos);
  EXPECT_NE(result->debugNote.find("fix_split=16x32x32"), std::string::npos);
}

TEST(MatmulApiTilingBackendTest, GeneratesApiTilingForBatchRequest) {
  MatmulApiTilingBackend backend;
  MatmulTilingRequest request = makeSupportedBatchRequest();

  auto result = backend.generate(request);

  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result->backendKind, "api");
  EXPECT_EQ(result->strategyName, "matmul-api");
  EXPECT_GT(result->blockDim, 0u);
  EXPECT_FALSE(result->tilingData.empty());
  EXPECT_NE(result->debugNote.find("batch=2"), std::string::npos);
}

TEST(MatmulApiTilingBackendTest, DisablesSplitKWhenExplicitlyRequested) {
  MatmulApiTilingBackend backend;
  MatmulTilingRequest request = makeSupportedRequest();
  request.problem.K = 128;
  request.hints.preferTileM = 16;
  request.hints.preferTileN = 32;
  request.hints.preferTileK = 32;
  request.hints.preferSplitK = false;

  auto result = backend.generate(request);

  ASSERT_TRUE(static_cast<bool>(result));
  ASSERT_TRUE(result->splitKEnabled.has_value());
  EXPECT_FALSE(*result->splitKEnabled);
  EXPECT_FALSE(result->plannedBlockDim.has_value());
  EXPECT_NE(result->debugNote.find("requested_block_dim=none"),
            std::string::npos);
  EXPECT_NE(result->debugNote.find("split_k=0"), std::string::npos);
  EXPECT_NE(result->debugNote.find("fix_split=16x32x-1"), std::string::npos);
}

TEST(MatmulApiTilingBackendTest, RejectsUnsupportedExplicitBiasDType) {
  MatmulApiTilingBackend backend;
  MatmulTilingRequest request = makeSupportedRequest();
  request.problem.biasDType = DType::INT8;

  EXPECT_FALSE(backend.supports(request));
  auto result = backend.generate(request);
  ASSERT_FALSE(static_cast<bool>(result));
  std::string errorText = llvm::toString(result.takeError());
  EXPECT_NE(errorText.find("unsupported matmul api tiling request"),
            std::string::npos);
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
  auto mResult = backend.generate(request);
  EXPECT_FALSE(static_cast<bool>(mResult));
  (void)llvm::toString(mResult.takeError());

  request.problem.M = 16;
  request.problem.N = static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
  EXPECT_FALSE(backend.supports(request));
  auto nResult = backend.generate(request);
  EXPECT_FALSE(static_cast<bool>(nResult));
  (void)llvm::toString(nResult.takeError());

  request.problem.N = 32;
  request.problem.K = static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
  EXPECT_FALSE(backend.supports(request));
  auto kResult = backend.generate(request);
  EXPECT_FALSE(static_cast<bool>(kResult));
  (void)llvm::toString(kResult.takeError());
}

TEST(MatmulApiTilingBackendTest, ReturnsErrorWhenVendorGetTilingFails) {
  ScopedMatmulApiTilingGetTilingHook hook(&failGetTiling);
  MatmulApiTilingBackend backend;
  MatmulTilingRequest request = makeSupportedRequest();

  auto result = backend.generate(request);
  ASSERT_FALSE(static_cast<bool>(result));
  std::vector<std::string> messages;
  llvm::handleAllErrors(result.takeError(),
                        [&](const llvm::ErrorInfoBase &info) {
                          messages.push_back(info.message());
                        });
  ASSERT_FALSE(messages.empty());
  EXPECT_NE(messages[0].find("matmul api tiling failed"),
            std::string::npos);
}
