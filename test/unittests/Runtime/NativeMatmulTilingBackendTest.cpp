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

} // namespace

TEST(NativeMatmulTilingBackendTest, IsNotSelectedForProductionRequests) {
  NativeMatmulTilingBackend backend;
  MatmulTilingRequest request = makeSupportedRequest();

  EXPECT_EQ(backend.name(), "native");
  EXPECT_FALSE(backend.supports(request));
}

TEST(NativeMatmulTilingBackendTest, RejectsDirectGeneration) {
  NativeMatmulTilingBackend backend;
  MatmulTilingRequest request = makeSupportedRequest();

  auto result = backend.generate(request);

  ASSERT_FALSE(static_cast<bool>(result));
  std::vector<std::string> messages;
  llvm::handleAllErrors(result.takeError(),
                        [&](const llvm::ErrorInfoBase &info) {
                          messages.push_back(info.message());
                        });
  ASSERT_FALSE(messages.empty());
  EXPECT_NE(messages[0].find(
                "native matmul tiling backend is not available"),
            std::string::npos);
}
