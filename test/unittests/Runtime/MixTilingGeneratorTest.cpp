#include "Runtime/Mix/MixTilingGenerator.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <string>
#include <vector>

using namespace mlir::runtime;

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

MixTilingRequest makeRequestWithMalformedBiasShape() {
  MixTilingRequest request = makeRequestWithUnsupportedBiasDType();
  request.inputs[2].dtype = DType::BF16;
  request.inputs[2].shape = {1, 32};
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

TEST(MixTilingGeneratorTest, RejectsMalformedBiasShapeThroughAdapter) {
  auto result = generateMixTilingInProcess(makeRequestWithMalformedBiasShape());

  ASSERT_FALSE(static_cast<bool>(result));
  std::vector<std::string> messages;
  llvm::handleAllErrors(result.takeError(),
                        [&](const llvm::ErrorInfoBase &info) {
                          messages.push_back(info.message());
                        });
  ASSERT_FALSE(messages.empty());
  EXPECT_NE(messages[0].find("bias must be a 1D vector with length N"),
            std::string::npos);
}
