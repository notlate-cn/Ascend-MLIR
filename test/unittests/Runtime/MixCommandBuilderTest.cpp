#include "Runtime/MixCommandBuilder.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>

using namespace mlir::runtime;

TEST(MixCommandBuilderTest, CarriesBiasShapeToExternalHelper) {
  auto cmd = buildMixTilingHelperCommand(
      "matmul_bias_relu", "Ascend910B1", {16, 64}, DType::F16, {64, 32},
      DType::BF16, {16, 32}, DType::F32, DType::BF16, {32}, "tiling.bin",
      "launch.txt");

  bool sawBiasShape = false;
  for (size_t i = 0; i + 1 < cmd.size(); ++i) {
    if (cmd[i] == "--bias-shape" && cmd[i + 1] == "32") {
      sawBiasShape = true;
      break;
    }
  }
  EXPECT_TRUE(sawBiasShape);
}
