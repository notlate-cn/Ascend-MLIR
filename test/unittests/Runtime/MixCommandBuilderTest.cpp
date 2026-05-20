#include "Runtime/MixCommandBuilder.h"

#include "gtest/gtest.h"

#include <algorithm>
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

TEST(MixCommandBuilderTest, HostSharedLinkAvoidsSimulatorOnlyLibraries) {
  auto cmd = buildHostSharedLinkCommand("host_stub.o", "kernel.so",
                                        "Ascend910B1", "/tmp/device-lib");

  EXPECT_EQ(std::find(cmd.begin(), cmd.end(), "-lruntime_camodel"), cmd.end());
  EXPECT_EQ(std::find(cmd.begin(), cmd.end(), "-lnpu_drv"), cmd.end());
  EXPECT_EQ(std::find(cmd.begin(), cmd.end(), "-lstars"), cmd.end());
  EXPECT_EQ(std::find(cmd.begin(), cmd.end(), "-lmodel_top"), cmd.end());
  EXPECT_FALSE(std::any_of(cmd.begin(), cmd.end(), [](const std::string &arg) {
    return arg.find("/simulator/") != std::string::npos;
  }));
}
