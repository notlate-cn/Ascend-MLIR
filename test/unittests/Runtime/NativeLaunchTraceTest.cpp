#include "LaunchTrace.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace mlir::runtime;

namespace {

TEST(NativeLaunchTraceTest, RendersLaunchMetadataAndAlignment) {
  std::vector<LaunchTraceTensor> inputs = {
      {1280, 0x1000},
      {640000, 0x1200},
  };
  std::vector<LaunchTraceTensor> outputs = {
      {640000, 0x1408},
  };
  std::vector<uint8_t> tiling = {
      64, 0, 0, 0, 0, 0, 0, 0,
      64, 0, 0, 0, 0, 0, 0, 0,
      128, 2, 0, 0, 0, 0, 0, 0,
  };
  std::vector<uint64_t> launchArgs = {
      0x1000, 0x1200, 0x1408, 0x1600, 64, 64, 640,
  };

  LaunchTraceRecord record;
  record.kernelName = "relu_transpose_broadcast_add";
  record.binaryPath = "/tmp/kernel.o";
  record.magic = 0x41415246u;
  record.deviceId = 7;
  record.blockDim = 8;
  record.inputs = inputs;
  record.outputs = outputs;
  record.workspaceSize = 16777216;
  record.workspacePtr = 0x1600;
  record.tilingBytes = tiling;
  record.launchArgs = launchArgs;
  record.argsSize = static_cast<uint32_t>(launchArgs.size() * sizeof(uint64_t));

  std::string text;
  llvm::raw_string_ostream os(text);
  printNativeLaunchTrace(record, os);
  os.flush();

  EXPECT_NE(text.find("[npu-launch] kernel=relu_transpose_broadcast_add"),
            std::string::npos);
  EXPECT_NE(text.find("binary=/tmp/kernel.o"), std::string::npos);
  EXPECT_NE(text.find("magic=0x41415246"), std::string::npos);
  EXPECT_NE(text.find("device_id=7"), std::string::npos);
  EXPECT_NE(text.find("block_dim=8"), std::string::npos);
  EXPECT_NE(text.find("args_count=7 args_size=56"), std::string::npos);
  EXPECT_NE(text.find("input[0] bytes=1280 ptr=0x1000 align512=yes"),
            std::string::npos);
  EXPECT_NE(text.find("output[0] bytes=640000 ptr=0x1408 align512=no"),
            std::string::npos);
  EXPECT_NE(text.find("workspace bytes=16777216 ptr=0x1600 align512=yes"),
            std::string::npos);
  EXPECT_NE(text.find("tiling_bytes=24 tiling_words=3"), std::string::npos);
  EXPECT_NE(text.find("tiling.word[0]=64"), std::string::npos);
  EXPECT_NE(text.find("tiling.word[1]=64"), std::string::npos);
  EXPECT_NE(text.find("tiling.word[2]=640"), std::string::npos);
}

TEST(NativeLaunchTraceTest, RendersBufferSamples) {
  std::vector<uint8_t> bytes = {0x00, 0x01, 0xab, 0xff};

  std::string text;
  llvm::raw_string_ostream os(text);
  printNativeLaunchBufferSample("input[0].h2d_roundtrip", bytes,
                                /*totalBytes=*/1280, os);
  os.flush();

  EXPECT_NE(text.find("[npu-launch] input[0].h2d_roundtrip bytes=1280"),
            std::string::npos);
  EXPECT_NE(text.find("sample_bytes=4"), std::string::npos);
  EXPECT_NE(text.find("sample_hex=0001abff"), std::string::npos);
}

} // namespace
