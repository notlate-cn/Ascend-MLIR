#include "Runtime/MixCompileMetadata.h"
#include "Runtime/Mix/MixDirectCompileInternal.h"

#include "gtest/gtest.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

#include <string>

using namespace mlir::runtime;

namespace {

MixAbiMetadata makeAbi() {
  MixAbiMetadata abi;
  abi.logicalKernelName = "matmul";
  abi.runtimeKernelName = "matmul_kernel";
  abi.workspaceBytes = 1024;
  abi.blockDim = 4;
  abi.workspaceMode = "fixed";
  abi.tilingMode = "generated_file";
  abi.tilingSource = "out/tiling.bin";
  abi.launcherSymbol = "aclrtlaunch_matmul_kernel";
  abi.aicEntry = "matmul_kernel_0_mix_aic";
  abi.aivEntry = "matmul_kernel_0_mix_aiv";
  abi.inputs = {
      {"a", "matmul_kernel.a.input.bin", "", DType::F16, {16, 64}},
      {"b", "matmul_kernel.b.input.bin", "", DType::F16, {64, 32}},
  };
  abi.outputs = {
      {"out", "matmul_kernel.out.output.bin", "matmul_kernel.out.golden.bin",
       DType::F32, {16, 32}},
  };
  return abi;
}

MixDirectTilingOutputs makeTiling(llvm::StringRef backend,
                                  llvm::StringRef strategy,
                                  llvm::StringRef debugNote = {}) {
  MixDirectTilingOutputs tiling;
  tiling.blockDim = 4;
  tiling.tilingArtifactPath = "out/tiling.bin";
  tiling.launchInfoPath = "out/launch_info.txt";
  tiling.backendKind = backend.str();
  tiling.strategyName = strategy.str();
  tiling.debugNote = debugNote.str();
  tiling.runnerCompileCommand = "tiling-cmd";
  tiling.tilingEmitCommand = "tiling-cmd";
  return tiling;
}

llvm::Expected<MixCompileMetadata>
writeAndParseMetadata(const MixDirectTilingOutputs &tiling) {
  llvm::SmallString<256> path;
  if (auto ec = llvm::sys::fs::createTemporaryFile("mix_direct_metadata", "json",
                                                   path)) {
    return llvm::createStringError(ec, "failed to create temp metadata file");
  }
  auto metadataPathOr = writeMixDirectCompileMetadataFile(
      path, "matmul_kernel", "Ascend910B1", "mix_1c1v",
      "work/generated/auto_gen_matmul.cpp", {}, {}, "out/device.o",
      "out/libkernel.so", tiling, makeAbi());
  if (!metadataPathOr)
    return metadataPathOr.takeError();
  auto bufferOr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOr)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to read generated metadata");
  auto parsedOr = parseMixCompileMetadataJson((*bufferOr)->getBuffer());
  llvm::sys::fs::remove(path);
  return parsedOr;
}

TEST(MixDirectTilingArtifactsTest, WritesRuntimeNativeBackendMetadata) {
  auto metadataOr =
      writeAndParseMetadata(makeTiling("api", "matmul-api", "bias=1"));
  ASSERT_TRUE(static_cast<bool>(metadataOr));
  EXPECT_EQ(metadataOr->hostLaunch.mode, "runtime-native");
  EXPECT_EQ(metadataOr->hostLaunch.helperKind, "in-process-mix-tiling");
  EXPECT_NE(metadataOr->hostLaunch.helperInputsJson.find("\"tiling_backend\": \"api\""),
            std::string::npos);
  EXPECT_NE(metadataOr->hostLaunch.helperInputsJson.find("\"tiling_strategy\": \"matmul-api\""),
            std::string::npos);
  EXPECT_NE(metadataOr->hostLaunch.helperInputsJson.find("\"tiling_debug_note\": \"bias=1\""),
            std::string::npos);
}

TEST(MixDirectTilingArtifactsTest, WritesHelperFallbackMetadataTruthfully) {
  auto metadataOr =
      writeAndParseMetadata(makeTiling("helper", "mix-tiling-helper"));
  ASSERT_TRUE(static_cast<bool>(metadataOr));
  EXPECT_EQ(metadataOr->hostLaunch.mode, "helper");
  EXPECT_EQ(metadataOr->hostLaunch.helperKind, "mix-tiling-helper");
  EXPECT_NE(metadataOr->hostLaunch.helperInputsJson.find("\"tiling_backend\": \"helper\""),
            std::string::npos);
  EXPECT_NE(metadataOr->hostLaunch.helperInputsJson.find("\"tiling_strategy\": \"mix-tiling-helper\""),
            std::string::npos);
}

} // namespace
