#include "Runtime/Mix/MixAbi.h"
#include "Runtime/MixAbiExtractor.h"

#include "gtest/gtest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <fstream>
#include <map>
#include <string>

using namespace mlir::runtime;

namespace {

TEST(MixAbiTest, RoundTripsMatmulSemantics) {
  MixAbiMetadata abi;
  abi.logicalKernelName = "matmul";
  abi.runtimeKernelName = "matmul_kernel";
  abi.workspaceBytes = 1024;
  abi.blockDim = 4;
  abi.workspaceMode = "fixed";
  abi.tilingMode = "generated_file";
  abi.tilingSource = "out/tiling.bin";
  abi.inputs = {
      {"a", "matmul_kernel.a.input.bin", "", DType::F16, {16, 64}},
      {"b", "matmul_kernel.b.input.bin", "", DType::F16, {64, 32}},
      {"bias", "matmul_kernel.bias.input.bin", "", DType::F32, {32}},
  };
  abi.outputs = {
      {"out", "matmul_kernel.out.output.bin", "matmul_kernel.out.golden.bin",
       DType::F32, {16, 32}},
  };
  abi.matmul = MixAbiMatmulDesc{
      "matmul", false, true, true, "ND", "NZ", "ND", "BiasAddLeakyRelu", {}};

  auto textOr = serializeMixAbiManifest(abi);
  ASSERT_TRUE(static_cast<bool>(textOr));

  std::map<std::string, std::string> manifest;
  llvm::SmallVector<llvm::StringRef> lines;
  llvm::StringRef(*textOr).split(lines, '\n', -1, false);
  for (llvm::StringRef line : lines) {
    if (line.empty())
      continue;
    auto parts = line.split('=');
    manifest[parts.first.str()] = parts.second.str();
  }

  auto parsedOr = parseMixAbiManifest(manifest);
  ASSERT_TRUE(static_cast<bool>(parsedOr));
  ASSERT_TRUE(parsedOr->matmul.has_value());
  EXPECT_EQ(parsedOr->matmul->opKind, "matmul");
  EXPECT_FALSE(parsedOr->matmul->transA);
  EXPECT_TRUE(parsedOr->matmul->transB);
  EXPECT_TRUE(parsedOr->matmul->hasBias);
  EXPECT_EQ(parsedOr->matmul->layoutA, "ND");
  EXPECT_EQ(parsedOr->matmul->layoutB, "NZ");
  EXPECT_EQ(parsedOr->matmul->layoutC, "ND");
  EXPECT_EQ(parsedOr->matmul->epilogueKind, "BiasAddLeakyRelu");
}

TEST(MixAbiTest, ExtractorKeepsMatmulSemanticsEmptyWithoutStableMarker) {
  llvm::SmallString<256> path;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("mix_abi_matmul", "mlir", path));
  std::ofstream os(path.str());
  os << R"mlir(
func.func @matmul_add_leakyrelu(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<ui8>, %arg6: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>) attributes {cann.num_inputs = 4 : i32} {
  %0 = memref.load %arg0[%c0, %c0] : memref<?x?xf16>
  %1 = memref.load %arg1[%c0, %c0] : memref<?x?xf16>
  %2 = memref.load %arg2[%c0] : memref<?xf32>
  %3 = memref.load %arg3[%c0, %c0] : memref<?x?xf32>
  %4 = arith.extf %0 : f16 to f32
  %5 = arith.extf %1 : f16 to f32
  %6 = arith.addf %4, %5 : f32
  %7 = arith.addf %6, %2 : f32
  memref.store %7, %arg4[%c0, %c0] : memref<?x?xf32>
  return
}
)mlir";
  os.close();

  auto abiOr = extractMixAbiFromCannMlir(path);
  ASSERT_TRUE(static_cast<bool>(abiOr));
  EXPECT_FALSE(abiOr->matmul.has_value());

  llvm::sys::fs::remove(path);
}

} // namespace
