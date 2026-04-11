// test/tools/runtime/test_capi_runtime.cpp
//
// Focused smoke test for the C API runtime shim backed by the unified runtime.
//
// Build (on xvm):
//   cd /home/niu/code/Codex-Ascend-MLIR
//   g++ -std=c++17 -I include -I /home/niu/code/llvm-project/build/include \
//       -I /home/niu/code/llvm-project/llvm/include \
//       test/tools/runtime/test_capi_runtime.cpp \
//       build/lib/libAFIRRuntimeCAPI.so \
//       build/lib/libAscendCRuntime.a \
//       $(/home/niu/code/llvm-project/build/bin/llvm-config --ldflags --libs support) \
//       -ldl -o /tmp/test_capi_runtime
//   LD_LIBRARY_PATH=build/lib:/home/niu/code/llvm-project/build/lib /tmp/test_capi_runtime

#include "CAPI/Runtime.h"
#include "Runtime/NpyIO.h"
#include "Runtime/TilingSchema.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

using namespace mlir::runtime;

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT(cond, msg)                                                     \
  do {                                                                        \
    if (cond) {                                                               \
      llvm::outs() << "  PASS: " << (msg) << "\n";                           \
      ++g_pass;                                                               \
    } else {                                                                  \
      llvm::errs() << "  FAIL: " << (msg) << "\n";                           \
      ++g_fail;                                                               \
    }                                                                         \
  } while (0)

static std::vector<uint8_t> readBinaryFile(const std::string &path) {
  std::ifstream is(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(is)),
                              std::istreambuf_iterator<char>());
}

static llvm::Expected<std::vector<uint8_t>>
packExampleTiling(const std::string &schemaPath) {
  auto schemaOr = TilingSchema::fromJson(schemaPath);
  if (!schemaOr)
    return schemaOr.takeError();

  const std::string params =
      "TB_M=64,TB_N=64,dim_arg0_0=640,dim_arg1_0=500,"
      "dim_arg0_1=1,dim_arg1_1=640";

  std::map<std::string, int64_t> values;
  std::istringstream stream(params);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const size_t eq = token.find('=');
    if (eq == std::string::npos)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "invalid tiling token: %s",
                                     token.c_str());
    values[token.substr(0, eq)] = std::stoll(token.substr(eq + 1));
  }

  std::vector<std::pair<std::string, int64_t>> ordered;
  for (const auto &field : schemaOr->fields()) {
    auto it = values.find(field.name);
    if (it == values.end()) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "missing tiling field: %s",
                                     field.name.c_str());
    }
    ordered.push_back({field.name, it->second});
  }

  return schemaOr->pack(ordered);
}

int main() {
  const std::string exampleDir = "examples/relu-broadcast-transpose";
  const std::string kernelPath = exampleDir + "/step8_kernel.cpp";
  const std::string input0Path = exampleDir + "/input_data0.npy";
  const std::string input1Path = exampleDir + "/input_data1.npy";
  const std::string expectedPath = exampleDir + "/output_expected.npy";
  const std::string tilingSchemaPath = exampleDir + "/tiling_space.json";

  const std::filesystem::path outputDir =
      std::filesystem::temp_directory_path() / "afirt-capi-smoke-out";
  std::error_code ec;
  std::filesystem::remove_all(outputDir, ec);
  std::filesystem::create_directories(outputDir, ec);
  EXPECT(!ec, "prepare temp output directory");
  if (ec) {
    llvm::outs() << g_pass << " passed, " << g_fail << " failed\n";
    return 1;
  }

  AfirtCompiler compiler =
      afirt_compiler_create("Ascend910B1", "dav-c220-vec", 3);
  EXPECT(compiler != nullptr, "create compiler handle");
  if (!compiler)
    return 1;

  char binPath[1024] = {};
  int compileRc = afirt_compiler_compile(compiler, kernelPath.c_str(),
                                         outputDir.string().c_str(),
                                         "relu_transpose_broadcast_add",
                                         binPath, sizeof(binPath));
  EXPECT(compileRc == 0, "compile example kernel through C API");
  EXPECT(std::filesystem::exists(binPath),
         "compiled binary path exists");
  if (compileRc != 0) {
    afirt_compiler_destroy(compiler);
    llvm::outs() << g_pass << " passed, " << g_fail << " failed\n";
    return 1;
  }

  auto input0Or = LoadNpy(input0Path);
  auto input1Or = LoadNpy(input1Path);
  auto expectedOr = LoadNpy(expectedPath);
  auto tilingOr = packExampleTiling(tilingSchemaPath);
  EXPECT((bool)input0Or && (bool)input1Or && (bool)expectedOr && (bool)tilingOr,
         "load example fixtures");
  if (!input0Or || !input1Or || !expectedOr || !tilingOr) {
    if (!input0Or)
      llvm::consumeError(input0Or.takeError());
    if (!input1Or)
      llvm::consumeError(input1Or.takeError());
    if (!expectedOr)
      llvm::consumeError(expectedOr.takeError());
    if (!tilingOr)
      llvm::consumeError(tilingOr.takeError());
    afirt_compiler_destroy(compiler);
    llvm::outs() << g_pass << " passed, " << g_fail << " failed\n";
    return 1;
  }

  std::vector<uint8_t> actual(expectedOr->nbytes());
  const void *inputPtrs[2] = {input0Or->data, input1Or->data};
  size_t inputBytes[2] = {input0Or->nbytes(), input1Or->nbytes()};
  void *outputPtrs[1] = {actual.data()};
  size_t outputBytes[1] = {actual.size()};

  AfirtExecutor executor = afirt_executor_create();
  EXPECT(executor != nullptr, "create executor handle");
  if (!executor) {
    afirt_compiler_destroy(compiler);
    llvm::outs() << g_pass << " passed, " << g_fail << " failed\n";
    return 1;
  }

  char errBuf[1024] = {};
  int initRc = afirt_executor_initialize(executor, 0, errBuf, sizeof(errBuf));
  EXPECT(initRc == 0, "initialize executor");
  EXPECT(errBuf[0] == '\0', "initialize produces no error text");

  std::vector<uint8_t> binary = readBinaryFile(binPath);
  int runRc = afirt_executor_run(
      executor, binary.data(), binary.size(), "relu_transpose_broadcast_add",
      2, inputPtrs, inputBytes, 1, outputPtrs, outputBytes, tilingOr->data(),
      tilingOr->size(), 8, 0x41415246u, errBuf, sizeof(errBuf));
  EXPECT(runRc == 0, "execute example kernel through C API");
  EXPECT(errBuf[0] == '\0', "run produces no error text");

  EXPECT(std::memcmp(actual.data(), expectedOr->data, expectedOr->nbytes()) == 0,
         "C API runtime output matches example expected output");

  std::fill(actual.begin(), actual.end(), 0);
  std::memset(errBuf, 0, sizeof(errBuf));
  EXPECT(!binary.empty(), "compiled binary is readable");
  int runFileRc = afirt_executor_run_file(
      executor, binPath, "relu_transpose_broadcast_add", 2, inputPtrs,
      inputBytes, 1, outputPtrs, outputBytes, tilingOr->data(),
      tilingOr->size(), 8, 0x41415246u, errBuf, sizeof(errBuf));
  EXPECT(runFileRc == 0, "execute example kernel through C API run_file");
  EXPECT(errBuf[0] == '\0', "run_file produces no error text");
  EXPECT(std::memcmp(actual.data(), expectedOr->data, expectedOr->nbytes()) == 0,
         "C API run_file output matches example expected output");

  afirt_executor_destroy(executor);
  afirt_compiler_destroy(compiler);

  llvm::outs() << g_pass << " passed, " << g_fail << " failed\n";
  llvm::outs().flush();
  llvm::errs().flush();
  _Exit(g_fail ? 1 : 0);
}
