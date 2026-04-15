#include "MixDirectCompileInternal.h"

#include "Runtime/MixCommandBuilder.h"
#include "Runtime/Support/PathUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdlib>
#include <cstring>
#include <optional>
#include <sstream>
#include <utility>

namespace mlir::runtime {

namespace {

static constexpr const char *kStageBuildRunner = "build host runner";
static constexpr const char *kStageEmitTilingArtifact = "emit tiling artifact";

static llvm::Expected<uint32_t> readBlockDimFromLaunchInfo(llvm::StringRef path) {
  auto textOr = readTextFileOrErr(path);
  if (!textOr)
    return textOr.takeError();

  llvm::SmallVector<llvm::StringRef> lines;
  llvm::StringRef(*textOr).split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (!line.starts_with("block_dim="))
      continue;
    uint64_t value = 0;
    if (line.drop_front(strlen("block_dim=")).getAsInteger(10, value))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Invalid block_dim in launch info: %s",
                                     path.str().c_str());
    return static_cast<uint32_t>(value);
  }

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "Missing block_dim in launch info: %s",
                                 path.str().c_str());
}

static llvm::StringRef getAclDataType(DType dtype) {
  switch (dtype) {
  case DType::F16:
    return "ge::DT_FLOAT16";
  case DType::BF16:
    return "ge::DT_BF16";
  case DType::F32:
    return "ge::DT_FLOAT";
  case DType::INT8:
    return "ge::DT_INT8";
  case DType::INT32:
    return "ge::DT_INT32";
  case DType::INT64:
    return "ge::DT_INT64";
  default:
    return "ge::DT_FLOAT";
  }
}

static bool useLegacyMixRunner() {
  const char *env = std::getenv("AFIR_MIX_USE_LEGACY_RUNNER");
  return env && llvm::StringRef(env) == "1";
}

static std::string getRunnerToolkitHome() {
  return findAscendHome();
}

static std::string getRunnerLib64(const std::string &ascendHome) {
  return findAscendLib64Dir(ascendHome);
}

static std::string getRunnerSimLibDir(const std::string &ascendHome,
                                      llvm::StringRef socVersion) {
  return findAscendSimulatorLibDir(ascendHome, socVersion);
}

static std::string getRunnerDeviceLibDir(const std::string &ascendHome) {
  return findAscendDeviceLibDir(ascendHome);
}

static llvm::Error writeFileOrErr(llvm::StringRef path,
                                  llvm::StringRef content) {
  return writeTextFile(path, content);
}

static std::string emitRunnerDataUtilsHeader() {
  return R"(#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

inline bool ReadFile(const std::string &path, size_t &actualSize, void *dst, size_t expectedSize) {
  std::ifstream is(path, std::ios::binary);
  if (!is) return false;
  is.seekg(0, std::ios::end);
  actualSize = static_cast<size_t>(is.tellg());
  is.seekg(0, std::ios::beg);
  if (actualSize > expectedSize) return false;
  is.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(actualSize));
  return is.good() || is.eof();
}

inline bool WriteFile(const std::string &path, const void *src, size_t size) {
  std::ofstream os(path, std::ios::binary);
  if (!os) return false;
  os.write(reinterpret_cast<const char *>(src), static_cast<std::streamsize>(size));
  return static_cast<bool>(os);
}

inline void ensureParentDir(const std::string &path) {
  std::string::size_type pos = path.find_last_of('/');
  if (pos == std::string::npos) return;
  std::string dir = path.substr(0, pos);
  std::string cmd = "mkdir -p " + dir;
  (void)std::system(cmd.c_str());
}
)";
}

static std::string emitRunnerMainSource(llvm::StringRef kernelName,
                                        const MixAbiMetadata &abi) {
  const auto &inputA = abi.inputs[0];
  const auto &inputB = abi.inputs[1];
  const auto *bias = abi.inputs.size() > 2 ? &abi.inputs[2] : nullptr;
  const auto &output = abi.outputs[0];
  const auto tensorBytes = [](const MixAbiTensorDesc &tensor) {
    size_t elements = 1;
    for (int64_t dim : tensor.shape)
      elements *= static_cast<size_t>(dim);
    return elements * dtypeBytes(tensor.dtype);
  };
  std::ostringstream os;
  os << "#include <cstdint>\n"
     << "#include <cstdlib>\n"
     << "#include <iostream>\n"
     << "#include <string>\n"
     << "#include \"acl/acl.h\"\n"
     << "#include \"data_utils.h\"\n"
     << "#include \"aclrtlaunch_" << kernelName.str() << ".h\"\n\n"
     << "extern \"C\" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf);\n"
     << "extern \"C\" uint32_t GetBlockDim(const char *socVersion);\n\n"
     << "#define CHECK_ACL(expr) do { \\\n"
     << "  aclError _err = (expr); \\\n"
     << "  if (_err != ACL_SUCCESS) { \\\n"
     << "    std::cerr << \"ACL error: \" << _err << \" at \" << __FILE__ << \":\" << __LINE__ << std::endl; \\\n"
     << "    return 4; \\\n"
     << "  } \\\n"
     << "} while (0)\n\n"
     << "int main(int argc, char **argv) {\n"
     << "  std::string emitTilingFile;\n"
     << "  std::string emitLaunchInfoFile;\n"
     << "  std::string inputDir;\n"
     << "  std::string outputFile;\n"
     << "  for (int i = 1; i < argc; ++i) {\n"
     << "    std::string arg = argv[i];\n"
     << "    if (arg == \"--emit-tiling-file\" && i + 1 < argc) {\n"
     << "      emitTilingFile = argv[++i];\n"
     << "    } else if (arg == \"--emit-launch-info\" && i + 1 < argc) {\n"
     << "      emitLaunchInfoFile = argv[++i];\n"
     << "    } else if (arg == \"--input-dir\" && i + 1 < argc) {\n"
     << "      inputDir = argv[++i];\n"
     << "    } else if (arg == \"--output-file\" && i + 1 < argc) {\n"
     << "      outputFile = argv[++i];\n"
     << "    }\n"
     << "  }\n\n"
     << "  const size_t aFileSize = " << tensorBytes(inputA) << ";\n"
     << "  const size_t bFileSize = " << tensorBytes(inputB) << ";\n";
  if (bias)
    os << "  const size_t biasFileSize = " << tensorBytes(*bias) << ";\n";
  os << "  const size_t cFileSize = " << tensorBytes(output) << ";\n"
     << "  const size_t workspaceSize = " << abi.workspaceBytes << ";\n"
     << "  uint8_t *tilingBuf = static_cast<uint8_t*>(std::malloc(4096));\n"
     << "  if (!tilingBuf) return 1;\n"
     << "  GenerateTiling(SOC_VERSION, tilingBuf);\n"
     << "  uint32_t blockDim = GetBlockDim(SOC_VERSION);\n"
     << "  const size_t tilingFileSize = 4096;\n\n"
     << "  if (!emitTilingFile.empty()) {\n"
     << "    ensureParentDir(emitTilingFile);\n"
     << "    if (!WriteFile(emitTilingFile, tilingBuf, tilingFileSize)) {\n"
     << "      free(tilingBuf);\n"
     << "      return 5;\n"
     << "    }\n"
     << "  }\n"
     << "  if (!emitLaunchInfoFile.empty()) {\n"
     << "    ensureParentDir(emitLaunchInfoFile);\n"
     << "    std::string payload = std::string(\"block_dim=\") + std::to_string(blockDim) + \"\\n\";\n"
     << "    if (!WriteFile(emitLaunchInfoFile, payload.data(), payload.size())) {\n"
     << "      free(tilingBuf);\n"
     << "      return 5;\n"
     << "    }\n"
     << "  }\n"
     << "  if (!emitTilingFile.empty() || !emitLaunchInfoFile.empty()) {\n"
     << "    free(tilingBuf);\n"
     << "    return 0;\n"
     << "  }\n\n"
     << "  CHECK_ACL(aclInit(nullptr));\n"
     << "  int32_t deviceId = 0;\n"
     << "  CHECK_ACL(aclrtSetDevice(deviceId));\n"
     << "  aclrtStream stream = nullptr;\n"
     << "  CHECK_ACL(aclrtCreateStream(&stream));\n\n"
     << "  auto readHostToDevice = [&](const std::string& path, size_t bytes, uint8_t** host, uint8_t** device) {\n"
     << "    size_t fileSize = 0;\n"
     << "    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(host), bytes));\n"
     << "    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(device), bytes, ACL_MEM_MALLOC_HUGE_FIRST));\n"
     << "    if (!ReadFile(path, fileSize, *host, bytes)) return false;\n"
     << "    CHECK_ACL(aclrtMemcpy(*device, bytes, *host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));\n"
     << "    return true;\n"
     << "  };\n\n"
     << "  uint8_t *inputAHost = nullptr, *inputADevice = nullptr;\n"
     << "  uint8_t *inputBHost = nullptr, *inputBDevice = nullptr;\n"
     << "  uint8_t *inputBiasHost = nullptr, *inputBiasDevice = nullptr;\n"
     << "  uint8_t *outputCHost = nullptr, *outputCDevice = nullptr;\n"
     << "  uint8_t *tilingHost = nullptr, *tilingDevice = nullptr;\n"
     << "  uint8_t *workspaceDevice = nullptr;\n\n"
     << "  if (!readHostToDevice(inputDir + \"/" << inputA.runtimeFile
     << "\", aFileSize, &inputAHost, &inputADevice)) return 2;\n"
     << "  if (!readHostToDevice(inputDir + \"/" << inputB.runtimeFile
     << "\", bFileSize, &inputBHost, &inputBDevice)) return 2;\n";
  if (bias) {
    os << "  if (!readHostToDevice(inputDir + \"/" << bias->runtimeFile
       << "\", biasFileSize, &inputBiasHost, &inputBiasDevice)) return 2;\n\n";
  } else {
    os << "\n";
  }
  os << "  CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&outputCHost), cFileSize));\n"
     << "  CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&outputCDevice), cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));\n"
     << "  CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&tilingHost), tilingFileSize));\n"
     << "  CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&tilingDevice), tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));\n"
     << "  CHECK_ACL(aclrtMemcpy(tilingHost, tilingFileSize, tilingBuf, tilingFileSize, ACL_MEMCPY_HOST_TO_HOST));\n"
     << "  CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));\n"
     << "  CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&workspaceDevice), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));\n\n"
     << "  ACLRT_LAUNCH_KERNEL(" << kernelName.str() << ")(blockDim, stream, inputADevice, inputBDevice, inputBiasDevice, outputCDevice, workspaceDevice, tilingDevice);\n"
     << "  CHECK_ACL(aclrtSynchronizeStream(stream));\n"
     << "  CHECK_ACL(aclrtMemcpy(outputCHost, cFileSize, outputCDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST));\n\n"
     << "  size_t lastSlash = outputFile.find_last_of('/');\n"
     << "  if (lastSlash != std::string::npos) {\n"
     << "    std::string outDir = outputFile.substr(0, lastSlash);\n"
     << "    std::string mkdirCmd = \"mkdir -p \" + outDir;\n"
     << "    (void)std::system(mkdirCmd.c_str());\n"
     << "  }\n"
     << "  if (!WriteFile(outputFile, outputCHost, cFileSize)) return 3;\n\n"
     << "  CHECK_ACL(aclrtFree(inputADevice));\n"
     << "  CHECK_ACL(aclrtFreeHost(inputAHost));\n"
     << "  CHECK_ACL(aclrtFree(inputBDevice));\n"
     << "  CHECK_ACL(aclrtFreeHost(inputBHost));\n"
     << "  CHECK_ACL(aclrtFree(outputCDevice));\n"
     << "  CHECK_ACL(aclrtFreeHost(outputCHost));\n"
     << "  CHECK_ACL(aclrtFree(inputBiasDevice));\n"
     << "  CHECK_ACL(aclrtFreeHost(inputBiasHost));\n"
     << "  CHECK_ACL(aclrtFree(tilingDevice));\n"
     << "  CHECK_ACL(aclrtFreeHost(tilingHost));\n"
     << "  CHECK_ACL(aclrtFree(workspaceDevice));\n"
     << "  CHECK_ACL(aclrtDestroyStream(stream));\n"
     << "  CHECK_ACL(aclrtResetDevice(deviceId));\n"
     << "  CHECK_ACL(aclFinalize());\n"
     << "  free(tilingBuf);\n"
     << "  return 0;\n"
     << "}\n";
  return os.str();
}

static std::string emitRunnerTilingSource(const MixAbiMetadata &abi) {
  const auto &inputA = abi.inputs[0];
  const auto &inputB = abi.inputs[1];
  const auto &output = abi.outputs[0];
  const bool hasBias = abi.inputs.size() > 2;
  const int64_t m = output.shape[0];
  const int64_t n = output.shape[1];
  const int64_t k = inputA.shape[1];
  std::ostringstream os;
  os << "#include <cstdint>\n"
     << "#include \"tiling/tiling_api.h\"\n"
     << "#include \"tiling/platform/platform_ascendc.h\"\n\n"
     << "using namespace matmul_tiling;\n\n"
     << "extern \"C\" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf) {\n"
     << "  int M = " << m << ";\n"
     << "  int N = " << n << ";\n"
     << "  int K = " << k << ";\n"
     << "  optiling::TCubeTiling tilingData;\n"
     << "  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);\n"
     << "  MatmulApiTiling tilingApi(*ascendcPlatform);\n\n"
     << "  tilingApi.SetAType(TPosition::GM, CubeFormat::ND, "
     << getAclDataType(inputA.dtype).str() << ", false);\n"
     << "  tilingApi.SetBType(TPosition::GM, CubeFormat::ND, "
     << getAclDataType(inputB.dtype).str() << ", false);\n"
     << "  tilingApi.SetCType(TPosition::GM, CubeFormat::ND, "
     << getAclDataType(output.dtype).str() << ");\n";
  if (hasBias)
    os << "  tilingApi.SetBiasType(TPosition::GM, CubeFormat::ND, "
       << getAclDataType(abi.inputs[2].dtype).str() << ");\n";
  os << "  tilingApi.SetOrgShape(M, N, K);\n"
     << "  tilingApi.SetShape(M, N, K);\n"
     << "  tilingApi.SetBias(" << (hasBias ? "true" : "false") << ");\n"
     << "  tilingApi.SetTraverse(MatrixTraverse::FIRSTM);\n"
     << "  tilingApi.SetFixSplit(M, N, -1);\n"
     << "  tilingApi.SetBufferSpace(-1, -1, -1);\n"
     << "  (void)tilingApi.GetTiling(tilingData);\n"
     << "  tilingData.SaveToBuffer(tilingBuf, tilingData.GetDataSize());\n"
     << "}\n\n"
     << "extern \"C\" uint32_t GetBlockDim(const char *socVersion) {\n"
     << "  int M = " << m << ";\n"
     << "  int N = " << n << ";\n"
     << "  int K = " << k << ";\n"
     << "  optiling::TCubeTiling tilingData;\n"
     << "  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);\n"
     << "  MatmulApiTiling tilingApi(*ascendcPlatform);\n\n"
     << "  tilingApi.SetAType(TPosition::GM, CubeFormat::ND, "
     << getAclDataType(inputA.dtype).str() << ", false);\n"
     << "  tilingApi.SetBType(TPosition::GM, CubeFormat::ND, "
     << getAclDataType(inputB.dtype).str() << ", false);\n"
     << "  tilingApi.SetCType(TPosition::GM, CubeFormat::ND, "
     << getAclDataType(output.dtype).str() << ");\n";
  if (hasBias)
    os << "  tilingApi.SetBiasType(TPosition::GM, CubeFormat::ND, "
       << getAclDataType(abi.inputs[2].dtype).str() << ");\n";
  os << "  tilingApi.SetOrgShape(M, N, K);\n"
     << "  tilingApi.SetShape(M, N, K);\n"
     << "  tilingApi.SetBias(" << (hasBias ? "true" : "false") << ");\n"
     << "  tilingApi.SetTraverse(MatrixTraverse::FIRSTM);\n"
     << "  tilingApi.SetFixSplit(M, N, -1);\n"
     << "  tilingApi.SetBufferSpace(-1, -1, -1);\n"
     << "  (void)tilingApi.GetTiling(tilingData);\n"
     << "  return static_cast<uint32_t>(tilingData.get_usedCoreNum());\n"
     << "}\n";
  return os.str();
}

} // namespace

llvm::Expected<MixDirectTilingOutputs>
executeMixDirectTilingStage(const MixCompileLayout &layout,
                            llvm::StringRef runtimeKernelName,
                            llvm::StringRef socVersion,
                            const MixAbiMetadata &abi) {
  MixDirectTilingOutputs outputs;
  outputs.usedLegacyRunner = useLegacyMixRunner();
  outputs.tilingArtifactPath = layout.tilingArtifactPath;
  outputs.launchInfoPath = layout.launchInfoPath;

  std::vector<std::string> tilingEmitCmd;
  std::vector<std::string> runnerCompileCmd;
  if (outputs.usedLegacyRunner) {
    if (auto err = writeFileOrErr(layout.runnerDataUtilsPath,
                                  emitRunnerDataUtilsHeader()))
      return std::move(err);
    if (auto err = writeFileOrErr(layout.runnerMainPath,
                                  emitRunnerMainSource(runtimeKernelName, abi)))
      return std::move(err);
    if (auto err = writeFileOrErr(layout.runnerTilingPath,
                                  emitRunnerTilingSource(abi)))
      return std::move(err);

    const std::string ascendHome = getRunnerToolkitHome();
    auto davSimLibDirOr = requireAscendDavSimulatorLibDir(ascendHome);
    if (!davSimLibDirOr)
      return davSimLibDirOr.takeError();
    const std::string runnerLib64 = getRunnerLib64(ascendHome);
    const std::string hostCannArch = getHostCannArchDir();
    const std::string runnerAltLib64 =
        hostCannArch.empty() ? std::string()
                             : ascendHome + "/" + hostCannArch + "/lib64";
    const std::string runnerDeviceLibDir = getRunnerDeviceLibDir(ascendHome);
    const std::string runnerSimLibDir =
        getRunnerSimLibDir(ascendHome, socVersion);
    const std::string davSimLibDir = *davSimLibDirOr;
    runnerCompileCmd = buildHostRunnerCompileCommand(
        layout.workDir, layout.launcherDir, layout.outIncludeDir,
        layout.runnerMainPath, layout.runnerTilingPath, layout.runnerBinaryPath,
        layout.kernelSoPath, runnerLib64, runnerSimLibDir, davSimLibDir,
        runnerDeviceLibDir, socVersion);
    const std::string runnerBuildContext = makeStageContext({
        {"main_source", layout.runnerMainPath},
        {"tiling_source", layout.runnerTilingPath},
        {"kernel_so", layout.kernelSoPath},
        {"runner_binary", layout.runnerBinaryPath},
        {"soc_version", socVersion},
    });
    std::string runnerLdLibraryPath = runnerLib64;
    if (runnerAltLib64 != runnerLib64)
      runnerLdLibraryPath += ":" + runnerAltLib64;
    if (!runnerDeviceLibDir.empty())
      runnerLdLibraryPath += ":" + runnerDeviceLibDir;
    runnerLdLibraryPath += ":" + runnerSimLibDir + ":" + davSimLibDir +
                           ":${LD_LIBRARY_PATH:-}";
    tilingEmitCmd = {"/bin/bash",
                     "-lc",
                     "LD_LIBRARY_PATH='" + runnerLdLibraryPath + "' " +
                         layout.runnerBinaryPath + " --emit-tiling-file " +
                         layout.tilingArtifactPath + " --emit-launch-info " +
                         layout.launchInfoPath};
    if (auto err =
            runProcess(runnerCompileCmd, kStageBuildRunner, runnerBuildContext))
      return std::move(err);
    if (auto err = ensureFileExists(layout.runnerBinaryPath, kStageBuildRunner,
                                    runnerBuildContext))
      return std::move(err);
    outputs.runnerSourcePath = layout.runnerMainPath;
    outputs.runnerBinaryPath = layout.runnerBinaryPath;
    outputs.runnerCompileCommand = renderCommandForDebug(runnerCompileCmd);
  } else {
    tilingEmitCmd = buildMixTilingHelperCommand(
        runtimeKernelName, socVersion, abi.inputs[0].shape, abi.inputs[0].dtype,
        abi.inputs[1].shape, abi.inputs[1].dtype, abi.outputs[0].shape,
        abi.outputs[0].dtype,
        abi.inputs.size() > 2 ? std::optional<DType>(abi.inputs[2].dtype)
                              : std::nullopt,
        layout.tilingArtifactPath, layout.launchInfoPath);
    outputs.runnerCompileCommand = renderCommandForDebug(tilingEmitCmd);
  }

  const std::string tilingArtifactContext = makeStageContext({
      {outputs.usedLegacyRunner ? "runner_binary" : "helper",
       tilingEmitCmd.front()},
      {"tiling_artifact", layout.tilingArtifactPath},
      {"launch_info", layout.launchInfoPath},
      {"kernel", runtimeKernelName},
      {"soc_version", socVersion},
  });
  if (auto err = runProcess(tilingEmitCmd, kStageEmitTilingArtifact,
                            tilingArtifactContext))
    return std::move(err);
  if (auto err = ensureFileExists(layout.tilingArtifactPath,
                                  kStageEmitTilingArtifact,
                                  tilingArtifactContext))
    return std::move(err);
  if (auto err = ensureFileExists(layout.launchInfoPath,
                                  kStageEmitTilingArtifact,
                                  tilingArtifactContext))
    return std::move(err);

  auto blockDimOr = readBlockDimFromLaunchInfo(layout.launchInfoPath);
  if (!blockDimOr)
    return blockDimOr.takeError();
  outputs.blockDim = *blockDimOr;
  outputs.tilingEmitCommand = renderCommandForDebug(tilingEmitCmd);
  return outputs;
}

} // namespace mlir::runtime
