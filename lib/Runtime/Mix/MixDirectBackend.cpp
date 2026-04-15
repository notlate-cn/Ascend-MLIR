#include "Runtime/MixDirectBackend.h"
#include "Runtime/MixCommandBuilder.h"
#include "Runtime/MixAbi.h"
#include "Runtime/MixAbiExtractor.h"
#include "Runtime/Mix/MixLegacyCompileCompat.h"
#include "Runtime/NpyIO.h"
#include "Runtime/PathUtils.h"
#include "Runtime/MixSourceAnalyzer.h"
#include "Runtime/MixStubTemplate.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <utility>

namespace mlir::runtime {

namespace {

static llvm::Error writeTextFile(llvm::StringRef path, llvm::StringRef content) {
  std::ofstream os(path.str(), std::ios::binary);
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot write file: %s", path.str().c_str());
  os << content.str();
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to write file: %s", path.str().c_str());
  return llvm::Error::success();
}

static llvm::Expected<std::string> readTextFileOrErr(llvm::StringRef path) {
  auto bufferOr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOr)
    return llvm::createStringError(bufferOr.getError(),
                                   "Cannot read file: %s", path.str().c_str());
  return (*bufferOr)->getBuffer().str();
}

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

static constexpr const char *kStageAnalyzeSource = "analyze source";
static constexpr const char *kStageBuildRunner = "build host runner";
static constexpr const char *kStageEmitTilingArtifact = "emit tiling artifact";

static std::string makeStageContext(
    std::initializer_list<std::pair<llvm::StringRef, llvm::StringRef>> fields) {
  std::string out;
  llvm::raw_string_ostream os(out);
  bool first = true;
  for (const auto &field : fields) {
    if (field.second.empty())
      continue;
    if (!first)
      os << ", ";
    first = false;
    os << field.first << "=" << field.second;
  }
  os.flush();
  return out;
}

static llvm::Error ensureFileExists(llvm::StringRef path, llvm::StringRef stage,
                                    llvm::StringRef context = {}) {
  if (!llvm::sys::fs::exists(path))
  {
    const std::string contextText = context.str();
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "[%s] completed but did not create expected file: %s%s%s",
        stage.str().c_str(), path.str().c_str(), context.empty() ? "" : " (",
        context.empty() ? "" : contextText.c_str(),
        context.empty() ? "" : ")");
  }
  return llvm::Error::success();
}

static std::string joinPath(llvm::StringRef base, llvm::StringRef leaf);
static llvm::Error runProcess(const std::vector<std::string> &args,
                              llvm::StringRef stage,
                              llvm::StringRef context = {});

static std::string joinDefinitions(llvm::ArrayRef<std::string> defs) {
  std::string out;
  for (size_t i = 0; i < defs.size(); ++i) {
    if (i)
      out.push_back(';');
    out += defs[i];
  }
  return out;
}

static bool useLegacyMixRunner() {
  if (const char *value = std::getenv("AFIR_MIX_USE_LEGACY_RUNNER"))
    return *value != '\0' && std::strcmp(value, "0") != 0;
  return false;
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

static llvm::Error writeFileOrErr(llvm::StringRef path, llvm::StringRef content) {
  return writeTextFile(path, content);
}

static std::string emitRunnerDataUtilsHeader() {
  return R"runner(#pragma once
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "acl/acl.h"

#define CHECK_ACL(x) do { aclError __ret = (x); if (__ret != ACL_ERROR_NONE) { \
  std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
} } while (0)

static bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize) {
  struct stat sBuf;
  if (stat(filePath.data(), &sBuf) == -1) return false;
  if (S_ISREG(sBuf.st_mode) == 0) return false;
  std::ifstream file(filePath, std::ios::binary);
  if (!file.is_open()) return false;
  std::filebuf *buf = file.rdbuf();
  size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
  if (size == 0 || size > bufferSize) return false;
  buf->pubseekpos(0, std::ios::in);
  buf->sgetn(static_cast<char *>(buffer), size);
  fileSize = size;
  return true;
}

static bool WriteFile(const std::string &filePath, const void *buffer, size_t size) {
  if (buffer == nullptr) return false;
  int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  if (fd < 0) return false;
  size_t writeSize = write(fd, buffer, size);
  (void)close(fd);
  return writeSize == size;
}
)runner";
}

static std::string emitRunnerMainSource(llvm::StringRef kernelName,
                                        const MixAbiMetadata &abi);
static std::string emitRunnerTilingSource(const MixAbiMetadata &abi);

static llvm::Error runProcess(const std::vector<std::string> &args,
                              llvm::StringRef stage,
                              llvm::StringRef context) {
  if (args.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "[%s] received an empty command",
                                   stage.str().c_str());

  std::vector<llvm::StringRef> argv;
  argv.reserve(args.size());
  for (const auto &arg : args)
    argv.push_back(arg);

  std::string errMsg;
  std::optional<llvm::StringRef> redirects[3];
  int ret = llvm::sys::ExecuteAndWait(argv[0], argv, std::nullopt, redirects,
                                      300, 0, &errMsg);
  if (ret != 0) {
    const std::string program = argv[0].str();
    const std::string contextText = context.str();
    std::string renderedCommand = renderCommandForDebug(args);
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "[%s] program=%s%s%s failed (exit %d): %s\n  command: %s",
        stage.str().c_str(), program.c_str(),
        context.empty() ? "" : " inputs: ",
        context.empty() ? "" : contextText.c_str(), ret, errMsg.c_str(),
        renderedCommand.c_str());
  }
  return llvm::Error::success();
}

static std::string joinPath(llvm::StringRef base, llvm::StringRef leaf) {
  llvm::SmallString<256> joined(base);
  llvm::sys::path::append(joined, leaf);
  return joined.str().str();
}

static uint64_t getElementBytes(DType dtype) {
  if (dtype == DType::F16)
    return sizeof(int16_t);
  if (dtype == DType::F32)
    return sizeof(float);
  return 0;
}

static uint64_t getTensorElementCount(llvm::ArrayRef<int64_t> shape) {
  uint64_t count = 1;
  for (int64_t dim : shape)
    count *= static_cast<uint64_t>(dim);
  return count;
}

static bool hasDynamicShape(llvm::ArrayRef<int64_t> shape) {
  for (int64_t dim : shape)
    if (dim < 0)
      return true;
  return false;
}

static llvm::Expected<size_t>
findFirstDynamicTensorIndex(llvm::ArrayRef<MixAbiTensorDesc> tensors) {
  for (size_t i = 0; i < tensors.size(); ++i)
    if (hasDynamicShape(tensors[i].shape))
      return i;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "no dynamic tensor shape found");
}

static uint64_t getTensorBytes(const MixAbiTensorDesc &tensor) {
  return getTensorElementCount(tensor.shape) * getElementBytes(tensor.dtype);
}

static llvm::StringRef getDTypeName(DType dtype) {
  if (dtype == DType::F16)
    return "f16";
  if (dtype == DType::BF16)
    return "bf16";
  if (dtype == DType::F32)
    return "f32";
  if (dtype == DType::INT8)
    return "int8";
  if (dtype == DType::INT32)
    return "int32";
  if (dtype == DType::INT64)
    return "int64";
  return "unknown";
}

static std::string buildOrdinalTensorNpyName(bool isOutput, size_t index) {
  return (llvm::Twine(isOutput ? "output" : "input") + llvm::Twine(index) +
          ".npy")
      .str();
}

static llvm::Expected<std::string>
resolveTensorNpyPath(llvm::StringRef npyDir, const MixAbiTensorDesc &tensor,
                     size_t index, bool isOutput) {
  const std::string namedPath = joinPath(npyDir, tensor.name + ".npy");
  if (llvm::sys::fs::exists(namedPath))
    return namedPath;
  const std::string ordinalPath =
      joinPath(npyDir, buildOrdinalTensorNpyName(isOutput, index));
  if (llvm::sys::fs::exists(ordinalPath))
    return ordinalPath;
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "RuntimeMix cannot resolve concrete %s tensor shape for ABI tensor '%s': "
      "expected either %s or %s",
      isOutput ? "output" : "input", tensor.name.c_str(), namedPath.c_str(),
      ordinalPath.c_str());
}

static llvm::Error reconcileTensorWithNpy(MixAbiTensorDesc &tensor,
                                          llvm::StringRef npyPath) {
  auto arrayOr = LoadNpy(npyPath.str());
  if (!arrayOr)
    return arrayOr.takeError();
  if (arrayOr->dtype != tensor.dtype)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix ABI dtype mismatch for tensor '%s': MLIR expects %s but %s "
        "contains %s",
        tensor.name.c_str(), getDTypeName(tensor.dtype).str().c_str(),
        npyPath.str().c_str(), getDTypeName(arrayOr->dtype).str().c_str());
  if (!tensor.shape.empty() && tensor.shape.size() != arrayOr->shape.size())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix ABI rank mismatch for tensor '%s': MLIR rank=%zu but %s "
        "rank=%zu",
        tensor.name.c_str(), tensor.shape.size(), npyPath.str().c_str(),
        arrayOr->shape.size());
  if (tensor.shape.empty()) {
    tensor.shape = arrayOr->shape;
    return llvm::Error::success();
  }
  for (size_t i = 0; i < arrayOr->shape.size(); ++i) {
    if (tensor.shape[i] >= 0 && tensor.shape[i] != arrayOr->shape[i])
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "RuntimeMix ABI shape mismatch for tensor '%s' dim %zu: MLIR expects "
          "%lld but %s provides %lld",
          tensor.name.c_str(), i, static_cast<long long>(tensor.shape[i]),
          npyPath.str().c_str(), static_cast<long long>(arrayOr->shape[i]));
    tensor.shape[i] = arrayOr->shape[i];
  }
  return llvm::Error::success();
}

static llvm::Error resolveDynamicShapesFromNpyDir(MixAbiMetadata &abi,
                                                  llvm::StringRef npyDir) {
  for (size_t i = 0; i < abi.inputs.size(); ++i) {
    if (!hasDynamicShape(abi.inputs[i].shape))
      continue;
    auto npyPathOr = resolveTensorNpyPath(npyDir, abi.inputs[i], i, false);
    if (!npyPathOr)
      return npyPathOr.takeError();
    if (auto err = reconcileTensorWithNpy(abi.inputs[i], *npyPathOr))
      return err;
  }
  for (size_t i = 0; i < abi.outputs.size(); ++i) {
    if (!hasDynamicShape(abi.outputs[i].shape))
      continue;
    auto npyPathOr = resolveTensorNpyPath(npyDir, abi.outputs[i], i, true);
    if (!npyPathOr)
      return npyPathOr.takeError();
    if (auto err = reconcileTensorWithNpy(abi.outputs[i], *npyPathOr))
      return err;
  }
  return llvm::Error::success();
}

static llvm::StringRef getAclDataType(DType dtype) {
  if (dtype == DType::F16)
    return "DataType::DT_FLOAT16";
  if (dtype == DType::F32)
    return "DataType::DT_FLOAT";
  return "DataType::DT_UNDEFINED";
}

static std::string emitRunnerMainSource(llvm::StringRef kernelName,
                                        const MixAbiMetadata &abi) {
  const auto &inputA = abi.inputs[0];
  const auto &inputB = abi.inputs[1];
  const auto &output = abi.outputs[0];
  const MixAbiTensorDesc *bias =
      abi.inputs.size() > 2 ? &abi.inputs[2] : nullptr;
  std::ostringstream os;
  os << "#include \"data_utils.h\"\n"
     << "#include \"kernel_tiling/kernel_tiling.h\"\n"
     << "#include \"tiling/platform/platform_ascendc.h\"\n"
     << "#include \"acl/acl.h\"\n"
     << "#include \"aclrtlaunch_" << kernelName.str() << ".h\"\n"
     << "#include <cstdint>\n"
     << "#include <cstdlib>\n"
     << "#include <cstring>\n"
     << "#include <string>\n\n"
     << "extern \"C\" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf);\n"
     << "extern \"C\" uint32_t GetBlockDim(const char *socVersion);\n\n"
     << "int main(int argc, char *argv[]) {\n"
     << "  std::string inputDir = \"./input\";\n"
     << "  std::string outputFile = \"./output/" << output.runtimeFile << "\";\n"
     << "  std::string emitTilingFile;\n"
     << "  std::string emitLaunchInfoFile;\n"
     << "  for (int i = 1; i < argc; ++i) {\n"
     << "    std::string arg = argv[i];\n"
     << "    if (arg == \"--input-dir\" && i + 1 < argc) inputDir = argv[++i];\n"
     << "    else if (arg == \"--output-file\" && i + 1 < argc) outputFile = argv[++i];\n"
     << "    else if (arg == \"--emit-tiling-file\" && i + 1 < argc) emitTilingFile = argv[++i];\n"
     << "    else if (arg == \"--emit-launch-info\" && i + 1 < argc) emitLaunchInfoFile = argv[++i];\n"
     << "  }\n\n"
     << "  const char *socVersion = SOC_VERSION;\n"
     << "  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);\n"
     << "  size_t aFileSize = " << getTensorBytes(inputA) << ";\n"
     << "  size_t bFileSize = " << getTensorBytes(inputB) << ";\n"
     << "  size_t cFileSize = " << getTensorBytes(output) << ";\n";
  if (bias)
    os << "  size_t biasFileSize = " << getTensorBytes(*bias) << ";\n";
  os << "  size_t userWorkspaceSize = "
     << static_cast<unsigned long long>(abi.workspaceBytes) << ";\n"
     << "  size_t systemWorkspaceSize = ascendcPlatform ? static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize()) : 0;\n"
     << "  size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;\n"
     << "  size_t tilingFileSize = sizeof(TCubeTiling);\n"
     << "  uint8_t *tilingBuf = static_cast<uint8_t *>(malloc(tilingFileSize));\n"
     << "  GenerateTiling(socVersion, tilingBuf);\n"
     << "  uint32_t blockDim = GetBlockDim(socVersion);\n"
     << "  auto ensureParentDir = [&](const std::string& path) {\n"
     << "    size_t lastSlash = path.find_last_of('/');\n"
     << "    if (lastSlash != std::string::npos) {\n"
     << "      std::string outDir = path.substr(0, lastSlash);\n"
     << "      std::string mkdirCmd = \"mkdir -p \" + outDir;\n"
     << "      (void)std::system(mkdirCmd.c_str());\n"
     << "    }\n"
     << "  };\n"
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

static llvm::Error writeDebugManifest(const MixAnalyzedKernel &analyzed,
                                      llvm::StringRef runtimeKernelName,
                                      const MixAbiMetadata &abi,
                                      llvm::StringRef sourcePath,
                                      llvm::StringRef hostSourcePath,
                                      llvm::StringRef preprocessCompileCommandsPath,
                                      llvm::StringRef preprocessCommand,
                                      llvm::StringRef preprocessGeneratedDir,
                                      llvm::StringRef generatedSourcePath,
                                      llvm::StringRef aicDefinitions,
                                      llvm::StringRef aivDefinitions,
                                      llvm::StringRef workDir,
                                      llvm::StringRef objectDir,
                                      llvm::StringRef outDir,
                                      llvm::StringRef mergeDir,
                                      llvm::StringRef launcherHeaderDir,
                                      llvm::StringRef hostStubSourcePath,
                                      llvm::StringRef hostStubObjectPath,
                                      llvm::StringRef kernelSoPath,
                                      llvm::StringRef mixFlagPath,
                                      llvm::StringRef runnerSourcePath,
                                      llvm::StringRef runnerBinaryPath,
                                      llvm::StringRef aicObj,
                                      llvm::StringRef aivObj,
                                      llvm::StringRef aicRelocObj,
                                      llvm::StringRef aivRelocObj,
                                      llvm::StringRef mergedDeviceObj,
                                      llvm::StringRef aicCompileCmd,
                                      llvm::StringRef aivCompileCmd,
                                      llvm::StringRef aicRelocCmd,
                                      llvm::StringRef aivRelocCmd,
                                      llvm::StringRef mergeCmd,
                                      llvm::StringRef hostCompileCmd,
                                      llvm::StringRef hostBishengObjectPath,
                                      llvm::StringRef hostBishengCmd,
                                      llvm::StringRef hostObjectDir,
                                      llvm::StringRef packCmd,
                                      llvm::StringRef linkCmd,
                                      llvm::StringRef recompileCmd,
                                      llvm::StringRef runnerCompileCmd,
                                      llvm::StringRef metadataPath,
                                      llvm::StringRef manifestPath) {
  std::string manifest;
  manifest += std::string("kernel_name=") + runtimeKernelName.str() + "\n";
  manifest += std::string("requested_kernel_name=") + analyzed.kernelName + "\n";
  manifest += std::string("soc_version=") + analyzed.socVersion + "\n";
  manifest += std::string("kernel_kind=mix\n");
  manifest += std::string("mix_resource_type=mix_1c1v\n");
  manifest += std::string("source_path=") + sourcePath.str() + "\n";
  if (!hostSourcePath.empty())
    manifest += std::string("host_source_path=") + hostSourcePath.str() + "\n";
  manifest += std::string("preprocess_compile_commands=") +
              preprocessCompileCommandsPath.str() + "\n";
  manifest += std::string("preprocess_command=") + preprocessCommand.str() +
              "\n";
  manifest += std::string("preprocess_generated_dir=") +
              preprocessGeneratedDir.str() + "\n";
  manifest += std::string("generated_source_path=") + generatedSourcePath.str() +
              "\n";
  manifest += std::string("aic_definitions=") + aicDefinitions.str() + "\n";
  manifest += std::string("aiv_definitions=") + aivDefinitions.str() + "\n";
  manifest += std::string("work_dir=") + workDir.str() + "\n";
  manifest += std::string("build_dir=") + objectDir.str() + "\n";
  manifest += std::string("install_dir=") + outDir.str() + "\n";
  manifest += std::string("object_dir=") + objectDir.str() + "\n";
  manifest += std::string("out_dir=") + outDir.str() + "\n";
  if (metadataPath.empty()) {
    manifest += std::string("abi_kind=mix_gm_workspace_tiling\n");
    auto abiManifestOr = serializeMixAbiManifest(abi);
    if (!abiManifestOr)
      return abiManifestOr.takeError();
    manifest += *abiManifestOr;
  }
  manifest += std::string("merge_obj_dir=") + mergeDir.str() + "\n";
  manifest += std::string("launcher_header_dir=") +
              launcherHeaderDir.str() + "\n";
  manifest += std::string("host_runner_path=") + runnerBinaryPath.str() + "\n";
  manifest += std::string("manifest_path=") + manifestPath.str() + "\n";
  if (!metadataPath.empty())
    manifest += std::string("metadata_path=") + metadataPath.str() + "\n";
  if (!hostStubSourcePath.empty())
    manifest += std::string("host_stub_source_path=") +
                hostStubSourcePath.str() + "\n";
  manifest += std::string("host_stub_object_path=") + hostStubObjectPath.str() +
              "\n";
  if (!hostBishengObjectPath.empty())
    manifest += std::string("host_bisheng_object=") +
                hostBishengObjectPath.str() + "\n";
  if (!hostObjectDir.empty())
    manifest += std::string("host_object_dir=") + hostObjectDir.str() + "\n";
  manifest += std::string("kernel_so_path=") + kernelSoPath.str() + "\n";
  manifest += std::string("mix_build_flag=") + mixFlagPath.str() + "\n";
  manifest += std::string("host_runner_source_path=") +
              runnerSourcePath.str() + "\n";
  manifest += std::string("aic_object=") + aicObj.str() + "\n";
  manifest += std::string("aiv_object=") + aivObj.str() + "\n";
  manifest += std::string("aic_reloc_object=") + aicRelocObj.str() + "\n";
  manifest += std::string("aiv_reloc_object=") + aivRelocObj.str() + "\n";
  manifest += std::string("bisheng_aic=") + aicCompileCmd.str() + "\n";
  manifest += std::string("bisheng_aiv=") + aivCompileCmd.str() + "\n";
  manifest += std::string("lld_reloc_aic=") + aicRelocCmd.str() + "\n";
  manifest += std::string("lld_reloc_aiv=") + aivRelocCmd.str() + "\n";
  manifest += std::string("lld_merge=") + mergeCmd.str() + "\n";
  manifest += std::string("host_compile_cmd=") + hostCompileCmd.str() + "\n";
  if (!hostBishengCmd.empty())
    manifest += std::string("host_bisheng_cmd=") + hostBishengCmd.str() + "\n";
  manifest += std::string("pack_cmd=") + packCmd.str() + "\n";
  manifest += std::string("host_link_cmd=") + linkCmd.str() + "\n";
  if (!recompileCmd.empty())
    manifest += std::string("recompile_cmd=") + recompileCmd.str() + "\n";
  manifest += std::string("host_runner_compile_cmd=") + runnerCompileCmd.str() +
              "\n";
  if (!mergedDeviceObj.empty())
    manifest += std::string("device_object_path=") + mergedDeviceObj.str() +
                "\n";
  return writeTextFile(manifestPath, manifest);
}

} // namespace

llvm::Expected<MixArtifact>
MixDirectBackend::compile(const MixDirectCompileConfig &cfg) {
  if (cfg.outputDir.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "RuntimeMix direct backend requires an output directory");
  if (cfg.kernelSrc.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "RuntimeMix direct backend requires a kernel source path");
  if (cfg.kernelName.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "RuntimeMix direct backend requires a kernel name");
  if (auto ascendHomeOr = requireAscendHome(); !ascendHomeOr)
    return ascendHomeOr.takeError();

  auto layoutOr = buildLegacyMixCompileLayout(cfg.outputDir, cfg.kernelName);
  if (!layoutOr)
    return layoutOr.takeError();
  const MixCompileLayout &layout = *layoutOr;

  const std::string analyzeContext = makeStageContext({
      {"kernel", cfg.kernelName},
      {"source", cfg.kernelSrc},
      {"soc_version", cfg.socVersion},
  });
  llvm::SmallString<256> sourcePath(cfg.kernelSrc);
  if (auto ec = llvm::sys::fs::make_absolute(sourcePath))
    return llvm::createStringError(
        ec, "[%s] cannot resolve kernel source path: %s (inputs: %s)",
        kStageAnalyzeSource, cfg.kernelSrc.c_str(),
        analyzeContext.c_str());
  if (!llvm::sys::fs::exists(sourcePath))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "[%s] kernel source file not found: %s (inputs: %s)",
        kStageAnalyzeSource, sourcePath.c_str(),
        analyzeContext.c_str());

  auto analyzed = analyzeMixKernel(sourcePath, cfg.kernelName, cfg.socVersion);
  if (!analyzed)
  {
    const std::string analysisError = llvm::toString(analyzed.takeError());
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "[%s] inputs: %s: %s", kStageAnalyzeSource,
        analyzeContext.c_str(), analysisError.c_str());
  }

  const std::string &aicObj = layout.aicObj;
  const std::string &aivObj = layout.aivObj;
  const std::string &aicRelocObj = layout.aicRelocObj;
  const std::string &aivRelocObj = layout.aivRelocObj;
  const std::string &mergedDeviceObj = layout.mergedDeviceObj;
  const std::string &manifestPath = layout.manifestPath;
  const std::string &metadataPath = layout.metadataPath;
  const std::string &analysisPath = layout.analysisPath;
  const std::string &hostStubObjectPath = layout.hostStubObjectPath;
  const std::string &kernelSoPath = layout.kernelSoPath;
  const std::string &mixFlagPath = layout.mixFlagPath;
  const std::string &runnerMainPath = layout.runnerMainPath;
  const std::string &runnerTilingPath = layout.runnerTilingPath;
  const std::string &runnerDataUtilsPath = layout.runnerDataUtilsPath;
  const std::string &runnerBinaryPath = layout.runnerBinaryPath;
  const std::string &tilingArtifactPath = layout.tilingArtifactPath;
  const std::string &launchInfoPath = layout.launchInfoPath;
  const std::string tilingArtifactSource = "out/tiling.bin";

  MixAnalyzedKernel deviceAnalyzed = *analyzed;
  std::string generatedSourcePath;
  std::string hostStubSourcePath;
  std::string preprocessCompileCommandsPath;
  std::string preprocessCommand;
  std::string preprocessGeneratedDir;
  std::string runtimeKernelName = cfg.kernelName;
  auto compatOr = loadLegacyMixCompileContract(
      layout, sourcePath, cfg.kernelName, cfg.socVersion, *analyzed);
  if (!compatOr)
    return compatOr.takeError();

  generatedSourcePath = compatOr->generatedSourcePath;
  deviceAnalyzed.aicDefines = compatOr->aicDefinitions;
  deviceAnalyzed.aivDefines = compatOr->aivDefinitions;
  runtimeKernelName = compatOr->runtimeKernelName;
  auto buildOr = executeLegacyMixBinaryBuild(*compatOr, sourcePath, cfg.kernelName,
                                             cfg.socVersion);
  if (!buildOr)
    return buildOr.takeError();
  hostStubSourcePath = buildOr->hostStubSourcePath;
  preprocessCompileCommandsPath = buildOr->preprocessCompileCommandsPath;
  preprocessCommand = buildOr->preprocessCommand;
  preprocessGeneratedDir = buildOr->preprocessGeneratedDir;

  MixAbiMetadata abi;
  if (cfg.cannMlirPath && !cfg.cannMlirPath->empty()) {
    auto abiOr = extractMixAbiFromCannMlir(*cfg.cannMlirPath);
    if (!abiOr)
      return abiOr.takeError();
    abi = std::move(*abiOr);
    if (cfg.npyDir && !cfg.npyDir->empty()) {
      if (auto err = resolveDynamicShapesFromNpyDir(abi, *cfg.npyDir))
        return err;
    }
    if (auto inputIndexOr = findFirstDynamicTensorIndex(abi.inputs))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "RuntimeMix MLIR ABI extraction from %s produced unresolved dynamic "
          "input shape for tensor '%s'; pass --npy-dir with concrete IO data "
          "to resolve dynamic extents",
          cfg.cannMlirPath->c_str(), abi.inputs[*inputIndexOr].name.c_str());
    else
      llvm::consumeError(inputIndexOr.takeError());
    if (auto outputIndexOr = findFirstDynamicTensorIndex(abi.outputs))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "RuntimeMix MLIR ABI extraction from %s produced unresolved dynamic "
          "output shape for tensor '%s'; pass --npy-dir with concrete IO data "
          "to resolve dynamic extents",
          cfg.cannMlirPath->c_str(), abi.outputs[*outputIndexOr].name.c_str());
    else
      llvm::consumeError(outputIndexOr.takeError());
  } else {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix direct backend requires --cann-mlir to derive ABI and "
        "canonical IO metadata for kernel '%s'",
        cfg.kernelName.c_str());
  }
  if (abi.logicalKernelName.empty())
    abi.logicalKernelName = runtimeKernelName;
  abi.runtimeKernelName = runtimeKernelName;
  if (abi.launcherSymbol.empty())
    abi.launcherSymbol = "aclrtlaunch_" + runtimeKernelName;
  if (abi.aicEntry.empty())
    abi.aicEntry = runtimeKernelName + "_0_mix_aic";
  if (abi.aivEntry.empty())
    abi.aivEntry = runtimeKernelName + "_0_mix_aiv";
  if (abi.workspaceBytes == 0)
    abi.workspaceBytes = 16777216ULL;
  abi.workspaceMode = "fixed";
  abi.tilingMode = "generated_file";
  abi.tilingSource = tilingArtifactSource;
  const bool useLegacyRunner = useLegacyMixRunner();
  std::vector<std::string> tilingEmitCmd;
  std::vector<std::string> runnerCompileCmd;
  std::string runnerMainSourcePath;
  std::string runnerBinaryOutputPath;
  if (useLegacyRunner) {
    if (auto err =
            writeFileOrErr(runnerDataUtilsPath, emitRunnerDataUtilsHeader()))
      return err;
    if (auto err = writeFileOrErr(
            runnerMainPath, emitRunnerMainSource(runtimeKernelName, abi)))
      return err;
    if (auto err =
            writeFileOrErr(runnerTilingPath, emitRunnerTilingSource(abi)))
      return err;

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
        getRunnerSimLibDir(ascendHome, cfg.socVersion);
    const std::string davSimLibDir = *davSimLibDirOr;
        runnerCompileCmd = buildHostRunnerCompileCommand(
        layout.workDir, layout.launcherDir, layout.outIncludeDir,
        runnerMainPath, runnerTilingPath,
        runnerBinaryPath, kernelSoPath, runnerLib64, runnerSimLibDir,
        davSimLibDir, runnerDeviceLibDir, cfg.socVersion);
    const std::string runnerBuildContext = makeStageContext({
        {"main_source", runnerMainPath},
        {"tiling_source", runnerTilingPath},
        {"kernel_so", kernelSoPath},
        {"runner_binary", runnerBinaryPath},
        {"soc_version", cfg.socVersion},
    });
    std::string runnerLdLibraryPath = runnerLib64;
    if (runnerAltLib64 != runnerLib64)
      runnerLdLibraryPath += ":" + runnerAltLib64;
    if (!runnerDeviceLibDir.empty())
      runnerLdLibraryPath += ":" + runnerDeviceLibDir;
    runnerLdLibraryPath += ":" + runnerSimLibDir + ":" + davSimLibDir +
                           ":${LD_LIBRARY_PATH:-}";
    tilingEmitCmd = {
        "/bin/bash",
        "-lc",
        "LD_LIBRARY_PATH='" + runnerLdLibraryPath + "' " + runnerBinaryPath +
            " --emit-tiling-file " + tilingArtifactPath +
            " --emit-launch-info " + launchInfoPath,
    };
    if (auto err = runProcess(runnerCompileCmd, kStageBuildRunner,
                              runnerBuildContext))
      return err;
    if (auto err = ensureFileExists(runnerBinaryPath, kStageBuildRunner,
                                    runnerBuildContext))
      return err;
    runnerMainSourcePath = runnerMainPath;
    runnerBinaryOutputPath = runnerBinaryPath;
  } else {
    tilingEmitCmd = buildMixTilingHelperCommand(
        runtimeKernelName, cfg.socVersion, abi.inputs[0].shape,
        abi.inputs[0].dtype, abi.inputs[1].shape, abi.inputs[1].dtype,
        abi.outputs[0].shape, abi.outputs[0].dtype,
        abi.inputs.size() > 2 ? std::optional<DType>(abi.inputs[2].dtype)
                              : std::nullopt,
        tilingArtifactPath, launchInfoPath);
  }
  const std::string tilingArtifactContext = makeStageContext({
      {useLegacyRunner ? "runner_binary" : "helper", tilingEmitCmd.front()},
      {"tiling_artifact", tilingArtifactPath},
      {"launch_info", launchInfoPath},
      {"kernel", runtimeKernelName},
      {"soc_version", cfg.socVersion},
  });
  if (auto err =
          runProcess(tilingEmitCmd, kStageEmitTilingArtifact, tilingArtifactContext))
    return err;
  if (auto err = ensureFileExists(tilingArtifactPath, kStageEmitTilingArtifact,
                                  tilingArtifactContext))
    return err;
  if (auto err = ensureFileExists(launchInfoPath, kStageEmitTilingArtifact,
                                  tilingArtifactContext))
    return err;

  auto blockDimOr = readBlockDimFromLaunchInfo(launchInfoPath);
  if (!blockDimOr)
    return blockDimOr.takeError();
  abi.blockDim = *blockDimOr;

  auto metadataPathOr = writeLegacyMixCompileMetadataFile(
      metadataPath, runtimeKernelName, cfg.socVersion,
      "mix_1c1v", generatedSourcePath, deviceAnalyzed.aicDefines,
      deviceAnalyzed.aivDefines, mergedDeviceObj, kernelSoPath,
      tilingArtifactPath, launchInfoPath, abi, useLegacyRunner);
  if (!metadataPathOr)
    return metadataPathOr.takeError();

  if (auto err = writeTextFile(
          analysisPath,
          std::string("kernel_name=") + runtimeKernelName + "\n" +
              std::string("requested_kernel_name=") + analyzed->kernelName +
              "\n" +
              std::string("soc_version=") + analyzed->socVersion + "\n" +
              std::string("source_path=") + sourcePath.str().str() + "\n" +
              (buildOr->hostSourcePath.empty() ? std::string{}
                                      : std::string("host_source_path=") +
                                            buildOr->hostSourcePath + "\n") +
              std::string("generated_source_path=") + generatedSourcePath + "\n" +
              std::string("aic_definitions=") +
              joinDefinitions(deviceAnalyzed.aicDefines) + "\n" +
              std::string("aiv_definitions=") +
              joinDefinitions(deviceAnalyzed.aivDefines) + "\n" +
              std::string("aic_object=") + aicObj + "\n" +
              std::string("aiv_object=") + aivObj + "\n" +
              std::string("aic_reloc_object=") + aicRelocObj + "\n" +
              std::string("aiv_reloc_object=") + aivRelocObj + "\n" +
              std::string("device_object=") + mergedDeviceObj + "\n"))
    return err;

  if (auto err = writeDebugManifest(*analyzed, runtimeKernelName, abi, sourcePath,
                                    buildOr->hostSourcePath,
                                    preprocessCompileCommandsPath,
                                    preprocessCommand,
                                    preprocessGeneratedDir,
                                    generatedSourcePath,
                                    joinDefinitions(deviceAnalyzed.aicDefines),
                                    joinDefinitions(deviceAnalyzed.aivDefines),
                                    layout.workDir, layout.objectDir,
                                    layout.outDir, layout.mergeDir,
                                    layout.outIncludeDir,
                                    hostStubSourcePath, hostStubObjectPath,
                                    kernelSoPath, mixFlagPath,
                                    runnerMainSourcePath, runnerBinaryOutputPath,
                                    aicObj, aivObj,
                                    aicRelocObj, aivRelocObj, mergedDeviceObj,
                                    buildOr->aicCompileCommand,
                                    buildOr->aivCompileCommand,
                                    buildOr->aicRelocCommand,
                                    buildOr->aivRelocCommand,
                                    buildOr->mergeCommand,
                                    buildOr->hostCompileCommand,
                                    buildOr->hostBishengObjectPath,
                                    buildOr->hostBishengCommand,
                                    buildOr->hostObjectDir,
                                    buildOr->packCommand,
                                    buildOr->hostLinkCommand,
                                    buildOr->recompileCommand,
                                    renderCommandForDebug(useLegacyRunner
                                                              ? runnerCompileCmd
                                                              : tilingEmitCmd),
                                    *metadataPathOr,
                                    manifestPath))
    return err;

  MixArtifact artifact;
  artifact.kernel_name = runtimeKernelName;
  artifact.soc_version = analyzed->socVersion;
  artifact.work_dir = layout.workDir;
  artifact.build_dir = layout.objectDir;
  artifact.install_dir = layout.outDir;
  artifact.kernel_so_path = kernelSoPath;
  artifact.launcher_header_dir = layout.outIncludeDir;
  artifact.host_runner_path = runnerBinaryOutputPath;
  artifact.host_stub_source_path = hostStubSourcePath;
  artifact.device_object_path = mergedDeviceObj;
  artifact.manifest_path = manifestPath;
  artifact.metadata_path = *metadataPathOr;
  return artifact;
}

KernelArtifact normalizeMixArtifact(const MixArtifact &artifact, KernelKind kind,
                                    MixResourceType mixResourceType) {
  KernelArtifact normalized;
  normalized.kernelName = artifact.kernel_name;
  normalized.kernelKind = kind;
  normalized.mixResourceType = mixResourceType;
  normalized.socVersion = artifact.soc_version;
  normalized.artifactRoot = llvm::sys::path::parent_path(artifact.work_dir).str();
  if (normalized.artifactRoot.empty())
    normalized.artifactRoot = artifact.install_dir;
  normalized.deviceBinaryPath = artifact.device_object_path.empty()
                                    ? artifact.kernel_so_path
                                    : artifact.device_object_path;
  normalized.packedSharedObjectPath = artifact.kernel_so_path;
  normalized.manifestPath = artifact.manifest_path;
  normalized.metadataPath = artifact.metadata_path;
  return normalized;
}

} // namespace mlir::runtime
