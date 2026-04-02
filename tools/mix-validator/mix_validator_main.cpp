#include "Runtime/Executor.h"
#include "Runtime/Types.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <dlfcn.h>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace mlir::runtime;

static llvm::cl::opt<std::string> ArtifactRoot("artifact-root",
                                               llvm::cl::Required);
static llvm::cl::opt<std::string> InputDir("input-dir", llvm::cl::Required);
static llvm::cl::opt<std::string> Golden("golden", llvm::cl::Required);
static llvm::cl::opt<std::string> OutputFile("output-file",
                                            llvm::cl::init(""));
static llvm::cl::opt<std::string> SocVersion("soc",
                                             llvm::cl::init("Ascend910B1"));
static llvm::cl::opt<bool> ForceDirectPacked(
    "force-direct-packed",
    llvm::cl::desc("Bypass mix_runner and validate through direct packed execution"),
    llvm::cl::init(false));

static llvm::Expected<std::vector<uint8_t>> readBinary(const std::string &path) {
  auto buf = llvm::MemoryBuffer::getFile(path, false);
  if (!buf)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot read binary: %s", path.c_str());
  const auto *data =
      reinterpret_cast<const uint8_t *>((*buf)->getBufferStart());
  return std::vector<uint8_t>(data, data + (*buf)->getBufferSize());
}

static llvm::Error writeBinary(const std::string &path,
                               llvm::ArrayRef<uint8_t> bytes) {
  llvm::SmallString<256> dir(path);
  llvm::sys::path::remove_filename(dir);
  if (!dir.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(dir))
      return llvm::createStringError(ec, "Cannot create directory: %s",
                                     dir.c_str());
  }
  std::ofstream os(path, std::ios::binary);
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot write file: %s", path.c_str());
  os.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to write file: %s", path.c_str());
  return llvm::Error::success();
}

static std::map<std::string, std::string> readManifest(const std::string &path) {
  std::map<std::string, std::string> out;
  auto bufOr = llvm::MemoryBuffer::getFile(path, false);
  if (!bufOr)
    return out;
  llvm::StringRef content = (*bufOr)->getBuffer();
  llvm::SmallVector<llvm::StringRef> lines;
  content.split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (line.empty() || line.starts_with("#"))
      continue;
    auto eq = line.find('=');
    if (eq == llvm::StringRef::npos)
      continue;
    out.emplace(line.substr(0, eq).str(), line.substr(eq + 1).str());
  }
  return out;
}

struct AbiTensorDesc {
  std::string name;
  std::string file;
  DType dtype = DType::F16;
  std::vector<int64_t> shape;
};

struct AbiMetadata {
  std::vector<AbiTensorDesc> inputs;
  std::vector<AbiTensorDesc> outputs;
  size_t workspaceBytes = 0;
  std::string workspaceMode;
  std::string tilingMode;
  std::string tilingSource;
};

static llvm::Expected<DType> parseDType(llvm::StringRef dtype) {
  if (dtype == "f16")
    return DType::F16;
  if (dtype == "f32")
    return DType::F32;
  if (dtype == "bf16")
    return DType::BF16;
  if (dtype == "int8")
    return DType::INT8;
  if (dtype == "int32")
    return DType::INT32;
  if (dtype == "int64")
    return DType::INT64;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "Unsupported ABI dtype: %s",
                                 dtype.str().c_str());
}

static llvm::Expected<std::vector<int64_t>>
parseShapeList(llvm::StringRef value) {
  std::vector<int64_t> shape;
  llvm::SmallVector<llvm::StringRef> dims;
  value.split(dims, ',', -1, false);
  for (llvm::StringRef dim : dims) {
    int64_t parsed = 0;
    if (dim.trim().getAsInteger(10, parsed))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Invalid ABI shape dim: %s",
                                     dim.str().c_str());
    shape.push_back(parsed);
  }
  return shape;
}

static llvm::Expected<size_t> parseSizeValue(llvm::StringRef value,
                                             llvm::StringRef field) {
  uint64_t parsed = 0;
  if (value.getAsInteger(10, parsed))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Invalid %s: %s", field.str().c_str(),
                                   value.str().c_str());
  return static_cast<size_t>(parsed);
}

static llvm::Expected<AbiTensorDesc>
parseAbiTensor(const std::map<std::string, std::string> &manifest,
               llvm::StringRef prefix, size_t index) {
  const std::string base = (prefix + std::to_string(index)).str();
  auto getRequired = [&](llvm::StringRef suffix) -> llvm::Expected<std::string> {
    const std::string key = base + suffix.str();
    auto it = manifest.find(key);
    if (it == manifest.end() || it->second.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Missing ABI manifest key: %s",
                                     key.c_str());
    return it->second;
  };

  AbiTensorDesc tensor;
  auto nameOr = getRequired("_name");
  if (!nameOr)
    return nameOr.takeError();
  tensor.name = *nameOr;
  auto fileOr = getRequired("_file");
  if (!fileOr)
    return fileOr.takeError();
  tensor.file = *fileOr;
  auto dtypeOr = getRequired("_dtype");
  if (!dtypeOr)
    return dtypeOr.takeError();
  auto parsedDType = parseDType(*dtypeOr);
  if (!parsedDType)
    return parsedDType.takeError();
  tensor.dtype = *parsedDType;
  auto shapeOr = getRequired("_shape");
  if (!shapeOr)
    return shapeOr.takeError();
  auto parsedShape = parseShapeList(*shapeOr);
  if (!parsedShape)
    return parsedShape.takeError();
  tensor.shape = std::move(*parsedShape);
  return tensor;
}

static llvm::Expected<AbiMetadata>
parseAbiMetadata(const std::map<std::string, std::string> &manifest) {
  auto getRequired = [&](llvm::StringRef key) -> llvm::Expected<std::string> {
    auto it = manifest.find(key.str());
    if (it == manifest.end() || it->second.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Missing ABI manifest key: %s",
                                     key.str().c_str());
    return it->second;
  };

  AbiMetadata abi;
  auto inputCountOr = getRequired("abi_input_count");
  if (!inputCountOr)
    return inputCountOr.takeError();
  auto inputCount = parseSizeValue(*inputCountOr, "abi_input_count");
  if (!inputCount)
    return inputCount.takeError();
  for (size_t i = 0; i < *inputCount; ++i) {
    auto tensorOr = parseAbiTensor(manifest, "abi_input", i);
    if (!tensorOr)
      return tensorOr.takeError();
    abi.inputs.push_back(std::move(*tensorOr));
  }

  auto outputCountOr = getRequired("abi_output_count");
  if (!outputCountOr)
    return outputCountOr.takeError();
  auto outputCount = parseSizeValue(*outputCountOr, "abi_output_count");
  if (!outputCount)
    return outputCount.takeError();
  for (size_t i = 0; i < *outputCount; ++i) {
    auto tensorOr = parseAbiTensor(manifest, "abi_output", i);
    if (!tensorOr)
      return tensorOr.takeError();
    abi.outputs.push_back(std::move(*tensorOr));
  }

  auto workspaceBytesOr = getRequired("abi_workspace_bytes");
  if (!workspaceBytesOr)
    return workspaceBytesOr.takeError();
  auto workspaceBytes =
      parseSizeValue(*workspaceBytesOr, "abi_workspace_bytes");
  if (!workspaceBytes)
    return workspaceBytes.takeError();
  abi.workspaceBytes = *workspaceBytes;

  auto workspaceModeOr = getRequired("abi_workspace_mode");
  if (!workspaceModeOr)
    return workspaceModeOr.takeError();
  abi.workspaceMode = *workspaceModeOr;

  auto tilingModeOr = getRequired("abi_tiling_mode");
  if (!tilingModeOr)
    return tilingModeOr.takeError();
  abi.tilingMode = *tilingModeOr;

  auto tilingSourceOr = getRequired("abi_tiling_source");
  if (!tilingSourceOr)
    return tilingSourceOr.takeError();
  abi.tilingSource = *tilingSourceOr;
  return abi;
}

static std::string resolvePath(const std::string &base, const std::string &path) {
  if (path.empty())
    return path;
  if (llvm::sys::path::is_absolute(path))
    return path;
  if (llvm::sys::fs::exists(path)) {
    llvm::SmallString<256> abs(path);
    if (!llvm::sys::fs::make_absolute(abs))
      return abs.str().str();
  }
  llvm::SmallString<256> joined(base);
  llvm::sys::path::append(joined, path);
  return joined.str().str();
}

static std::string findAscendHome() {
  auto hasRuntimeLibs = [](const std::string &root) {
    if (root.empty())
      return false;
    const std::string libDirs[] = {
        root + "/lib64",
        root + "/aarch64-linux/lib64",
        root + "/arm64-linux/lib64",
    };
    for (const std::string &libDir : libDirs) {
      if (llvm::sys::fs::exists(libDir + "/libplatform.so") &&
          llvm::sys::fs::exists(libDir + "/libunified_dlog.so") &&
          llvm::sys::fs::exists(libDir + "/libmmpa.so") &&
          llvm::sys::fs::exists(libDir + "/libc_sec.so"))
        return true;
    }
    return false;
  };
  if (const char *ascendHome = std::getenv("ASCEND_HOME_PATH");
      ascendHome && *ascendHome && hasRuntimeLibs(ascendHome))
    return ascendHome;
  if (const char *toolkitHome = std::getenv("ASCEND_TOOLKIT_HOME");
      toolkitHome && *toolkitHome && hasRuntimeLibs(toolkitHome))
    return toolkitHome;
  if (const char *userHome = std::getenv("HOME")) {
    const std::string latest = std::string(userHome) + "/Ascend/latest";
    if (hasRuntimeLibs(latest))
      return latest;
    const std::string toolkitLatest =
        std::string(userHome) + "/Ascend/ascend-toolkit/latest";
    if (hasRuntimeLibs(toolkitLatest))
      return toolkitLatest;
  }
  const std::string sysDefault = "/usr/local/Ascend/ascend-toolkit/latest";
  if (hasRuntimeLibs(sysDefault))
    return sysDefault;
  return "";
}

static std::string findAscendLib64(const std::string &ascendHome) {
  const std::string candidates[] = {
      ascendHome + "/lib64",
      ascendHome + "/aarch64-linux/lib64",
      ascendHome + "/arm64-linux/lib64",
  };
  for (const std::string &candidate : candidates) {
    if (llvm::sys::fs::exists(candidate + "/libplatform.so") &&
        llvm::sys::fs::exists(candidate + "/libunified_dlog.so"))
      return candidate;
  }
  return "";
}

static std::string findSimulatorLibDir(const std::string &ascendHome,
                                       const std::string &socVersion) {
  const std::string candidates[] = {
      ascendHome + "/aarch64-linux/simulator/" + socVersion + "/lib",
      ascendHome + "/arm64-linux/simulator/" + socVersion + "/lib",
      ascendHome + "/tools/simulator/" + socVersion + "/lib",
  };
  for (const std::string &candidate : candidates) {
    if (llvm::sys::fs::exists(candidate + "/libruntime_camodel.so"))
      return candidate;
  }
  return "";
}

static std::string findDavSimulatorLibDir(const std::string &ascendHome) {
  const std::string candidates[] = {
      ascendHome + "/aarch64-linux/simulator/dav_3002/lib",
      ascendHome + "/arm64-linux/simulator/dav_3002/lib",
      ascendHome + "/tools/simulator/dav_3002/lib",
  };
  for (const std::string &candidate : candidates) {
    if (llvm::sys::fs::exists(candidate + "/libmodel_top.so"))
      return candidate;
  }
  return "";
}

static std::string findDeviceLibDir(const std::string &ascendHome) {
  const std::string candidates[] = {
      ascendHome + "/aarch64-linux/lib64/device/lib64",
      ascendHome + "/arm64-linux/lib64/device/lib64",
  };
  for (const std::string &candidate : candidates) {
    if (llvm::sys::fs::exists(candidate))
      return candidate;
  }
  return "";
}

static void prependEnvPath(const char *name, const std::string &prefix) {
  if (prefix.empty())
    return;
  const char *current = std::getenv(name);
  std::string value = prefix;
  if (current && *current) {
    value.push_back(':');
    value += current;
  }
  ::setenv(name, value.c_str(), 1);
}

static llvm::Error configureRuntimeEnv(const std::string &socVersion,
                                       const std::string &artifactRoot,
                                       const std::string &kernelSoPath,
                                       std::string *ascendHomeOut = nullptr,
                                       std::string *ascendLib64Out = nullptr,
                                       std::string *simLibDirOut = nullptr,
                                       std::string *davSimLibDirOut = nullptr,
                                       std::string *deviceLibDirOut = nullptr) {
  const std::string ascendHome = findAscendHome();
  if (ascendHome.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot find Ascend toolkit root; set ASCEND_HOME_PATH or "
        "ASCEND_TOOLKIT_HOME");
  const std::string ascendLib64 = findAscendLib64(ascendHome);
  if (ascendLib64.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot find Ascend lib64 under %s", ascendHome.c_str());
  const std::string simLibDir = findSimulatorLibDir(ascendHome, socVersion);
  if (simLibDir.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot find simulator libs for %s under %s", socVersion.c_str(),
        ascendHome.c_str());
  const std::string davSimLibDir = findDavSimulatorLibDir(ascendHome);
  if (davSimLibDir.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot find dav_3002 simulator libs under %s", ascendHome.c_str());
  const std::string deviceLibDir = findDeviceLibDir(ascendHome);
  if (deviceLibDir.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot find device libs under %s", ascendHome.c_str());
  if (ascendHomeOut)
    *ascendHomeOut = ascendHome;
  if (ascendLib64Out)
    *ascendLib64Out = ascendLib64;
  if (simLibDirOut)
    *simLibDirOut = simLibDir;
  if (davSimLibDirOut)
    *davSimLibDirOut = davSimLibDir;
  if (deviceLibDirOut)
    *deviceLibDirOut = deviceLibDir;
  ::setenv("ASCEND_HOME_PATH", ascendHome.c_str(), 1);
  ::setenv("ASCEND_TOOLKIT_HOME", ascendHome.c_str(), 1);
  ::setenv("SOC_VERSION", socVersion.c_str(), 1);

  llvm::SmallString<256> outDir(artifactRoot);
  llvm::sys::path::append(outDir, "out");
  prependEnvPath("LD_LIBRARY_PATH", outDir.str().str());
  prependEnvPath("LD_LIBRARY_PATH", ascendLib64);
  prependEnvPath("LD_LIBRARY_PATH", simLibDir);
  prependEnvPath("LD_LIBRARY_PATH", davSimLibDir);
  prependEnvPath("LD_LIBRARY_PATH", deviceLibDir);
  if (!kernelSoPath.empty()) {
    llvm::SmallString<256> kernelDir(kernelSoPath);
    llvm::sys::path::remove_filename(kernelDir);
    prependEnvPath("LD_LIBRARY_PATH", kernelDir.str().str());
  }
  return llvm::Error::success();
}

static llvm::Expected<NDArray> loadRawTensor(const std::string &path,
                                             std::vector<int64_t> shape,
                                             DType dtype) {
  auto bytesOr = readBinary(path);
  if (!bytesOr)
    return bytesOr.takeError();
  NDArray arr;
  arr.shape = std::move(shape);
  arr.dtype = dtype;
  if (arr.nbytes() != bytesOr->size())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Unexpected file size for %s: got %zu bytes, expected %zu bytes",
        path.c_str(), bytesOr->size(), arr.nbytes());
  arr.allocate();
  std::memcpy(arr.data, bytesOr->data(), bytesOr->size());
  return arr;
}

static llvm::Expected<std::vector<uint8_t>>
buildTilingFromAbi(const AbiMetadata &abi, const std::string &artifactRoot) {
  if (abi.tilingMode == "generated_file") {
    const std::string tilingPath = resolvePath(artifactRoot, abi.tilingSource);
    return readBinary(tilingPath);
  }
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "Unsupported ABI tiling description: mode=%s source=%s",
      abi.tilingMode.c_str(), abi.tilingSource.c_str());
}

static llvm::Error runShell(const std::string &cmd) {
  std::vector<std::string> args = {"/bin/bash", "-lc", cmd};
  std::vector<llvm::StringRef> argv;
  for (const auto &a : args)
    argv.push_back(a);
  std::string err_msg;
  int ret =
      llvm::sys::ExecuteAndWait(argv[0], argv, std::nullopt, {}, 300, 0,
                                      &err_msg);
  if (ret != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Process failed (exit %d): %s", ret,
                                   err_msg.c_str());
  return llvm::Error::success();
}

static llvm::Error preloadSharedLibrary(const std::string &path) {
  if (path.empty())
    return llvm::Error::success();
  void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!handle)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "dlopen failed (%s): %s", path.c_str(),
                                   dlerror());
  return llvm::Error::success();
}

static llvm::Error runDirectValidator(const std::string &artifactRoot,
                                      const std::string &goldenPath,
                                      const std::string &outputPath,
                                      const std::string &socVersion,
                                      const AbiMetadata &abi,
                                      std::vector<NDArray> &inputs,
                                      NDArray &outputArr,
                                      const std::string &kernelSoPath,
                                      const std::string &kernelName) {
  std::string ascendHome;
  std::string ascendLib64;
  std::string simLibDir;
  if (auto err = configureRuntimeEnv(socVersion, artifactRoot, kernelSoPath,
                                     &ascendHome, &ascendLib64, &simLibDir))
    return err;
  const std::vector<std::string> supportPreloads = {
      ascendLib64 + "/libc_sec.so",
      ascendLib64 + "/libmmpa.so",
      ascendLib64 + "/libascend_protobuf.so",
      ascendLib64 + "/libunified_dlog.so",
      ascendLib64 + "/libascend_watchdog.so",
      ascendLib64 + "/libplatform.so",
  };
  for (const std::string &path : supportPreloads)
    if (llvm::sys::fs::exists(path))
      (void)preloadSharedLibrary(path);

  const std::vector<std::string> simPreloads = {
      simLibDir + "/libffts_model.so",
      simLibDir + "/libstars.so",
      simLibDir + "/libmodel_top.so",
      simLibDir + "/libesl_top_wrapper.so",
      simLibDir + "/libmcu_wrapper.so",
      simLibDir + "/libmcu_loop.so",
      simLibDir + "/libpem_davinci.so",
      simLibDir + "/libnpu_drv.so",
      simLibDir + "/libnpu_drv_camodel.so",
      simLibDir + "/libruntime_camodel.so",
  };
  for (const std::string &path : simPreloads)
    if (llvm::sys::fs::exists(path))
      (void)preloadSharedLibrary(path);

  RunArgs args;
  args.block_dim      = 1;
  args.workspace_size = abi.workspaceBytes;
  auto tilingOr = buildTilingFromAbi(abi, artifactRoot);
  if (!tilingOr)
    return tilingOr.takeError();
  args.tiling         = std::move(*tilingOr);
  args.inputs         = std::move(inputs);
  args.outputs.push_back(std::move(outputArr));

  Executor executor(BackendMode::Simulation);
  if (auto err = executor.Initialize())
    return err;
  if (auto err = executor.RunPackedMixFile(kernelSoPath, kernelName, args))
    return err;

  if (auto err = writeBinary(
          outputPath,
          llvm::ArrayRef<uint8_t>(
              reinterpret_cast<const uint8_t *>(args.outputs[0].data),
              args.outputs[0].nbytes())))
    return err;

  auto actual = readBinary(outputPath);
  if (!actual)
    return actual.takeError();
  auto golden = readBinary(goldenPath);
  if (!golden)
    return golden.takeError();
  if (actual->size() != golden->size()) {
    llvm::errs() << "FAIL size mismatch: actual=" << actual->size()
                 << " golden=" << golden->size() << "\n";
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Direct packed validator size mismatch");
  }

  double maxAbsDiff = 0.0;
  double meanAbsDiff = 0.0;
  size_t count = actual->size() / sizeof(float);
  for (size_t i = 0; i < count; ++i) {
    float a = 0.0f;
    float g = 0.0f;
    std::memcpy(&a, actual->data() + i * sizeof(float), sizeof(float));
    std::memcpy(&g, golden->data() + i * sizeof(float), sizeof(float));
    double diff = std::abs(static_cast<double>(a) - static_cast<double>(g));
    maxAbsDiff = std::max(maxAbsDiff, diff);
    meanAbsDiff += diff;
  }
  if (count > 0)
    meanAbsDiff /= static_cast<double>(count);

  llvm::outs() << "max_abs_diff=" << maxAbsDiff << "\n";
  llvm::outs() << "mean_abs_diff=" << meanAbsDiff << "\n";
  llvm::outs() << (maxAbsDiff == 0.0 ? "PASS" : "FAIL") << "\n";
  return maxAbsDiff == 0.0
             ? llvm::Error::success()
             : llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Direct packed validator output mismatch");
}

static std::string buildInputPath(const std::string &inputDir,
                                  const std::string &name) {
  llvm::SmallString<256> path(inputDir);
  llvm::sys::path::append(path, name);
  return path.str().str();
}

static std::string shellQuote(llvm::StringRef value) {
  std::string quoted = "'";
  for (char c : value) {
    if (c == '\'')
      quoted += "'\\''";
    else
      quoted.push_back(c);
  }
  quoted.push_back('\'');
  return quoted;
}

static llvm::Error runDirectValidatorChildProcess(const std::string &selfPath,
                                                  const std::string &artifactRoot,
                                                  const std::string &inputDir,
                                                  const std::string &golden,
                                                  const std::string &outputPath,
                                                  const std::string &socVersion,
                                                  bool forceDirectPacked) {
  std::string ascendHome;
  std::string ascendLib64;
  std::string simLibDir;
  std::string davSimLibDir;
  std::string deviceLibDir;
  if (auto err = configureRuntimeEnv(socVersion, artifactRoot, "",
                                     &ascendHome, &ascendLib64, &simLibDir,
                                     &davSimLibDir, &deviceLibDir))
    return err;
  const std::string baseLdLibraryPath =
      ascendLib64 + ":" + deviceLibDir + ":" + simLibDir + ":" + davSimLibDir;
  std::string cmd;
  cmd += "export RUNTIMEMIX_DIRECT_CHILD=1 ";
  cmd += "ASCEND_HOME_PATH=" + shellQuote(ascendHome) + " ";
  cmd += "ASCEND_TOOLKIT_HOME=" + shellQuote(ascendHome) + " ";
  cmd += "SOC_VERSION=" + shellQuote(socVersion) + " ";
  cmd += "LD_LIBRARY_PATH=" + shellQuote(baseLdLibraryPath) +
         ":${LD_LIBRARY_PATH:-}; ";
  cmd += shellQuote(selfPath);
  cmd += " --artifact-root " + shellQuote(artifactRoot);
  cmd += " --input-dir " + shellQuote(inputDir);
  cmd += " --golden " + shellQuote(golden);
  cmd += " --output-file " + shellQuote(outputPath);
  cmd += " --soc " + shellQuote(socVersion);
  if (forceDirectPacked)
    cmd += " --force-direct-packed";
  return runShell(cmd);
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "RuntimeMix mix validator\n");

  llvm::SmallString<256> artifactRoot(ArtifactRoot);
  llvm::sys::fs::make_absolute(artifactRoot);
  llvm::SmallString<256> manifestPath(artifactRoot);
  llvm::sys::path::append(manifestPath, "out", "manifest.txt");

  const auto manifest = readManifest(manifestPath.str().str());
  auto abiOr = parseAbiMetadata(manifest);
  if (!abiOr) {
    llvm::errs() << "Error parsing ABI metadata: "
                 << llvm::toString(abiOr.takeError()) << "\n";
    return 4;
  }
  const AbiMetadata abi = std::move(*abiOr);

  std::string outputPath = OutputFile;
  if (outputPath.empty()) {
    if (abi.outputs.empty() || abi.outputs[0].file.empty()) {
      llvm::errs() << "Error: missing ABI output file name in manifest\n";
      return 4;
    }
    const std::string outputName = abi.outputs[0].file;
    outputPath = resolvePath(artifactRoot.str().str(), outputName);
  }

  const std::string manifestKernelName =
      manifest.count("kernel_name") ? manifest.at("kernel_name") : "";
  const std::string manifestKernelSo = resolvePath(
      artifactRoot.str().str(),
      manifest.count("kernel_so_path") ? manifest.at("kernel_so_path") : "");
  const std::string manifestRunner = resolvePath(
      artifactRoot.str().str(),
      manifest.count("host_runner_path") ? manifest.at("host_runner_path")
                                          : "");

  llvm::SmallString<256> runnerPath(artifactRoot);
  if (!manifestRunner.empty())
    runnerPath = llvm::StringRef(manifestRunner);
  else
    llvm::sys::path::append(runnerPath, "out", "bin", "mix_runner");
  if (!llvm::sys::fs::exists(runnerPath)) {
    runnerPath = artifactRoot;
    llvm::sys::path::append(runnerPath, "bin", "mix_runner");
  }
  const bool canUseRunner = llvm::sys::fs::exists(runnerPath) &&
                            !ForceDirectPacked;
  // Runner-first is the default contract. The direct packed path remains
  // available as a fallback or explicit comparison path when the artifact
  // provides a kernel .so and ABI metadata.
  const bool canUseDirectPacked = !manifestKernelName.empty() &&
                                  !manifestKernelSo.empty() &&
                                  llvm::sys::fs::exists(manifestKernelSo) &&
                                  !canUseRunner;
  const bool isDirectChild =
      std::getenv("RUNTIMEMIX_DIRECT_CHILD") != nullptr;

  if (canUseDirectPacked) {
    std::vector<NDArray> inputs;
    for (const AbiTensorDesc &tensor : abi.inputs) {
      auto inputOr = loadRawTensor(buildInputPath(InputDir, tensor.file),
                                   tensor.shape, tensor.dtype);
      if (!inputOr) {
        llvm::errs() << "Error loading input " << tensor.file << ": "
                     << llvm::toString(inputOr.takeError()) << "\n";
        return 4;
      }
      inputs.push_back(std::move(*inputOr));
    }

    auto goldenOr = readBinary(Golden);
    if (!goldenOr) {
      llvm::errs() << llvm::toString(goldenOr.takeError()) << "\n";
      return 4;
    }
    if (abi.outputs.size() != 1) {
      llvm::errs() << "Error: direct packed fallback only supports a single "
                      "output in the current ABI surface\n";
      return 4;
    }
    NDArray output;
    output.shape = abi.outputs[0].shape;
    output.dtype = abi.outputs[0].dtype;
    output.allocate();
    if (output.nbytes() != goldenOr->size()) {
      llvm::errs() << "Error: golden size mismatch, expected output bytes="
                   << output.nbytes() << " got=" << goldenOr->size() << "\n";
      return 4;
    }

    llvm::SmallString<256> selfPath(argv[0]);
    llvm::sys::fs::make_absolute(selfPath);
    llvm::Error directErr = llvm::Error::success();
    if (!isDirectChild) {
      directErr = runDirectValidatorChildProcess(
          selfPath.str().str(), artifactRoot.str().str(), InputDir, Golden,
          outputPath, SocVersion, ForceDirectPacked);
    } else {
      directErr = runDirectValidator(artifactRoot.str().str(), Golden,
                                     outputPath, SocVersion, abi, inputs, output,
                                     manifestKernelSo, manifestKernelName);
      if (!directErr) {
        llvm::outs().flush();
        llvm::errs().flush();
        _Exit(0);
      }
    }

    if (directErr) {
      llvm::errs() << "Warning: constrained direct packed fallback failed: "
                   << llvm::toString(std::move(directErr))
                   << "\nContinuing with the runner-first path.\n";
    } else {
      return 0;
    }
  }

  if (!canUseRunner) {
    llvm::errs() << "Error: runner not found under " << artifactRoot << "\n";
    return 4;
  }

  std::string ascendHome;
  std::string ascendLib64;
  std::string socSimLibDir;
  std::string davSimLibDir;
  std::string deviceLibDir;
  if (auto err = configureRuntimeEnv(SocVersion, artifactRoot.str().str(), "",
                                     &ascendHome, &ascendLib64, &socSimLibDir,
                                     &davSimLibDir, &deviceLibDir)) {
    llvm::errs() << "Error: " << llvm::toString(std::move(err)) << "\n";
    return 4;
  }
  llvm::SmallString<256> installDir(artifactRoot);
  llvm::sys::path::append(installDir, "out");
  if (!llvm::sys::fs::exists(installDir))
    installDir = artifactRoot;
  std::string cmd = llvm::formatv(
      "source \"{0}/bin/setenv.bash\" >/dev/null 2>&1 && "
      "export ASCEND_HOME_PATH=\"{0}\" ASCEND_TOOLKIT_HOME=\"{0}\" && "
      "export LD_LIBRARY_PATH=\"{1}:{2}:{3}:{4}:{5}:$LD_LIBRARY_PATH\" && "
      "\"{6}\" --input-dir \"{7}\" --output-file \"{8}\"",
      ascendHome, installDir.str(), ascendLib64, socSimLibDir, davSimLibDir,
      deviceLibDir, runnerPath.str(), InputDir, outputPath)
                        .str();
  if (auto err = runShell(cmd)) {
    llvm::errs() << "Run error: " << llvm::toString(std::move(err)) << "\n";
    return 2;
  }

  auto actual = readBinary(outputPath);
  if (!actual) {
    llvm::errs() << llvm::toString(actual.takeError()) << "\n";
    return 2;
  }
  auto golden = readBinary(Golden);
  if (!golden) {
    llvm::errs() << llvm::toString(golden.takeError()) << "\n";
    return 2;
  }
  if (actual->size() != golden->size()) {
    llvm::errs() << "FAIL size mismatch: actual=" << actual->size()
                 << " golden=" << golden->size() << "\n";
    return 1;
  }

  double maxAbsDiff = 0.0;
  double meanAbsDiff = 0.0;
  size_t count = actual->size() / sizeof(float);
  for (size_t i = 0; i < count; ++i) {
    float a = 0.0f;
    float g = 0.0f;
    std::memcpy(&a, actual->data() + i * sizeof(float), sizeof(float));
    std::memcpy(&g, golden->data() + i * sizeof(float), sizeof(float));
    double diff = std::abs(static_cast<double>(a) - static_cast<double>(g));
    maxAbsDiff = std::max(maxAbsDiff, diff);
    meanAbsDiff += diff;
  }
  if (count > 0)
    meanAbsDiff /= static_cast<double>(count);

  bool passed = maxAbsDiff == 0.0;
  llvm::outs() << "max_abs_diff=" << maxAbsDiff << "\n";
  llvm::outs() << "mean_abs_diff=" << meanAbsDiff << "\n";
  llvm::outs() << (passed ? "PASS" : "FAIL") << "\n";
  return passed ? 0 : 1;
}
