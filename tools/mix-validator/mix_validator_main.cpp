#include "Runtime/Executor.h"
#include "Runtime/MixAbi.h"
#include "Runtime/PathUtils.h"
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
using AbiTensorDesc = MixAbiTensorDesc;
using AbiMetadata = MixAbiMetadata;

static llvm::cl::opt<std::string> ArtifactRoot("artifact-root",
                                               llvm::cl::Required);
static llvm::cl::opt<std::string> InputDir("input-dir", llvm::cl::Required);
static llvm::cl::opt<std::string> Golden("golden", llvm::cl::init(""));
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

static bool hasAllFiles(const std::string &dir,
                        llvm::ArrayRef<llvm::StringRef> files) {
  if (dir.empty())
    return false;
  for (llvm::StringRef file : files) {
    if (!llvm::sys::fs::exists((llvm::Twine(dir) + "/" + file).str()))
      return false;
  }
  return true;
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
  const std::string ascendLib64 = findAscendLib64Dir(ascendHome);
  if (ascendHome.empty() ||
      !hasAllFiles(ascendLib64,
                   {"libplatform.so", "libunified_dlog.so", "libmmpa.so",
                    "libc_sec.so"}))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot find Ascend toolkit root; set ASCEND_HOME_PATH or "
        "ASCEND_TOOLKIT_HOME");
  if (!hasAllFiles(ascendLib64, {"libplatform.so", "libunified_dlog.so"}))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot find Ascend lib64 under %s", ascendHome.c_str());
  const std::string simLibDir = findAscendSimulatorLibDir(ascendHome, socVersion);
  if (!hasAllFiles(simLibDir, {"libruntime_camodel.so"}))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot find simulator libs for %s under %s", socVersion.c_str(),
        ascendHome.c_str());
  auto davSimLibDirOr = requireAscendDavSimulatorLibDir(ascendHome);
  if (!davSimLibDirOr)
    return davSimLibDirOr.takeError();
  const std::string davSimLibDir = *davSimLibDirOr;
  const std::string deviceLibDir = findAscendDeviceLibDir(ascendHome);
  if (!llvm::sys::fs::exists(deviceLibDir))
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

static void tryPreloadSharedLibrary(const std::string &path) {
  if (auto err = preloadSharedLibrary(path))
    llvm::errs() << "Warning: " << llvm::toString(std::move(err)) << "\n";
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
  llvm::errs() << "[direct] start\n";
  std::string ascendHome;
  std::string ascendLib64;
  std::string simLibDir;
  if (auto err = configureRuntimeEnv(socVersion, artifactRoot, kernelSoPath,
                                     &ascendHome, &ascendLib64, &simLibDir))
    return err;
  llvm::errs() << "[direct] env configured\n";
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
      tryPreloadSharedLibrary(path);

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
      tryPreloadSharedLibrary(path);
  llvm::errs() << "[direct] preload done\n";

  RunArgs args;
  args.block_dim      = static_cast<int>(abi.blockDim);
  args.workspace_size = abi.workspaceBytes;
  auto tilingOr = buildTilingFromAbi(abi, artifactRoot);
  if (!tilingOr)
    return tilingOr.takeError();
  args.tiling         = std::move(*tilingOr);
  args.inputs         = std::move(inputs);
  args.outputs.push_back(std::move(outputArr));
  llvm::errs() << "[direct] args ready\n";

  Executor executor(BackendMode::Simulation);
  if (auto err = executor.Initialize())
    return err;
  llvm::errs() << "[direct] executor initialized\n";
  if (auto err = executor.RunPackedMixFile(kernelSoPath, kernelName, args))
    return err;
  llvm::errs() << "[direct] packed run returned\n";

  if (auto err = writeBinary(
          outputPath,
          llvm::ArrayRef<uint8_t>(
              reinterpret_cast<const uint8_t *>(args.outputs[0].data),
              args.outputs[0].nbytes())))
    return err;
  llvm::errs() << "[direct] output written\n";

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

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "RuntimeMix mix validator\n");

  llvm::SmallString<256> artifactRoot(ArtifactRoot);
  llvm::sys::fs::make_absolute(artifactRoot);
  llvm::SmallString<256> manifestPath(artifactRoot);
  llvm::sys::path::append(manifestPath, "out", "manifest.txt");

  const auto manifest = readManifest(manifestPath.str().str());
  auto abiOr = parseMixAbiManifest(manifest);
  if (!abiOr) {
    llvm::errs() << "Error parsing ABI metadata: "
                 << llvm::toString(abiOr.takeError()) << "\n";
    return 4;
  }
  const AbiMetadata abi = std::move(*abiOr);

  std::string outputPath = OutputFile;
  if (outputPath.empty()) {
    if (abi.outputs.empty() || abi.outputs[0].runtimeFile.empty()) {
      llvm::errs() << "Error: missing ABI output file name in manifest\n";
      return 4;
    }
    const std::string outputName = abi.outputs[0].runtimeFile;
    outputPath = resolvePath(artifactRoot.str().str(), outputName);
  }

  std::string goldenPath = Golden;
  if (goldenPath.empty()) {
    if (abi.outputs.empty() || abi.outputs[0].name.empty()) {
      llvm::errs() << "Error: missing ABI golden file name in manifest\n";
      return 4;
    }
    const std::string goldenName =
        !abi.outputs[0].goldenFile.empty()
            ? abi.outputs[0].goldenFile
            : buildCanonicalGoldenFileName(
                  !abi.runtimeKernelName.empty() ? abi.runtimeKernelName
                                                 : abi.logicalKernelName,
                  abi.outputs[0].name);
    goldenPath = resolvePath(artifactRoot.str().str(), goldenName);
  }

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
  const bool canUseDirectPacked = !abi.runtimeKernelName.empty() &&
                                  !manifestKernelSo.empty() &&
                                  llvm::sys::fs::exists(manifestKernelSo) &&
                                  !canUseRunner;

  if (canUseDirectPacked) {
    std::vector<NDArray> inputs;
    for (const AbiTensorDesc &tensor : abi.inputs) {
      auto inputOr = loadRawTensor(buildInputPath(InputDir, tensor.runtimeFile),
                                   tensor.shape, tensor.dtype);
      if (!inputOr) {
        llvm::errs() << "Error loading input " << tensor.runtimeFile << ": "
                     << llvm::toString(inputOr.takeError()) << "\n";
        return 4;
      }
      inputs.push_back(std::move(*inputOr));
    }

    auto goldenOr = readBinary(goldenPath);
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

    llvm::Error directErr = runDirectValidator(
        artifactRoot.str().str(), goldenPath, outputPath, SocVersion, abi, inputs,
        output, manifestKernelSo, abi.runtimeKernelName);

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
  auto golden = readBinary(goldenPath);
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
