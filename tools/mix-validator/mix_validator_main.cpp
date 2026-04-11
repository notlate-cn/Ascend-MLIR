#include "Runtime/ExecutionSession.h"
#include "Runtime/MixAbi.h"
#include "Runtime/PathUtils.h"
#include "Runtime/TaskGraph.h"
#include "Runtime/Types.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
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
                                             llvm::cl::init(""));
static llvm::cl::opt<bool> ForceDirectPacked(
    "force-direct-packed",
    llvm::cl::desc(
        "Bypass runtime-session execution and validate through direct packed execution"),
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

static std::string buildInputPath(const std::string &inputDir,
                                  const std::string &name) {
  llvm::SmallString<256> path(inputDir);
  llvm::sys::path::append(path, name);
  return path.str().str();
}

static llvm::Expected<RuntimeTask>
buildValidationTask(const std::string &artifactRoot,
                    const std::string &manifestPath,
                    const std::string &kernelSoPath,
                    const std::string &deviceBinaryPath,
                    const std::string &kernelName,
                    const std::string &socVersion,
                    const AbiMetadata &abi,
                    llvm::StringRef inputDir,
                    llvm::StringRef outputPath,
                    llvm::StringRef tilingBinaryPath) {
  if (abi.outputs.empty() || abi.outputs[0].runtimeFile.empty()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "missing ABI output metadata for mix validation");
  }
  if (abi.inputs.empty()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "missing ABI input metadata for mix validation");
  }

  KernelArtifact artifact;
  artifact.kernelName = kernelName;
  artifact.kernelKind = KernelKind::Mix;
  artifact.mixResourceType = MixResourceType::Unknown;
  artifact.socVersion = socVersion;
  artifact.artifactRoot = artifactRoot;
  artifact.manifestPath = manifestPath;
  const std::string packedSharedObjectPath =
      !kernelSoPath.empty() ? kernelSoPath : deviceBinaryPath;
  if (packedSharedObjectPath.empty()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "mix artifact is missing packed shared object path");
  }
  artifact.packedSharedObjectPath = packedSharedObjectPath;
  artifact.deviceBinaryPath = !deviceBinaryPath.empty() ? deviceBinaryPath
                                                       : packedSharedObjectPath;

  RuntimeTask task;
  task.taskId = "main";
  task.artifact = std::move(artifact);

  for (const AbiTensorDesc &tensor : abi.inputs) {
    TensorBinding binding;
    binding.name = tensor.name;
    binding.sourceKind = BindingSourceKind::ExternalFile;
    binding.path = buildInputPath(inputDir.str(), tensor.runtimeFile);
    task.invocation.inputs.push_back(std::move(binding));
  }

  TensorBinding output;
  output.name = abi.outputs[0].name.empty() ? "out" : abi.outputs[0].name;
  output.sourceKind = BindingSourceKind::ExternalFile;
  output.path = outputPath.str();
  output.shape = abi.outputs[0].shape;
  output.dtype = abi.outputs[0].dtype;
  task.invocation.outputs.push_back(std::move(output));

  if (!tilingBinaryPath.empty()) {
    TilingBinding tiling;
    tiling.binaryPath = tilingBinaryPath.str();
    task.invocation.tiling = std::move(tiling);
  }

  task.invocation.blockDim = static_cast<int>(abi.blockDim);
  task.invocation.workspaceSize = abi.workspaceBytes;
  return task;
}

static llvm::Expected<TaskGraph>
buildValidationGraph(const std::string &artifactRoot,
                     const std::string &manifestPath,
                     const std::string &kernelSoPath,
                     const std::string &deviceBinaryPath,
                     const std::string &kernelName,
                     const std::string &socVersion,
                     const AbiMetadata &abi,
                     llvm::StringRef inputDir,
                     llvm::StringRef outputPath,
                     llvm::StringRef tilingBinaryPath) {
  auto taskOr = buildValidationTask(artifactRoot, manifestPath, kernelSoPath,
                                    deviceBinaryPath, kernelName, socVersion,
                                    abi, inputDir, outputPath, tilingBinaryPath);
  if (!taskOr)
    return taskOr.takeError();

  TaskGraph graph;
  if (auto err = graph.addTask(*taskOr))
    return std::move(err);
  return graph;
}

static llvm::Error compareOutputs(llvm::StringRef actualPath,
                                  llvm::StringRef goldenPath) {
  auto actual = readBinary(actualPath.str());
  if (!actual)
    return actual.takeError();
  auto golden = readBinary(goldenPath.str());
  if (!golden)
    return golden.takeError();
  if (actual->size() != golden->size()) {
    llvm::errs() << "FAIL size mismatch: actual=" << actual->size()
                 << " golden=" << golden->size() << "\n";
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "mix validator output size mismatch");
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
                                       "mix validator output mismatch");
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "RuntimeMix artifact validator for mix kernels\n");
  const std::string resolvedSocVersion =
      resolveSocVersion(SocVersion, "Ascend910B1");
  (void)ForceDirectPacked;

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

  const std::string manifestKernelSo = resolvePath(
      artifactRoot.str().str(),
      manifest.count("kernel_so_path") ? manifest.at("kernel_so_path") : "");
  const std::string manifestDeviceBinary = resolvePath(
      artifactRoot.str().str(),
      manifest.count("device_binary_path")
          ? manifest.at("device_binary_path")
          : (manifest.count("device_object_path") ? manifest.at("device_object_path")
                                                  : ""));
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

  if (auto err = configureRuntimeEnv(resolvedSocVersion, artifactRoot.str().str(),
                                     manifestKernelSo)) {
    llvm::errs() << "Error: " << llvm::toString(std::move(err)) << "\n";
    return 4;
  }

  auto manifestTilingOr = buildTilingFromAbi(abi, artifactRoot.str().str());
  if (!manifestTilingOr) {
    llvm::errs() << "Error loading tiling bytes: "
                 << llvm::toString(manifestTilingOr.takeError()) << "\n";
    return 4;
  }

  const std::string kernelName =
      !abi.runtimeKernelName.empty() ? abi.runtimeKernelName
                                     : abi.logicalKernelName;
  auto graphOr = buildValidationGraph(
      artifactRoot.str().str(), manifestPath.str().str(), manifestKernelSo,
      manifestDeviceBinary, kernelName, resolvedSocVersion, abi, InputDir,
      outputPath, *manifestTilingOr);
  if (!graphOr) {
    llvm::errs() << "Error building runtime task graph: "
                 << llvm::toString(graphOr.takeError()) << "\n";
    return 4;
  }

  for (const AbiTensorDesc &tensor : abi.inputs) {
    auto inputOr = loadRawTensor(buildInputPath(InputDir, tensor.runtimeFile),
                                 tensor.shape, tensor.dtype);
    if (!inputOr) {
      llvm::errs() << "Error loading input " << tensor.runtimeFile << ": "
                   << llvm::toString(inputOr.takeError()) << "\n";
      return 4;
    }
  }

  ExecutionSession session(ExecutionBackendKind::Simulation);
  auto traceOr = session.run(*graphOr);
  if (!traceOr) {
    llvm::errs() << "Error: " << llvm::toString(traceOr.takeError()) << "\n";
    return 2;
  }

  if (auto err = compareOutputs(outputPath, goldenPath)) {
    llvm::errs() << "Error: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }
  return 0;
}
