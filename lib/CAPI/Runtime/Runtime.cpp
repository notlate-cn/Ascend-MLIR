// lib/CAPI/Runtime/Runtime.cpp
// C API implementation wrapping runtime compatibility entrypoints.

#include "CAPI/Runtime.h"
#include "Runtime/ArtifactCompiler.h"
#include "Runtime/ExecutionSession.h"
#include "Runtime/RuntimeFrontendCore.h"
#include "Runtime/NpyIO.h"
#include "Runtime/Types.h"
#include "Runtime/TaskGraph.h"
#include "Runtime/TilingPack.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <cstring>
#include <utility>

using namespace mlir::runtime;

namespace {

struct CApiCompilerHandle {
  std::string socVersion;
  std::string arch = "dav-c220-vec";
  int optLevel = 3;
};

struct CApiExecutorHandle {
  int deviceId = 0;
  bool initialized = false;
};

static constexpr uint32_t kMagicAIVec = 0x41415246u;
static constexpr uint32_t kMagicAICube = 0x41494343u;

llvm::StringRef compatKernelTypeForArch(llvm::StringRef arch) {
  return arch.contains_insensitive("cube") ? "cube" : "vec";
}

std::string selectCompiledArtifactPath(const KernelArtifact &artifact) {
  if (!artifact.deviceBinaryPath.empty())
    return artifact.deviceBinaryPath;
  if (!artifact.sharedLibraryPath.empty())
    return artifact.sharedLibraryPath;
  return artifact.artifactRoot;
}

llvm::Expected<std::string> makeTemporaryDirectory(llvm::StringRef prefix) {
  std::error_code ec;
  const std::filesystem::path tempRoot = std::filesystem::temp_directory_path(ec);
  if (ec) {
    return llvm::createStringError(ec,
                                   "cannot determine temp directory for C API runtime");
  }

  llvm::SmallString<256> directoryPrefix(tempRoot.string());
  llvm::sys::path::append(directoryPrefix, prefix);

  llvm::SmallString<256> directory;
  if (auto createDirError =
          llvm::sys::fs::createUniqueDirectory(directoryPrefix, directory)) {
    return llvm::createStringError(createDirError,
                                   "cannot create temporary C API runtime directory");
  }
  return directory.str().str();
}

std::string joinPath(llvm::StringRef parent, llvm::StringRef leaf) {
  llvm::SmallString<256> path(parent);
  llvm::sys::path::append(path, leaf);
  return path.str().str();
}

llvm::Error writeBinaryFile(llvm::StringRef path, llvm::ArrayRef<uint8_t> bytes) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    return llvm::createStringError(ec, "cannot open %s for writing",
                                   path.str().c_str());
  }
  os.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  os.close();
  if (ec) {
    return llvm::createStringError(ec, "cannot write %s", path.str().c_str());
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<uint8_t>> readBinaryFile(llvm::StringRef path) {
  std::ifstream is(path.str(), std::ios::binary);
  if (!is) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot open %s", path.str().c_str());
  }
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(is)),
                             std::istreambuf_iterator<char>());
  if (!is.good() && !is.eof()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed reading %s", path.str().c_str());
  }
  return bytes;
}

llvm::Expected<KernelKind> kernelKindForMagic(uint32_t magic) {
  if (magic == kMagicAIVec)
    return KernelKind::Vec;
  if (magic == kMagicAICube)
    return KernelKind::Cube;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported kernel magic: 0x%08x", magic);
}

void fillArray(NDArray& arr, void* data, size_t bytes) {
  arr.data = data;
  if (bytes % 2 == 0) {
    arr.dtype = DType::F16;
    arr.shape = {static_cast<int64_t>(bytes / 2)};
  } else if (bytes % 4 == 0) {
    arr.dtype = DType::INT32;
    arr.shape = {static_cast<int64_t>(bytes / 4)};
  } else {
    arr.dtype = DType::F16;
    arr.shape = {static_cast<int64_t>((bytes + 1) / 2)};
  }
}

llvm::Error materializeInputNpy(const std::string &path,
                                const void *data, size_t bytes) {
  NDArray arr;
  fillArray(arr, const_cast<void *>(data), bytes);
  return SaveNpy(path, arr);
}

llvm::Error loadOutputNpy(const std::string &path, void *dst, size_t bytes) {
  auto arrayOr = LoadNpy(path);
  if (!arrayOr)
    return arrayOr.takeError();
  if (arrayOr->nbytes() != bytes) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "output byte size mismatch for %s: expected %zu, got %zu",
        path.c_str(), bytes, arrayOr->nbytes());
  }
  if (bytes > 0 && arrayOr->data)
    std::memcpy(dst, arrayOr->data, bytes);
  return llvm::Error::success();
}

llvm::Expected<std::string> materializeTilingBinary(
    llvm::StringRef tempDir, const uint8_t *tiling_data, size_t tiling_len) {
  if (!tiling_data || tiling_len == 0)
    return std::string{};
  const std::string tilingPath = joinPath(tempDir, "tiling.bin");
  if (auto err = writeBinaryFile(
          tilingPath, llvm::ArrayRef<uint8_t>(tiling_data, tiling_len)))
    return err;
  return tilingPath;
}

llvm::Error runWithExecutionSession(const std::string &binaryPath,
                                    const std::string &functionName,
                                    uint32_t magic, int num_inputs,
                                    const void **input_ptrs,
                                    const size_t *input_bytes, int num_outputs,
                                    void **output_ptrs,
                                    const size_t *output_bytes,
                                    const uint8_t *tiling_data,
                                    size_t tiling_len, int block_dim,
                                    int /*device_id*/) {
  if (functionName.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "function name is required");
  }
  if (num_inputs < 0 || num_outputs <= 0) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid input/output binding counts");
  }
  if ((num_inputs > 0 && (!input_ptrs || !input_bytes)) ||
      !output_ptrs || !output_bytes) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "input/output pointer arrays are required");
  }

  auto tempDirOr = makeTemporaryDirectory("afirt-runtime");
  if (!tempDirOr)
    return tempDirOr.takeError();
  const std::string tempDir = *tempDirOr;

  struct TempDirectoryCleanup {
    explicit TempDirectoryCleanup(std::string path) : path(std::move(path)) {}
    ~TempDirectoryCleanup() {
      if (!path.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
      }
    }
    std::string path;
  } cleanup(tempDir);

  auto kernelKindOr = kernelKindForMagic(magic);
  if (!kernelKindOr)
    return kernelKindOr.takeError();

  auto tilingPathOr = materializeTilingBinary(tempDir, tiling_data, tiling_len);
  if (!tilingPathOr)
    return tilingPathOr.takeError();

  KernelArtifact artifact;
  artifact.kernelName = functionName;
  artifact.kernelKind = *kernelKindOr;
  artifact.artifactRoot = tempDir;
  artifact.deviceBinaryPath = binaryPath;

  ExecutionInvocation invocation;
  invocation.blockDim = block_dim;

  if (!tilingPathOr->empty()) {
    TilingBinding tiling;
    tiling.binaryPath = *tilingPathOr;
    invocation.tiling = std::move(tiling);
  }

  for (int i = 0; i < num_inputs; ++i) {
    TensorBinding binding;
    binding.name = "data" + std::to_string(i);
    binding.sourceKind = BindingSourceKind::ExternalFile;
    binding.path = joinPath(tempDir, "input" + std::to_string(i) + ".npy");
    if (auto err = materializeInputNpy(binding.path, input_ptrs[i],
                                       input_bytes[i]))
      return err;
    invocation.inputs.push_back(std::move(binding));
  }

  for (int i = 0; i < num_outputs; ++i) {
    TensorBinding binding;
    binding.name = "out";
    binding.sourceKind = BindingSourceKind::ExternalFile;
    binding.path = joinPath(tempDir, "output" + std::to_string(i) + ".npy");
    NDArray shapeProbe;
    fillArray(shapeProbe, nullptr, output_bytes[i]);
    binding.shape = shapeProbe.shape;
    binding.dtype = shapeProbe.dtype;
    invocation.outputs.push_back(std::move(binding));
  }

  FrontendSingleTaskRunRequest request;
  request.backendKind = ExecutionBackendKind::Simulation;
  request.taskId = "main";
  request.artifact = artifact;
  request.invocation = std::move(invocation);

  auto preparedRunOr = prepareFrontendSingleTaskRun(request);
  if (!preparedRunOr)
    return preparedRunOr.takeError();

  ExecutionSession session(preparedRunOr->backendKind);
  FrontendRunSummary summary =
      executeFrontendPreparedRun(session, *preparedRunOr);
  if (!summary.success) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                   summary.rawErrorMessage.c_str());
  }

  for (int i = 0; i < num_outputs; ++i) {
    if (auto err = loadOutputNpy(request.invocation.outputs[i].path,
                                 output_ptrs[i], output_bytes[i]))
      return err;
  }

  return llvm::Error::success();
}

} // namespace

// ============================================================
// Helpers
// ============================================================

static void writeErr(char *buf, size_t len, const std::string &msg) {
  if (!buf || len == 0)
    return;
  strncpy(buf, msg.c_str(), len - 1);
  buf[len - 1] = '\0';
}

static void writeErr(char *buf, size_t len, llvm::Error err) {
  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << err;
  writeErr(buf, len, os.str());
}

// ============================================================
// Compiler
// ============================================================

AfirtCompiler afirt_compiler_create(const char *soc_version,
                                    const char *arch,
                                    int         opt_level) {
  auto *handle = new CApiCompilerHandle();
  if (soc_version && *soc_version)
    handle->socVersion = soc_version;
  if (arch && *arch)
    handle->arch = arch;
  handle->optLevel = opt_level;
  return reinterpret_cast<AfirtCompiler>(handle);
}

void afirt_compiler_destroy(AfirtCompiler compiler) {
  delete reinterpret_cast<CApiCompilerHandle *>(compiler);
}

int afirt_compiler_compile(AfirtCompiler compiler,
                           const char   *src_file,
                           const char   *output_dir,
                           const char   *kernel_name,
                           char         *out_bin_path,
                           size_t        buf_len) {
  auto *handle = reinterpret_cast<CApiCompilerHandle *>(compiler);
  if (!handle) {
    writeErr(out_bin_path, buf_len, "compiler handle is null");
    return 1;
  }

  FrontendCompileInput input;
  input.kernelSource = src_file ? src_file : "";
  input.outputDir = output_dir ? output_dir : "";
  input.kernelName = kernel_name ? kernel_name : "";
  input.socVersion = handle->socVersion;
  input.arch = handle->arch;
  input.kernelKind =
      compatKernelTypeForArch(handle->arch) == "cube" ? KernelKind::Cube
                                                       : KernelKind::Vec;
  input.optLevel = handle->optLevel;

  auto requestOr = buildFrontendCompileRequest(input);
  if (!requestOr) {
    writeErr(out_bin_path, buf_len, requestOr.takeError());
    return 1;
  }

  ArtifactCompiler artifactCompiler;
  auto artifactOr = artifactCompiler.compile(*requestOr);
  if (!artifactOr) {
    writeErr(out_bin_path, buf_len, artifactOr.takeError());
    return 1;
  }

  const std::string compiledPath = selectCompiledArtifactPath(*artifactOr);
  if (out_bin_path && buf_len > 0) {
    strncpy(out_bin_path, compiledPath.c_str(), buf_len - 1);
    out_bin_path[buf_len - 1] = '\0';
  }
  return 0;
}

// ============================================================
// Executor
// ============================================================

AfirtExecutor afirt_executor_create() {
  return reinterpret_cast<AfirtExecutor>(new CApiExecutorHandle());
}

void afirt_executor_destroy(AfirtExecutor executor) {
  delete reinterpret_cast<CApiExecutorHandle *>(executor);
}

int afirt_executor_initialize(AfirtExecutor executor, int device_id,
                              char *err_buf, size_t err_len) {
  auto *handle = reinterpret_cast<CApiExecutorHandle *>(executor);
  if (!handle) {
    writeErr(err_buf, err_len, "executor handle is null");
    return 1;
  }
  if (device_id < 0) {
    writeErr(err_buf, err_len, "device id must be non-negative");
    return 1;
  }
  handle->deviceId = device_id;
  handle->initialized = true;
  return 0;
}

int afirt_executor_run(AfirtExecutor  executor,
                       const uint8_t *binary_data, size_t binary_len,
                       const char    *function_name,
                       int            num_inputs,
                       const void   **input_ptrs,
                       const size_t  *input_bytes,
                       int            num_outputs,
                       void         **output_ptrs,
                       const size_t  *output_bytes,
                       const uint8_t *tiling_data, size_t tiling_len,
                       int            block_dim,
                       uint32_t       magic,
                       char          *err_buf, size_t err_len) {
  auto *handle = reinterpret_cast<CApiExecutorHandle *>(executor);
  if (!handle || !handle->initialized) {
    writeErr(err_buf, err_len, "executor has not been initialized");
    return 1;
  }
  if (!binary_data || binary_len == 0) {
    writeErr(err_buf, err_len, "binary data is required");
    return 1;
  }

  auto tempDirOr = makeTemporaryDirectory("afirt-runtime-bin");
  if (!tempDirOr) {
    writeErr(err_buf, err_len, tempDirOr.takeError());
    return 1;
  }
  const std::string binaryPath = joinPath(*tempDirOr, "kernel.bin");
  if (auto err = writeBinaryFile(
          binaryPath, llvm::ArrayRef<uint8_t>(binary_data, binary_len))) {
    writeErr(err_buf, err_len, std::move(err));
    return 1;
  }

  auto err = runWithExecutionSession(binaryPath,
                                     function_name ? function_name : "",
                                     magic, num_inputs, input_ptrs,
                                     input_bytes, num_outputs, output_ptrs,
                                     output_bytes, tiling_data, tiling_len,
                                     block_dim, handle->deviceId);
  if (err) {
    writeErr(err_buf, err_len, std::move(err));
    return 1;
  }
  return 0;
}

int afirt_executor_run_file(AfirtExecutor  executor,
                            const char    *binary_path,
                            const char    *function_name,
                            int            num_inputs,
                            const void   **input_ptrs,
                            const size_t  *input_bytes,
                            int            num_outputs,
                            void         **output_ptrs,
                            const size_t  *output_bytes,
                            const uint8_t *tiling_data, size_t tiling_len,
                            int            block_dim,
                            uint32_t       magic,
                            char          *err_buf, size_t err_len) {
  auto *handle = reinterpret_cast<CApiExecutorHandle *>(executor);
  if (!handle || !handle->initialized) {
    writeErr(err_buf, err_len, "executor has not been initialized");
    return 1;
  }
  if (!binary_path || !*binary_path) {
    writeErr(err_buf, err_len, "binary path is required");
    return 1;
  }

  auto bytesOr = readBinaryFile(binary_path);
  if (!bytesOr) {
    writeErr(err_buf, err_len, bytesOr.takeError());
    return 1;
  }

  auto err = runWithExecutionSession(binary_path, function_name ? function_name : "",
                                     magic, num_inputs, input_ptrs,
                                     input_bytes, num_outputs, output_ptrs,
                                     output_bytes, tiling_data, tiling_len,
                                     block_dim, handle->deviceId);
  if (err) {
    writeErr(err_buf, err_len, std::move(err));
    return 1;
  }
  return 0;
}
