// lib/CAPI/Runtime/Runtime.cpp
// C API implementation wrapping runtime compatibility entrypoints.

#include "CAPI/Runtime.h"
#include "Runtime/ArtifactCompiler.h"
#include "Runtime/CompatRuntime.h"
#include "Runtime/ExecutionSession.h"
#include "Runtime/NpyIO.h"
#include "Runtime/Types.h"
#include "Runtime/TaskGraph.h"
#include "Runtime/TilingPack.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <cstring>

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

llvm::StringRef compatKernelTypeForArch(llvm::StringRef arch) {
  return arch.contains_insensitive("cube") ? "cube" : "vec";
}

std::string selectCompiledArtifactPath(const KernelArtifact &artifact) {
  if (!artifact.deviceBinaryPath.empty())
    return artifact.deviceBinaryPath;
  if (!artifact.packedSharedObjectPath.empty())
    return artifact.packedSharedObjectPath;
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

llvm::Expected<std::string> kernelKindNameForMagic(uint32_t magic) {
  if (magic == Executor::MAGIC_ELF_AIVEC)
    return std::string("vec");
  if (magic == Executor::MAGIC_ELF_AICUBE)
    return std::string("cube");
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
  if (!input_ptrs || !input_bytes || !output_ptrs || !output_bytes) {
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

  auto kernelKindOr = kernelKindNameForMagic(magic);
  if (!kernelKindOr)
    return kernelKindOr.takeError();

  auto tilingPathOr = materializeTilingBinary(tempDir, tiling_data, tiling_len);
  if (!tilingPathOr)
    return tilingPathOr.takeError();

  TaskGraph graph;
  RuntimeTask task;
  task.taskId = "main";
  task.artifact.kernelName = functionName;
  task.artifact.kernelKind =
      *kernelKindOr == "cube" ? KernelKind::Cube : KernelKind::Vec;
  task.artifact.artifactRoot = tempDir;
  task.artifact.deviceBinaryPath = binaryPath;
  task.invocation.blockDim = block_dim;

  if (!tilingPathOr->empty()) {
    TilingBinding tiling;
    tiling.binaryPath = *tilingPathOr;
    task.invocation.tiling = std::move(tiling);
  }

  for (int i = 0; i < num_inputs; ++i) {
    TensorBinding binding;
    binding.name = "data" + std::to_string(i);
    binding.sourceKind = BindingSourceKind::ExternalFile;
    binding.path = joinPath(tempDir, "input" + std::to_string(i) + ".npy");
    if (auto err = materializeInputNpy(binding.path, input_ptrs[i],
                                       input_bytes[i]))
      return err;
    task.invocation.inputs.push_back(std::move(binding));
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
    task.invocation.outputs.push_back(std::move(binding));
  }

  if (auto err = graph.addTask(task))
    return err;

  ExecutionSession session(ExecutionBackendKind::Simulation);
  auto traceOr = session.run(graph);
  if (!traceOr)
    return traceOr.takeError();

  for (int i = 0; i < num_outputs; ++i) {
    if (auto err = loadOutputNpy(task.invocation.outputs[i].path,
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

  CompatCompileOptions options;
  options.kernelSourcePath = src_file ? src_file : "";
  options.outputRoot = output_dir ? output_dir : "";
  options.requestedKernelName = kernel_name ? kernel_name : "";
  options.socVersion = handle->socVersion;
  options.arch = handle->arch;
  options.kernelType = compatKernelTypeForArch(handle->arch).str();

  auto requestOr = buildCompatCompileRequest(options);
  if (!requestOr) {
    writeErr(out_bin_path, buf_len, requestOr.takeError());
    return 1;
  }
  requestOr->optLevel = handle->optLevel;

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
  return reinterpret_cast<AfirtExecutor>(new Executor());
}

void afirt_executor_destroy(AfirtExecutor executor) {
  delete reinterpret_cast<Executor *>(executor);
}

int afirt_executor_initialize(AfirtExecutor executor, int device_id,
                              char *err_buf, size_t err_len) {
  auto *e = reinterpret_cast<Executor *>(executor);
  if (auto err = e->Initialize(device_id)) {
    writeErr(err_buf, err_len, std::move(err));
    return 1;
  }
  return 0;
}

// Choose dtype+shape so that numElements()*dtypeBytes() == bytes exactly.
// Prefer F16 (2B), then INT32 (4B), then fall back to F16 with rounded-up
// shape (the +512 overalloc in Alloc() absorbs any padding).
static void fillArray(NDArray& arr, void* data, size_t bytes) {
  arr.data = data;
  if (bytes % 2 == 0) {
    arr.dtype = DType::F16;
    arr.shape = {static_cast<int64_t>(bytes / 2)};
  } else if (bytes % 4 == 0) {
    arr.dtype = DType::INT32;
    arr.shape = {static_cast<int64_t>(bytes / 4)};
  } else {
    // Odd byte count: round up to nearest even; overalloc absorbs the extra byte
    arr.dtype = DType::F16;
    arr.shape = {static_cast<int64_t>((bytes + 1) / 2)};
  }
}

// Build RunArgs from flat C arrays and call Executor::Run / RunFile.
static RunArgs buildRunArgs(int            num_inputs,
                            const void   **input_ptrs,
                            const size_t  *input_bytes,
                            int            num_outputs,
                            void         **output_ptrs,
                            const size_t  *output_bytes,
                            const uint8_t *tiling_data, size_t tiling_len,
                            int            block_dim) {
  RunArgs args;
  args.block_dim = block_dim;

  for (int i = 0; i < num_inputs; ++i) {
    NDArray arr;
    fillArray(arr, const_cast<void *>(input_ptrs[i]), input_bytes[i]);
    args.inputs.push_back(std::move(arr));
  }

  for (int i = 0; i < num_outputs; ++i) {
    NDArray arr;
    fillArray(arr, output_ptrs[i], output_bytes[i]);
    args.outputs.push_back(std::move(arr));
  }

  if (tiling_data && tiling_len > 0)
    args.tiling.assign(tiling_data, tiling_data + tiling_len);

  return args;
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
  auto *e = reinterpret_cast<Executor *>(executor);

  std::vector<uint8_t> bin(binary_data, binary_data + binary_len);
  RunArgs args = buildRunArgs(num_inputs, input_ptrs, input_bytes,
                              num_outputs, output_ptrs, output_bytes,
                              tiling_data, tiling_len, block_dim);

  if (auto err = e->Run(bin, function_name ? function_name : "", args, magic)) {
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
  auto *e = reinterpret_cast<Executor *>(executor);

  RunArgs args = buildRunArgs(num_inputs, input_ptrs, input_bytes,
                              num_outputs, output_ptrs, output_bytes,
                              tiling_data, tiling_len, block_dim);

  if (auto err = e->RunFile(binary_path ? binary_path : "",
                            function_name ? function_name : "", args, magic)) {
    writeErr(err_buf, err_len, std::move(err));
    return 1;
  }
  return 0;
}
