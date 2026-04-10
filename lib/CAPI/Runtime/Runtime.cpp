// lib/CAPI/Runtime/Runtime.cpp
// C API implementation wrapping mlir::runtime::{Compiler, Executor}.

#include "CAPI/Runtime.h"
#include "Runtime/Compiler.h"
#include "Runtime/Executor.h"
#include "Runtime/Types.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>

using namespace mlir::runtime;

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
  CompilerConfig cfg;
  if (soc_version && *soc_version)
    cfg.soc_version = soc_version;
  if (arch && *arch)
    cfg.arch = arch;
  cfg.opt_level = opt_level;
  return reinterpret_cast<AfirtCompiler>(new Compiler(cfg));
}

void afirt_compiler_destroy(AfirtCompiler compiler) {
  delete reinterpret_cast<Compiler *>(compiler);
}

int afirt_compiler_compile(AfirtCompiler compiler,
                           const char   *src_file,
                           const char   *output_dir,
                           const char   *kernel_name,
                           char         *out_bin_path,
                           size_t        buf_len) {
  auto *c = reinterpret_cast<Compiler *>(compiler);
  auto result = c->Compile(src_file ? src_file : "",
                           output_dir ? output_dir : "",
                           kernel_name ? kernel_name : "");
  if (!result) {
    writeErr(out_bin_path, buf_len, result.takeError());
    return 1;
  }
  if (out_bin_path && buf_len > 0) {
    strncpy(out_bin_path, result->c_str(), buf_len - 1);
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
