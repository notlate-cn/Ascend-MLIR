// include/CAPI/Runtime.h
// C API for runtime-native compile/execute entry points — callable from Python
// ctypes.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Opaque handle types
// ============================================================

typedef struct AfirtCompiler_s *AfirtCompiler;
typedef struct AfirtExecutor_s *AfirtExecutor;

// ============================================================
// Compiler API
// ============================================================

// Create a compile handle. soc_version / arch / opt_level mirror the runtime
// compile request surface.
// arch: "dav-c220-vec" or "dav-c220-cube"
// Returns NULL on failure; caller owns the handle.
AfirtCompiler afirt_compiler_create(const char *soc_version,
                                    const char *arch,
                                    int         opt_level);

void afirt_compiler_destroy(AfirtCompiler compiler);

// Compile src_file -> output_dir/kernel_name.bin
// Returns 0 on success; out_bin_path filled with the resulting path (up to
// buf_len bytes including NUL).  On error returns non-zero and writes an error
// message into out_bin_path.
int afirt_compiler_compile(AfirtCompiler compiler,
                           const char   *src_file,
                           const char   *output_dir,
                           const char   *kernel_name,
                           char         *out_bin_path,
                           size_t        buf_len);

// ============================================================
// Executor API
// ============================================================

// Create an execution handle (simulation mode).
// Returns NULL on failure.
AfirtExecutor afirt_executor_create();
void          afirt_executor_destroy(AfirtExecutor executor);

// Initialize: load runtime lib and set device.
// Returns 0 on success, non-zero on error.  err_buf receives message.
int afirt_executor_initialize(AfirtExecutor executor, int device_id,
                              char *err_buf, size_t err_len);

// Execute a kernel from an in-memory ELF binary.
//
//   binary_data / binary_len   : raw ELF bytes
//   function_name              : kernel entry point
//   num_inputs                 : number of input tensors
//   input_ptrs[i]              : pointer to flat data for input i
//   input_bytes[i]             : byte count for input i
//   num_outputs                : number of output tensors
//   output_ptrs[i]             : pointer to pre-allocated output buffer i
//   output_bytes[i]            : byte count for output i
//   tiling_data / tiling_len   : packed tiling bytes (little-endian)
//   block_dim                  : kernel block dimension
//   magic                      : ELF magic (0x41415246=AIVEC, 0x41494343=AICUBE)
//
// Returns 0 on success.
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
                       char          *err_buf, size_t err_len);

// Convenience: read binary from file, then call afirt_executor_run.
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
                            char          *err_buf, size_t err_len);

#ifdef __cplusplus
}
#endif
