// include/Runtime/Executor.h
#pragma once
#include "Runtime/Types.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>
#include <vector>

namespace mlir::runtime {

enum class BackendMode {
  Simulation,  // libruntime_camodel.so (CPU simulation)
  RealDevice,  // reserved, not implemented
};

// Mirrors Python DevBinary struct (field order MUST match runtime ABI)
struct DevBinary {
  uint32_t    magic;    // 0x41415246 for vec kernel
  uint32_t    version;  // 0
  const char* data;     // ELF bytes pointer
  uint64_t    length;   // ELF byte count
};

class Executor {
public:
  static constexpr uint32_t MAGIC_ELF_AIVEC  = 0x41415246u;
  static constexpr uint32_t MAGIC_ELF_AICUBE = 0x41494343u;

  explicit Executor(BackendMode mode = BackendMode::Simulation);
  ~Executor();

  llvm::Error Initialize(int device_id = 0);

  // binary_data: raw ELF bytes from .bin file
  llvm::Error Run(const std::vector<uint8_t>& binary_data,
                  const std::string& function_name,
                  RunArgs& args,
                  uint32_t magic = MAGIC_ELF_AIVEC);

  // Convenience: read binary from file, then call Run
  llvm::Error RunFile(const std::string& binary_path,
                      const std::string& function_name,
                      RunArgs& args,
                      uint32_t magic = MAGIC_ELF_AIVEC);

private:
  BackendMode mode_;
  void*       lib_handle_ = nullptr;

  // Runtime API function pointers (exact signatures from Python executor.py)
  int (*rtSetDevice_)(int32_t)                                           = nullptr;
  int (*rtDevBinaryRegister_)(const DevBinary*, void**)                  = nullptr;
  int (*rtFunctionRegister_)(void*, void*, const char*, void*, uint32_t) = nullptr;
  int (*rtMalloc_)(void**, uint64_t, uint32_t, uint16_t)                 = nullptr;
  int (*rtFree_)(void*)                                                   = nullptr;
  int (*rtMemcpy_)(void*, uint64_t, const void*, uint64_t, uint32_t)     = nullptr;
  int (*rtStreamCreate_)(void**, int32_t)                                 = nullptr;
  int (*rtStreamDestroy_)(void*)                                          = nullptr;
  int (*rtKernelLaunch_)(void*, uint32_t, void*, uint32_t, void*, void*) = nullptr;
  int (*rtDeviceSynchronize_)()                                           = nullptr;

  llvm::Error LoadLib();

  struct AllocInfo { void* raw; void* aligned; };
  std::vector<AllocInfo> alloc_map_;

  llvm::Expected<void*> Alloc(size_t nbytes);
  void FreeAll();

  // H2D in 256-byte chunks (kind=1)
  llvm::Error H2D(void* dst, const void* src, size_t n);
  // D2H in 4-byte chunks (kind=2); safe due to +512 overalloc
  llvm::Error D2H(void* dst, const void* src, size_t n);
};

} // namespace mlir::runtime
