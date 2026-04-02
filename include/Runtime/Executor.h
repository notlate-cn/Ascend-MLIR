// include/Runtime/Executor.h
#pragma once
#include "Runtime/Types.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>
#include <vector>

namespace mlir::runtime {

enum class BackendMode {
  Simulation,
  RealDevice,
};

struct DevBinary {
  uint32_t    magic;
  uint32_t    version;
  const char* data;
  uint64_t    length;
};

class Executor {
public:
  static constexpr uint32_t MAGIC_ELF_AIVEC  = 0x41415246u;
  static constexpr uint32_t MAGIC_ELF_AICUBE = 0x41494343u;

  explicit Executor(BackendMode mode = BackendMode::Simulation);
  ~Executor();

  llvm::Error Initialize(int device_id = 0);

  llvm::Expected<void*> RegisterBinary(const std::string& binary_path,
                                       const std::string& function_name,
                                       uint32_t magic = MAGIC_ELF_AIVEC);

  llvm::Error RunWithHandle(void* func_handle, RunArgs& args);

  llvm::Error Run(const std::vector<uint8_t>& binary_data,
                  const std::string& function_name,
                  RunArgs& args,
                  uint32_t magic = MAGIC_ELF_AIVEC);

  llvm::Error RunFile(const std::string& binary_path,
                      const std::string& function_name,
                      RunArgs& args,
                      uint32_t magic = MAGIC_ELF_AIVEC);

  llvm::Error RunPackedMixFile(const std::string& shared_lib_path,
                               const std::string& kernel_name,
                               RunArgs& args);

private:
  BackendMode mode_;
  void*       lib_handle_ = nullptr;
  void*       acl_handle_ = nullptr;
  void*       stream_     = nullptr;
  int32_t     device_id_  = 0;

  int (*rtSetDevice_)(int32_t)                                           = nullptr;
  int (*rtDevBinaryRegister_)(const DevBinary*, void**)                  = nullptr;
  int (*rtFunctionRegister_)(void*, void*, const char*, void*, uint32_t) = nullptr;
  int (*rtMalloc_)(void**, uint64_t, uint32_t, uint16_t)                 = nullptr;
  int (*rtFree_)(void*)                                                   = nullptr;
  int (*rtMemcpy_)(void*, uint64_t, const void*, uint64_t, uint32_t)     = nullptr;
  int (*rtStreamCreate_)(void**, int32_t)                                 = nullptr;
  int (*rtStreamDestroy_)(void*)                                          = nullptr;
  int (*rtKernelLaunch_)(void*, uint32_t, void*, uint32_t, void*, void*) = nullptr;
  int (*rtStreamSynchronize_)(void*)                                      = nullptr;
  int (*rtDeviceSynchronize_)()                                           = nullptr;
  int (*aclInit_)(const char*)                                            = nullptr;
  int (*aclFinalize_)()                                                   = nullptr;
  int (*aclrtSetDevice_)(int32_t)                                         = nullptr;
  int (*aclrtResetDevice_)(int32_t)                                       = nullptr;
  int (*aclrtCreateStream_)(void**)                                       = nullptr;
  int (*aclrtDestroyStream_)(void*)                                       = nullptr;
  int (*aclrtMalloc_)(void**, uint64_t, uint32_t)                         = nullptr;
  int (*aclrtFree_)(void*)                                                = nullptr;
  int (*aclrtMallocHost_)(void**, uint64_t)                               = nullptr;
  int (*aclrtFreeHost_)(void*)                                            = nullptr;
  int (*aclrtMemcpy_)(void*, uint64_t, const void*, uint64_t, int32_t)    = nullptr;
  int (*aclrtSynchronizeStream_)(void*)                                   = nullptr;

  llvm::Error LoadLib();

  struct AllocInfo {
    void* raw;
    void* aligned;
  };
  std::vector<AllocInfo> alloc_map_;
  std::vector<std::vector<uint8_t>> registered_binaries_;
  std::vector<std::string>          registered_names_;

  llvm::Expected<void*> Alloc(size_t nbytes);
  void FreeAll();
  llvm::Error H2D(void* dst, const void* src, size_t n);
  llvm::Error D2H(void* dst, const void* src, size_t n);
};

} // namespace mlir::runtime
