#pragma once

#include "Runtime/Execution/ExecutionRunner.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir::runtime {

struct DevBinary;

class NativeExecutionRunner {
public:
  explicit NativeExecutionRunner(ExecutionRunnerMode mode);
  ~NativeExecutionRunner();

  ExecutionRunnerMode mode() const { return mode_; }

  llvm::Error initialize(int deviceId = 0);
  llvm::Error runFile(const FileExecutionLaunch &launch, RunArgs &args);
  llvm::Error
  runDynamicLibraryArtifact(const DynamicLibraryExecutionLaunch &launch,
                            RunArgs &args);

private:
  llvm::Error loadRuntimeLibraries();
  llvm::Expected<void *> alloc(size_t nbytes);
  void freeAll();
  llvm::Error hostToDevice(void *dst, const void *src, size_t nbytes);
  llvm::Error deviceToHost(void *dst, const void *src, size_t nbytes);
  llvm::Expected<void *> registerBinary(const std::string &binaryPath,
                                        const std::string &functionName,
                                        uint32_t magic);
  llvm::Error runWithHandle(void *funcHandle, RunArgs &args);
  llvm::Error runBinary(const std::vector<uint8_t> &binaryData,
                        const std::string &functionName, RunArgs &args,
                        uint32_t magic);

  ExecutionRunnerMode mode_;
  bool initialized_ = false;
  void *libHandle_ = nullptr;
  void *aclHandle_ = nullptr;
  void *stream_ = nullptr;
  int32_t deviceId_ = 0;

  int (*rtSetDevice_)(int32_t) = nullptr;
  int (*rtDevBinaryRegister_)(const DevBinary *, void **) = nullptr;
  int (*rtFunctionRegister_)(void *, void *, const char *, void *, uint32_t) =
      nullptr;
  int (*rtMalloc_)(void **, uint64_t, uint32_t, uint16_t) = nullptr;
  int (*rtFree_)(void *) = nullptr;
  int (*rtMemcpy_)(void *, uint64_t, const void *, uint64_t, uint32_t) =
      nullptr;
  int (*rtStreamCreate_)(void **, int32_t) = nullptr;
  int (*rtStreamDestroy_)(void *) = nullptr;
  int (*rtKernelLaunch_)(void *, uint32_t, void *, uint32_t, void *, void *) =
      nullptr;
  int (*rtStreamSynchronize_)(void *) = nullptr;
  int (*rtDeviceSynchronize_)() = nullptr;
  int (*aclInit_)(const char *) = nullptr;
  int (*aclFinalize_)() = nullptr;
  int (*aclrtSetDevice_)(int32_t) = nullptr;
  int (*aclrtResetDevice_)(int32_t) = nullptr;
  int (*aclrtCreateStream_)(void **) = nullptr;
  int (*aclrtDestroyStream_)(void *) = nullptr;
  int (*aclrtMalloc_)(void **, uint64_t, uint32_t) = nullptr;
  int (*aclrtFree_)(void *) = nullptr;
  int (*aclrtMallocHost_)(void **, uint64_t) = nullptr;
  int (*aclrtFreeHost_)(void *) = nullptr;
  int (*aclrtMemcpy_)(void *, uint64_t, const void *, uint64_t, int32_t) =
      nullptr;
  int (*aclrtSynchronizeStream_)(void *) = nullptr;

  struct AllocInfo {
    void *raw;
    void *aligned;
  };

  std::vector<AllocInfo> allocMap_;
  std::vector<std::vector<uint8_t>> registeredBinaries_;
  std::vector<std::string> registeredNames_;
  std::unordered_map<std::string, void *> registeredFunctionHandles_;
};

} // namespace mlir::runtime
