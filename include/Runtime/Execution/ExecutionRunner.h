#pragma once

#include "Runtime/Support/Types.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace mlir::runtime {

enum class ExecutionRunnerMode {
  Simulation,
  RealDevice,
};

struct FileExecutionLaunch {
  std::string binaryPath;
  std::string kernelName;
  uint32_t magic = 0;
};

struct DynamicLibraryExecutionLaunch {
  std::string sharedLibraryPath;
  std::string symbolName;
};

class ExecutionRunner {
public:
  virtual ~ExecutionRunner() = default;

  virtual ExecutionRunnerMode mode() const = 0;
  virtual llvm::Error initialize(int deviceId = 0) = 0;
  virtual llvm::Error runFile(const FileExecutionLaunch &launch,
                              RunArgs &args) = 0;
  virtual llvm::Error
  runDynamicLibraryArtifact(const DynamicLibraryExecutionLaunch &launch,
                            RunArgs &args) = 0;
};

} // namespace mlir::runtime
