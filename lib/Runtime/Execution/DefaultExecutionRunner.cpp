#include "Runtime/Execution/DefaultExecutionRunner.h"
#include "Runtime/Execution/NativeExecutionRunner.h"

#include "llvm/Support/Error.h"

#include <memory>

namespace mlir::runtime {

namespace {

class DefaultExecutionRunner final : public ExecutionRunner {
public:
  explicit DefaultExecutionRunner(ExecutionRunnerMode mode) : runner_(mode) {}

  ExecutionRunnerMode mode() const override { return runner_.mode(); }

  llvm::Error initialize(int deviceId = 0) override {
    return runner_.initialize(deviceId);
  }

  llvm::Error runFile(const FileExecutionLaunch &launch, RunArgs &args) override {
    return runner_.runFile(launch, args);
  }

  llvm::Error
  runDynamicLibraryArtifact(const DynamicLibraryExecutionLaunch &launch,
                            RunArgs &args) override {
    return runner_.runDynamicLibraryArtifact(launch, args);
  }

private:
  NativeExecutionRunner runner_;
};

} // namespace

llvm::Expected<std::unique_ptr<ExecutionRunner>>
createDefaultExecutionRunner(ExecutionRunnerMode mode) {
  return std::make_unique<DefaultExecutionRunner>(mode);
}

} // namespace mlir::runtime
