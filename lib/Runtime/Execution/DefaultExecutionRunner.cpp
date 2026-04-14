#include "Runtime/Execution/DefaultExecutionRunner.h"

#include "Runtime/Executor.h"

#include "llvm/Support/Error.h"

#include <memory>

namespace mlir::runtime {

namespace {

BackendMode toLegacyBackendMode(ExecutionRunnerMode mode) {
  switch (mode) {
  case ExecutionRunnerMode::Simulation:
    return BackendMode::Simulation;
  case ExecutionRunnerMode::RealDevice:
    return BackendMode::RealDevice;
  }
  return BackendMode::Simulation;
}

class DefaultExecutionRunner final : public ExecutionRunner {
public:
  explicit DefaultExecutionRunner(ExecutionRunnerMode mode)
      : mode_(mode), executor_(toLegacyBackendMode(mode)) {}

  ExecutionRunnerMode mode() const override { return mode_; }

  llvm::Error initialize(int deviceId = 0) override {
    return executor_.Initialize(deviceId);
  }

  llvm::Error runFile(const FileExecutionLaunch &launch, RunArgs &args) override {
    return executor_.RunFile(launch.binaryPath, launch.kernelName, args,
                             launch.magic);
  }

  llvm::Error runPackedMixFile(const PackedMixExecutionLaunch &launch,
                               RunArgs &args) override {
    return executor_.RunPackedMixFile(launch.sharedLibraryPath,
                                      launch.kernelName, args);
  }

private:
  ExecutionRunnerMode mode_;
  Executor executor_;
};

} // namespace

llvm::Expected<std::unique_ptr<ExecutionRunner>>
createDefaultExecutionRunner(ExecutionRunnerMode mode) {
  return std::make_unique<DefaultExecutionRunner>(mode);
}

} // namespace mlir::runtime
