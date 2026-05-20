// include/Runtime/NpuBackend.h
#pragma once

#include "Runtime/Execution/ExecutionBackend.h"
#include "Runtime/Execution/ExecutionRunner.h"

#include <functional>
#include <memory>

namespace mlir::runtime {

using NpuExecutionRunnerFactory =
    std::function<llvm::Expected<std::unique_ptr<ExecutionRunner>>(
        ExecutionRunnerMode)>;

class NpuBackend final : public ExecutionBackend {
public:
  explicit NpuBackend(std::shared_ptr<ExecutionBackendDriver> driver = {});
  NpuBackend(std::shared_ptr<ExecutionBackendDriver> driver,
             NpuExecutionRunnerFactory runnerFactory);

  ExecutionBackendKind kind() const override;
  BackendCapabilities capabilities() const override;
  llvm::Expected<ExecutionResult> run(const ExecutionRequest &request) override;

private:
  std::shared_ptr<ExecutionBackendDriver> driver_;
  NpuExecutionRunnerFactory runnerFactory_;
};

} // namespace mlir::runtime
