// include/Runtime/NpuBackend.h
#pragma once

#include "Runtime/Execution/ExecutionBackend.h"

#include <memory>

namespace mlir::runtime {

class NpuBackend final : public ExecutionBackend {
public:
  explicit NpuBackend(std::shared_ptr<ExecutionBackendDriver> driver = {});

  ExecutionBackendKind kind() const override;
  BackendCapabilities capabilities() const override;
  llvm::Expected<ExecutionResult> run(const ExecutionRequest &request) override;

private:
  std::shared_ptr<ExecutionBackendDriver> driver_;
};

} // namespace mlir::runtime
