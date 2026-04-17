// include/Runtime/SimBackend.h
#pragma once

#include "Runtime/Execution/ExecutionBackend.h"

#include <memory>

namespace mlir::runtime {

class SimBackend final : public ExecutionBackend {
public:
  explicit SimBackend(std::shared_ptr<ExecutionBackendDriver> driver = {});

  ExecutionBackendKind kind() const override;
  BackendCapabilities capabilities() const override;
  bool allowsConcurrentTaskDispatch() const override;
  llvm::Expected<ExecutionResult> run(const ExecutionRequest &request) override;

private:
  std::shared_ptr<ExecutionBackendDriver> driver_;
};

} // namespace mlir::runtime
