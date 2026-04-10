// lib/Runtime/NpuBackend.cpp
#include "Runtime/NpuBackend.h"

#include "llvm/Support/Error.h"

#include <utility>

namespace mlir::runtime {

NpuBackend::NpuBackend(std::shared_ptr<ExecutionBackendDriver> driver)
    : driver_(std::move(driver)) {}

ExecutionBackendKind NpuBackend::kind() const {
  return ExecutionBackendKind::Npu;
}

llvm::Expected<ExecutionResult>
NpuBackend::run(const ExecutionRequest &request) {
  if (!driver_)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "npu backend driver not configured");
  return driver_->run(request);
}

} // namespace mlir::runtime
