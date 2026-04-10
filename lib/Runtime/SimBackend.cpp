// lib/Runtime/SimBackend.cpp
#include "Runtime/SimBackend.h"

#include "llvm/Support/Error.h"

#include <utility>

namespace mlir::runtime {

SimBackend::SimBackend(std::shared_ptr<ExecutionBackendDriver> driver)
    : driver_(std::move(driver)) {}

ExecutionBackendKind SimBackend::kind() const {
  return ExecutionBackendKind::Simulation;
}

llvm::Expected<ExecutionResult>
SimBackend::run(const ExecutionRequest &request) {
  if (!driver_)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "simulation backend driver not configured");
  return driver_->run(request);
}

} // namespace mlir::runtime
