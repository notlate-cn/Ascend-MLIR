// lib/Runtime/ExecutionBackend.cpp
#include "Runtime/ExecutionBackend.h"
#include "Runtime/NpuBackend.h"
#include "Runtime/SimBackend.h"

#include "llvm/Support/Error.h"

#include <memory>
#include <utility>

namespace mlir::runtime {

llvm::Expected<std::unique_ptr<ExecutionBackend>>
createExecutionBackend(ExecutionBackendKind kind,
                       std::shared_ptr<ExecutionBackendDriver> driver) {
  switch (kind) {
  case ExecutionBackendKind::Simulation:
    return std::make_unique<SimBackend>(std::move(driver));
  case ExecutionBackendKind::Npu:
    return std::make_unique<NpuBackend>(std::move(driver));
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid execution backend kind");
}

} // namespace mlir::runtime
