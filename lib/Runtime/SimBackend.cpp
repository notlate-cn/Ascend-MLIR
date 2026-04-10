// lib/Runtime/SimBackend.cpp
#include "Runtime/SimBackend.h"
#include "Runtime/ProfileUtils.h"

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
  auto resultOr = driver_->run(request);
  if (!resultOr)
    return resultOr.takeError();

  const std::string &sessionId =
      request.sessionId.empty() ? request.task.taskId : request.sessionId;
  auto traceOr = normalizeSimulatorProfileTrace(sessionId,
                                                request.task.taskId,
                                                resultOr->producedFiles);
  if (traceOr && !resultOr->profileTrace)
    resultOr->profileTrace = std::move(*traceOr);

  return resultOr;
}

} // namespace mlir::runtime
