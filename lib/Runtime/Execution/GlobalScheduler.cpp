#include "Runtime/Execution/GlobalScheduler.h"

namespace mlir::runtime {

size_t GlobalScheduler::sessionCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return sessions_.size();
}

llvm::Expected<SessionHandle>
GlobalScheduler::submit(ExecutionBackendKind backendKind,
                        const TaskGraph &graph) {
  auto orderedOr = graph.orderedTasks();
  if (!orderedOr)
    return orderedOr.takeError();
  if (orderedOr->empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot submit empty task graph");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  const std::string sessionId =
      "global-session-" + std::to_string(nextSessionOrdinal_++);
  sessions_.emplace(sessionId, backendKind);
  return SessionHandle(sessionId);
}

} // namespace mlir::runtime
