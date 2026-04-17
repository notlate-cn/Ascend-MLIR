#include "Runtime/Execution/GlobalScheduler.h"

namespace mlir::runtime {

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

  const std::string sessionId =
      "global-session-" + std::to_string(nextSessionOrdinal_++);
  sessions_.emplace(sessionId, backendKind);
  return SessionHandle(sessionId);
}

} // namespace mlir::runtime
