#include "Runtime/Execution/GlobalScheduler.h"

#include "llvm/ADT/StringRef.h"

namespace mlir::runtime {

GlobalScheduler::GlobalScheduler() {
  resourceScheduler_.configureSimDispatchLanes(1);
  resourceScheduler_.configureDeviceSlots(1);
  resourceScheduler_.configureWorkspaceBudget(1 << 20);
}

size_t GlobalScheduler::sessionCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return sessions_.size();
}

ResourceScheduler &GlobalScheduler::mutableResourceScheduler() {
  return resourceScheduler_;
}

void GlobalScheduler::configureResourceScheduler(size_t simDispatchLanes,
                                                 size_t deviceSlots,
                                                 size_t workspaceBudget) {
  std::lock_guard<std::mutex> lock(mutex_);
  resourceScheduler_.configureSimDispatchLanes(simDispatchLanes);
  resourceScheduler_.configureDeviceSlots(deviceSlots);
  resourceScheduler_.configureWorkspaceBudget(workspaceBudget);
  tryReserveReadyTasks();
}

size_t GlobalScheduler::taskCountInState(GlobalTaskRecord::State state) const {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t count = 0;
  for (const auto &entry : tasks_) {
    if (entry.second.state == state)
      ++count;
  }
  return count;
}

std::string GlobalScheduler::taskKey(llvm::StringRef sessionId,
                                     llvm::StringRef taskId) {
  std::string key = sessionId.str();
  key += "::";
  key += taskId.str();
  return key;
}

void GlobalScheduler::tryReserveReadyTasks() {
  for (auto &entry : tasks_) {
    GlobalTaskRecord &record = entry.second;
    if (record.state != GlobalTaskRecord::State::Ready)
      continue;

    auto reservationOr = resourceScheduler_.tryReserve(
        record.sessionId, record.taskId, record.resources);
    if (!reservationOr)
      continue;

    record.reservation = std::move(*reservationOr);
    record.state = GlobalTaskRecord::State::Reserved;
  }
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

  for (const RuntimeTask &task : *orderedOr) {
    if (!task.dependencies.empty())
      continue;

    GlobalTaskRecord record;
    record.sessionId = sessionId;
    record.taskId = task.taskId;
    record.backendKind = backendKind;
    record.resources.backendKind = backendKind;
    record.resources.workspaceBytes = task.invocation.workspaceSize;
    record.resources.requiresSerializedLaunch =
        backendKind == ExecutionBackendKind::Simulation;
    record.state = GlobalTaskRecord::State::Ready;
    tasks_.emplace(taskKey(record.sessionId, record.taskId), std::move(record));
  }

  tryReserveReadyTasks();

  return SessionHandle(sessionId);
}

} // namespace mlir::runtime
