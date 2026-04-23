#include "Runtime/Execution/GlobalScheduler.h"

#include "llvm/ADT/StringRef.h"

#include <limits>

namespace mlir::runtime {

GlobalScheduler::GlobalScheduler() {
  resourceScheduler_.configureSimDispatchLanes(1024);
  resourceScheduler_.configureDeviceSlots(1024);
  resourceScheduler_.configureWorkspaceBudget(
      std::numeric_limits<size_t>::max());
}

size_t GlobalScheduler::sessionCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return sessions_.size();
}

void GlobalScheduler::configureResourceScheduler(size_t simDispatchLanes,
                                                 size_t deviceSlots,
                                                 size_t workspaceBudget) {
  std::lock_guard<std::mutex> lock(mutex_);
  resourceScheduler_.configureSimDispatchLanes(simDispatchLanes);
  resourceScheduler_.configureDeviceSlots(deviceSlots);
  resourceScheduler_.configureWorkspaceBudget(workspaceBudget);
  tryReserveReadyTasksLocked();
  schedulerCv_.notify_all();
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

GlobalTaskRecord *GlobalScheduler::findTaskLocked(llvm::StringRef sessionId,
                                                  llvm::StringRef taskId) {
  auto it = tasks_.find(taskKey(sessionId, taskId));
  if (it == tasks_.end())
    return nullptr;
  return &it->second;
}

bool GlobalScheduler::sessionIsDrainedLocked(llvm::StringRef sessionId) const {
  auto sessionIt = sessions_.find(sessionId.str());
  if (sessionIt == sessions_.end())
    return true;
  const GlobalSessionRecord &session = sessionIt->second;
  return session.completedTasks == session.totalTasks ||
         (session.failed && session.runningTasks == 0);
}

void GlobalScheduler::cancelPendingSessionTasksLocked(llvm::StringRef sessionId) {
  for (auto &entry : tasks_) {
    GlobalTaskRecord &record = entry.second;
    if (record.sessionId != sessionId)
      continue;
    if (record.state == GlobalTaskRecord::State::Running ||
        record.state == GlobalTaskRecord::State::Succeeded ||
        record.state == GlobalTaskRecord::State::Failed ||
        record.state == GlobalTaskRecord::State::Cancelled)
      continue;
    if (record.reservation) {
      resourceScheduler_.release(*record.reservation);
      record.reservation.reset();
    }
    record.state = GlobalTaskRecord::State::Cancelled;
  }
}

void GlobalScheduler::tryReserveReadyTasksLocked() {
  for (auto &entry : tasks_) {
    GlobalTaskRecord &record = entry.second;
    if (record.state != GlobalTaskRecord::State::Ready)
      continue;

    auto sessionIt = sessions_.find(record.sessionId);
    if (sessionIt == sessions_.end() || sessionIt->second.failed)
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
GlobalScheduler::submit(ExecutionBackendKind backendKind, const TaskGraph &graph) {
  BackendCapabilities caps;
  if (backendKind == ExecutionBackendKind::Simulation) {
    caps.supportsConcurrentDispatch = true;
    caps.supportsConcurrentExecution = false;
    caps.requiresSerializedLaunch = true;
    caps.maxConcurrentTasks = 1024;
    caps.maxConcurrentStreams = 1;
  } else {
    caps.supportsConcurrentDispatch = true;
    caps.supportsConcurrentExecution = true;
    caps.requiresSerializedLaunch = false;
    caps.maxConcurrentTasks = 1024;
    caps.maxConcurrentStreams = 1024;
  }
  return submit(backendKind, caps, graph);
}

llvm::Expected<SessionHandle>
GlobalScheduler::submit(ExecutionBackendKind backendKind,
                        const BackendCapabilities &capabilities,
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
  GlobalSessionRecord session;
  session.backendKind = backendKind;
  session.totalTasks = orderedOr->size();
  sessions_.emplace(sessionId, std::move(session));

  for (const RuntimeTask &task : *orderedOr) {
    GlobalTaskRecord record;
    record.sessionId = sessionId;
    record.taskId = task.taskId;
    record.backendKind = backendKind;
    record.task = task;
    record.resources.backendKind = backendKind;
    record.resources.workspaceBytes = task.invocation.workspaceSize;
    record.resources.requiresSerializedLaunch =
        capabilities.requiresSerializedLaunch;
    record.resources.exclusiveDeviceAccess =
        backendKind == ExecutionBackendKind::Npu &&
        (!capabilities.supportsConcurrentExecution ||
         capabilities.maxConcurrentTasks <= 1);
    record.remainingDependencies = task.dependencies.size();
    record.state = task.dependencies.empty() ? GlobalTaskRecord::State::Ready
                                             : GlobalTaskRecord::State::Submitted;
    tasks_.emplace(taskKey(record.sessionId, record.taskId), std::move(record));
  }

  for (const RuntimeTask &task : *orderedOr) {
    for (const std::string &dependency : task.dependencies) {
      if (GlobalTaskRecord *depRecord = findTaskLocked(sessionId, dependency))
        depRecord->dependents.push_back(task.taskId);
    }
  }

  tryReserveReadyTasksLocked();
  schedulerCv_.notify_all();
  return SessionHandle(sessionId);
}

llvm::Expected<std::optional<RuntimeTask>>
GlobalScheduler::waitAndAcquireTask(llvm::StringRef sessionId) {
  std::unique_lock<std::mutex> lock(mutex_);
  while (true) {
    auto sessionIt = sessions_.find(sessionId.str());
    if (sessionIt == sessions_.end()) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "unknown global scheduler session: %s",
                                     sessionId.str().c_str());
    }

    for (auto &entry : tasks_) {
      GlobalTaskRecord &record = entry.second;
      if (record.sessionId != sessionId ||
          record.state != GlobalTaskRecord::State::Reserved) {
        continue;
      }
      record.state = GlobalTaskRecord::State::Running;
      ++sessionIt->second.runningTasks;
      return std::optional<RuntimeTask>(record.task);
    }

    if (sessionIsDrainedLocked(sessionId))
      return std::optional<RuntimeTask>();

    schedulerCv_.wait(lock);
  }
}

llvm::Expected<size_t> GlobalScheduler::completeTask(llvm::StringRef sessionId,
                                                     llvm::StringRef taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto sessionIt = sessions_.find(sessionId.str());
  if (sessionIt == sessions_.end()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unknown global scheduler session: %s",
                                   sessionId.str().c_str());
  }
  GlobalTaskRecord *record = findTaskLocked(sessionId, taskId);
  if (!record) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unknown global scheduler task: %s/%s",
                                   sessionId.str().c_str(),
                                   taskId.str().c_str());
  }
  if (record->state != GlobalTaskRecord::State::Running) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "task is not running: %s/%s",
                                   sessionId.str().c_str(),
                                   taskId.str().c_str());
  }

  if (record->reservation) {
    resourceScheduler_.release(*record->reservation);
    record->reservation.reset();
  }
  record->state = GlobalTaskRecord::State::Succeeded;
  ++sessionIt->second.completedTasks;
  if (sessionIt->second.runningTasks > 0)
    --sessionIt->second.runningTasks;

  size_t newlyReadyCount = 0;
  if (!sessionIt->second.failed) {
    for (const std::string &dependentId : record->dependents) {
      GlobalTaskRecord *dependent = findTaskLocked(sessionId, dependentId);
      if (!dependent ||
          dependent->state != GlobalTaskRecord::State::Submitted) {
        continue;
      }
      if (dependent->remainingDependencies == 0)
        continue;
      --dependent->remainingDependencies;
      if (dependent->remainingDependencies == 0) {
        dependent->state = GlobalTaskRecord::State::Ready;
        ++newlyReadyCount;
      }
    }
  }

  tryReserveReadyTasksLocked();
  schedulerCv_.notify_all();
  return newlyReadyCount;
}

llvm::Error GlobalScheduler::failTask(llvm::StringRef sessionId,
                                      llvm::StringRef taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto sessionIt = sessions_.find(sessionId.str());
  if (sessionIt == sessions_.end()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unknown global scheduler session: %s",
                                   sessionId.str().c_str());
  }
  GlobalTaskRecord *record = findTaskLocked(sessionId, taskId);
  if (!record) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unknown global scheduler task: %s/%s",
                                   sessionId.str().c_str(),
                                   taskId.str().c_str());
  }
  if (record->state != GlobalTaskRecord::State::Running) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "task is not running: %s/%s",
                                   sessionId.str().c_str(),
                                   taskId.str().c_str());
  }

  if (record->reservation) {
    resourceScheduler_.release(*record->reservation);
    record->reservation.reset();
  }
  record->state = GlobalTaskRecord::State::Failed;
  sessionIt->second.failed = true;
  if (sessionIt->second.runningTasks > 0)
    --sessionIt->second.runningTasks;

  cancelPendingSessionTasksLocked(sessionId);
  schedulerCv_.notify_all();
  return llvm::Error::success();
}

void GlobalScheduler::releaseSession(llvm::StringRef sessionId) {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto it = tasks_.begin(); it != tasks_.end();) {
    GlobalTaskRecord &record = it->second;
    if (record.sessionId != sessionId) {
      ++it;
      continue;
    }
    if (record.reservation)
      resourceScheduler_.release(*record.reservation);
    it = tasks_.erase(it);
  }

  sessions_.erase(sessionId.str());
  tryReserveReadyTasksLocked();
  schedulerCv_.notify_all();
}

} // namespace mlir::runtime
