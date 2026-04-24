#include "Runtime/Execution/GlobalScheduler.h"

#include "llvm/ADT/StringRef.h"

#include <limits>

namespace mlir::runtime {

GlobalScheduler::GlobalScheduler() {
  resourceScheduler_.configureSimDispatchLanes(1024);
  resourceScheduler_.configureDeviceSlots(1024);
  resourceScheduler_.configureWorkspaceBudget(
      std::numeric_limits<size_t>::max());
  resourceScheduler_.configureStreamCapacity(1024);
}

size_t GlobalScheduler::sessionCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return sessions_.size();
}

void GlobalScheduler::configureResourceScheduler(size_t simDispatchLanes,
                                                 size_t deviceSlots,
                                                 size_t workspaceBudget,
                                                 size_t streamCapacity) {
  std::lock_guard<std::mutex> lock(mutex_);
  resourceScheduler_.configureSimDispatchLanes(simDispatchLanes);
  resourceScheduler_.configureDeviceSlots(deviceSlots);
  resourceScheduler_.configureWorkspaceBudget(workspaceBudget);
  if (streamCapacity > 0)
    resourceScheduler_.configureStreamCapacity(streamCapacity);
  tryReserveReadyTasksLocked();
  schedulerCv_.notify_all();
}

size_t GlobalScheduler::taskCountInState(GlobalTaskRecord::State state) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return taskCountInStateLocked(state);
}

size_t GlobalScheduler::taskCountInStateLocked(
    GlobalTaskRecord::State state) const {
  size_t count = 0;
  for (const auto &entry : tasks_) {
    if (entry.second.state == state)
      ++count;
  }
  return count;
}

SchedulerObservabilitySnapshot GlobalScheduler::observabilitySnapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);

  SchedulerObservabilitySnapshot snapshot;
  snapshot.attributes["scheduler_policy"] =
      "global_string_key_order_baseline";
  snapshot.attributes["resource_model_version"] = "v2";
  snapshot.attributes["scheduler_stream_model"] = "enabled";
  snapshot.counters["scheduler.session_count"] =
      static_cast<int64_t>(sessions_.size());
  snapshot.counters["scheduler.task.submitted"] = static_cast<int64_t>(
      taskCountInStateLocked(GlobalTaskRecord::State::Submitted));
  snapshot.counters["scheduler.task.ready"] =
      static_cast<int64_t>(taskCountInStateLocked(GlobalTaskRecord::State::Ready));
  snapshot.counters["scheduler.task.reserved"] = static_cast<int64_t>(
      taskCountInStateLocked(GlobalTaskRecord::State::Reserved));
  snapshot.counters["scheduler.task.running"] = static_cast<int64_t>(
      taskCountInStateLocked(GlobalTaskRecord::State::Running));
  snapshot.counters["scheduler.task.succeeded"] = static_cast<int64_t>(
      taskCountInStateLocked(GlobalTaskRecord::State::Succeeded));
  snapshot.counters["scheduler.task.failed"] =
      static_cast<int64_t>(taskCountInStateLocked(GlobalTaskRecord::State::Failed));
  snapshot.counters["scheduler.task.cancelled"] = static_cast<int64_t>(
      taskCountInStateLocked(GlobalTaskRecord::State::Cancelled));

  int64_t resourceBlocked = 0;
  for (const auto &entry : tasks_) {
    const GlobalTaskRecord &record = entry.second;
    if (record.state != GlobalTaskRecord::State::Ready)
      continue;

    auto sessionIt = sessions_.find(record.sessionId);
    if (sessionIt == sessions_.end() || sessionIt->second.failed)
      continue;

    ++resourceBlocked;
  }
  snapshot.counters["scheduler.admission.resource_blocked"] = resourceBlocked;
  snapshot.counters["scheduler.admission.resource_blocked_total"] =
      resourceBlockedAdmissionCount_;
  int64_t streamBlocked = 0;
  for (const auto &entry : tasks_) {
    const GlobalTaskRecord &record = entry.second;
    if (record.state != GlobalTaskRecord::State::Ready ||
        !record.waitingOnStreamResources) {
      continue;
    }

    auto sessionIt = sessions_.find(record.sessionId);
    if (sessionIt == sessions_.end() || sessionIt->second.failed)
      continue;

    ++streamBlocked;
  }
  snapshot.counters["scheduler.stream.capacity_total"] =
      static_cast<int64_t>(resourceScheduler_.configuredStreamCapacity());
  snapshot.counters["scheduler.stream.capacity_available"] =
      static_cast<int64_t>(resourceScheduler_.availableStreamCapacity());
  snapshot.counters["scheduler.stream.reserved"] =
      static_cast<int64_t>(resourceScheduler_.reservedStreamUnits());
  snapshot.counters["scheduler.admission.stream_blocked"] = streamBlocked;
  snapshot.counters["scheduler.admission.stream_blocked_total"] =
      streamBlockedAdmissionCount_;
  snapshot.counters["scheduler.admission.reserved_total"] =
      successfulReservationCount_;
  snapshot.counters["scheduler.transition.completed"] =
      completedTaskCount_;
  snapshot.counters["scheduler.transition.failed"] = failedTaskCount_;
  snapshot.counters["scheduler.transition.session_release"] =
      releasedSessionCount_;
  return snapshot;
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
    if (!reservationOr) {
      const ResourceBlockReason blockReason = resourceScheduler_.lastBlockReason();
      const bool blockedOnStream =
          blockReason == ResourceBlockReason::StreamCapacity ||
          blockReason == ResourceBlockReason::ExclusiveStreamConflict;
      if (!record.waitingOnResources) {
        ++resourceBlockedAdmissionCount_;
        record.waitingOnResources = true;
      }
      if (blockedOnStream && !record.waitingOnStreamResources) {
        ++streamBlockedAdmissionCount_;
        record.waitingOnStreamResources = true;
      } else if (!blockedOnStream) {
        record.waitingOnStreamResources = false;
      }
      continue;
    }

    ++successfulReservationCount_;
    record.waitingOnResources = false;
    record.waitingOnStreamResources = false;
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
    record.resources.requiresStream = capabilities.maxConcurrentStreams > 0;
    record.resources.streamUnits =
        record.resources.requiresStream ? 1 : 0;
    record.resources.exclusiveStreamAccess = false;
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
  ++completedTaskCount_;
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
  ++failedTaskCount_;
  sessionIt->second.failed = true;
  if (sessionIt->second.runningTasks > 0)
    --sessionIt->second.runningTasks;

  cancelPendingSessionTasksLocked(sessionId);
  schedulerCv_.notify_all();
  return llvm::Error::success();
}

void GlobalScheduler::releaseSession(llvm::StringRef sessionId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto sessionIt = sessions_.find(sessionId.str());
  if (sessionIt == sessions_.end())
    return;

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

  ++releasedSessionCount_;
  sessions_.erase(sessionIt);
  tryReserveReadyTasksLocked();
  schedulerCv_.notify_all();
}

} // namespace mlir::runtime
