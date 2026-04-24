#include "Runtime/Execution/GlobalScheduler.h"

#include "llvm/ADT/StringRef.h"

#include <limits>

namespace mlir::runtime {

static int priorityRank(SessionPriorityClass priority) {
  switch (priority) {
  case SessionPriorityClass::Low:
    return 0;
  case SessionPriorityClass::Normal:
    return 1;
  case SessionPriorityClass::High:
    return 2;
  }
  return 1;
}

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
      "global_session_round_robin_baseline";
  snapshot.attributes["scheduler_fairness_policy"] = "session_round_robin";
  snapshot.attributes["scheduler_priority_policy"] = "static_session_priority";
  snapshot.attributes["scheduler_quota_policy"] = "session_admission_quota";
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
    if (record.state != GlobalTaskRecord::State::Ready ||
        !record.waitingOnResources ||
        record.blockedReason == ResourceBlockReason::None) {
      continue;
    }

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
        !record.waitingOnResources ||
        (record.blockedReason != ResourceBlockReason::StreamCapacity &&
         record.blockedReason !=
             ResourceBlockReason::ExclusiveStreamConflict)) {
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
  int64_t quotaBlocked = 0;
  int64_t highReady = 0;
  int64_t normalReady = 0;
  int64_t lowReady = 0;
  for (const auto &sessionEntry : sessions_) {
    const std::string &sessionId = sessionEntry.first;
    const GlobalSessionRecord &session = sessionEntry.second;
    if (session.failed)
      continue;

    bool hasReadyTask = false;
    bool hasQuotaEligibleReadyTask = false;
    for (const auto &taskEntry : tasks_) {
      const GlobalTaskRecord &record = taskEntry.second;
      if (record.sessionId != sessionId ||
          record.state != GlobalTaskRecord::State::Ready)
        continue;
      hasReadyTask = true;
      if (record.waitingOnQuota)
        ++quotaBlocked;
      if (session.scheduling.maxAdmittedTasks == 0 ||
          session.admittedTasks < session.scheduling.maxAdmittedTasks) {
        hasQuotaEligibleReadyTask = true;
      }
    }
    if (!hasReadyTask || !hasQuotaEligibleReadyTask)
      continue;

    switch (session.scheduling.priorityClass) {
    case SessionPriorityClass::High:
      ++highReady;
      break;
    case SessionPriorityClass::Normal:
      ++normalReady;
      break;
    case SessionPriorityClass::Low:
      ++lowReady;
      break;
    }
  }
  snapshot.counters["scheduler.quota.blocked"] = quotaBlocked;
  snapshot.counters["scheduler.quota.blocked_total"] =
      quotaBlockedAdmissionCount_;
  snapshot.counters["scheduler.priority.high_ready"] = highReady;
  snapshot.counters["scheduler.priority.normal_ready"] = normalReady;
  snapshot.counters["scheduler.priority.low_ready"] = lowReady;
  snapshot.counters["scheduler.admission.reserved_total"] =
      successfulReservationCount_;
  snapshot.counters["scheduler.transition.completed"] =
      completedTaskCount_;
  snapshot.counters["scheduler.transition.failed"] = failedTaskCount_;
  snapshot.counters["scheduler.transition.session_release"] =
      releasedSessionCount_;
  snapshot.counters["scheduler.fairness.cursor"] =
      static_cast<int64_t>(fairnessCursor_);
  snapshot.counters["scheduler.fairness.session_order_size"] =
      static_cast<int64_t>(sessionOrder_.size());
  snapshot.counters["scheduler.fairness.session_rotations_total"] =
      fairnessRotationCount_;
  snapshot.counters["scheduler.fairness.session_skips_total"] =
      fairnessSessionSkipCount_;
  snapshot.counters["scheduler.fairness.starvation_prevented_total"] =
      fairnessStarvationPreventedCount_;
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

void GlobalScheduler::releaseReservationLocked(GlobalSessionRecord &session,
                                               GlobalTaskRecord &record) {
  if (!record.reservation)
    return;
  resourceScheduler_.release(*record.reservation);
  record.reservation.reset();
  if (session.admittedTasks > 0)
    --session.admittedTasks;
}

void GlobalScheduler::cancelPendingSessionTasksLocked(llvm::StringRef sessionId) {
  auto sessionIt = sessions_.find(sessionId.str());
  if (sessionIt == sessions_.end())
    return;
  for (auto &entry : tasks_) {
    GlobalTaskRecord &record = entry.second;
    if (record.sessionId != sessionId)
      continue;
    if (record.state == GlobalTaskRecord::State::Running ||
        record.state == GlobalTaskRecord::State::Succeeded ||
        record.state == GlobalTaskRecord::State::Failed ||
        record.state == GlobalTaskRecord::State::Cancelled)
      continue;
    releaseReservationLocked(sessionIt->second, record);
    record.waitingOnResources = false;
    record.waitingOnStreamResources = false;
    record.waitingOnQuota = false;
    record.blockedReason = ResourceBlockReason::None;
    record.state = GlobalTaskRecord::State::Cancelled;
  }
}

bool GlobalScheduler::tryReserveOneReadyTaskForSessionLocked(
    llvm::StringRef sessionId, bool &sawReadyTask, bool &quotaBlocked) {
  sawReadyTask = false;
  quotaBlocked = false;
  auto sessionIt = sessions_.find(sessionId.str());
  if (sessionIt == sessions_.end() || sessionIt->second.failed)
    return false;
  const bool quotaExhausted =
      sessionIt->second.scheduling.maxAdmittedTasks > 0 &&
      sessionIt->second.admittedTasks >=
          sessionIt->second.scheduling.maxAdmittedTasks;

  for (auto &entry : tasks_) {
    GlobalTaskRecord &record = entry.second;
    if (record.sessionId != sessionId ||
        record.state != GlobalTaskRecord::State::Ready) {
      continue;
    }

    sawReadyTask = true;
    if (quotaExhausted) {
      if (!record.waitingOnQuota) {
        ++quotaBlockedAdmissionCount_;
        record.waitingOnQuota = true;
      }
      quotaBlocked = true;
      continue;
    }
    record.waitingOnQuota = false;

    auto reservationOr = resourceScheduler_.tryReserve(
        record.sessionId, record.taskId, record.resources);
    if (!reservationOr) {
      const ResourceBlockReason blockReason =
          resourceScheduler_.lastBlockReason();
      const bool blockedOnStream =
          blockReason == ResourceBlockReason::StreamCapacity ||
          blockReason == ResourceBlockReason::ExclusiveStreamConflict;
      record.blockedReason = blockReason;
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
    if (!lastAdmittedSessionId_.empty() && lastAdmittedSessionId_ != sessionId)
      ++fairnessStarvationPreventedCount_;
    lastAdmittedSessionId_ = sessionId.str();
    ++sessionIt->second.admittedTasks;
    record.waitingOnResources = false;
    record.waitingOnStreamResources = false;
    record.waitingOnQuota = false;
    record.blockedReason = ResourceBlockReason::None;
    record.reservation = std::move(*reservationOr);
    record.state = GlobalTaskRecord::State::Reserved;
    return true;
  }

  return false;
}

void GlobalScheduler::tryReserveReadyTasksLocked() {
  while (!sessionOrder_.empty()) {
    const size_t sessionCount = sessionOrder_.size();
    const size_t start = fairnessCursor_ % sessionCount;
    int selectedPriority = -1;
    for (size_t offset = 0; offset < sessionCount; ++offset) {
      const size_t index = (start + offset) % sessionCount;
      auto sessionIt = sessions_.find(sessionOrder_[index]);
      if (sessionIt == sessions_.end() || sessionIt->second.failed)
        continue;
      const bool quotaExhausted =
          sessionIt->second.scheduling.maxAdmittedTasks > 0 &&
          sessionIt->second.admittedTasks >=
              sessionIt->second.scheduling.maxAdmittedTasks;
      bool hasReadyTask = false;
      for (const auto &entry : tasks_) {
        const GlobalTaskRecord &record = entry.second;
        if (record.sessionId == sessionOrder_[index] &&
            record.state == GlobalTaskRecord::State::Ready) {
          hasReadyTask = true;
          break;
        }
      }
      if (!hasReadyTask || quotaExhausted)
        continue;
      selectedPriority = std::max(
          selectedPriority,
          priorityRank(sessionIt->second.scheduling.priorityClass));
    }
    if (selectedPriority < 0)
      break;

    bool admittedAny = false;

    for (size_t offset = 0; offset < sessionCount; ++offset) {
      const size_t index = (start + offset) % sessionCount;
      const std::string &sessionId = sessionOrder_[index];
      auto sessionIt = sessions_.find(sessionId);
      if (sessionIt == sessions_.end() || sessionIt->second.failed ||
          priorityRank(sessionIt->second.scheduling.priorityClass) !=
              selectedPriority) {
        continue;
      }
      bool sawReadyTask = false;
      bool quotaBlocked = false;
      if (tryReserveOneReadyTaskForSessionLocked(sessionId, sawReadyTask,
                                                 quotaBlocked)) {
        fairnessCursor_ =
            sessionOrder_.empty() ? 0 : ((index + 1) % sessionOrder_.size());
        ++fairnessRotationCount_;
        admittedAny = true;
        continue;
      }
      if (sawReadyTask || quotaBlocked)
        ++fairnessSessionSkipCount_;
    }

    if (!admittedAny)
      break;
  }
}

void GlobalScheduler::noteFairnessSessionRemovalLocked(llvm::StringRef sessionId) {
  const bool removedLastAdmitted = lastAdmittedSessionId_ == sessionId;
  for (size_t index = 0; index < sessionOrder_.size(); ++index) {
    if (sessionOrder_[index] != sessionId)
      continue;

    sessionOrder_.erase(sessionOrder_.begin() + index);
    if (sessionOrder_.empty()) {
      fairnessCursor_ = 0;
      break;
    }
    if (index < fairnessCursor_)
      --fairnessCursor_;
    if (fairnessCursor_ >= sessionOrder_.size())
      fairnessCursor_ %= sessionOrder_.size();
    break;
  }
  if (removedLastAdmitted) {
    lastAdmittedSessionId_.clear();
    return;
  }
  if (!sessionOrder_.empty() && !lastAdmittedSessionId_.empty()) {
    for (size_t index = 0; index < sessionOrder_.size(); ++index) {
      if (sessionOrder_[index] != lastAdmittedSessionId_)
        continue;
      fairnessCursor_ = (index + 1) % sessionOrder_.size();
      break;
    }
  }
}

llvm::Expected<SessionHandle>
GlobalScheduler::submit(ExecutionBackendKind backendKind, const TaskGraph &graph) {
  SessionSchedulingOptions scheduling;
  return submit(backendKind, graph, scheduling);
}

llvm::Expected<SessionHandle>
GlobalScheduler::submit(ExecutionBackendKind backendKind, const TaskGraph &graph,
                        const SessionSchedulingOptions &scheduling) {
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
  return submit(backendKind, caps, graph, scheduling);
}

llvm::Expected<SessionHandle>
GlobalScheduler::submit(ExecutionBackendKind backendKind,
                        const BackendCapabilities &capabilities,
                        const TaskGraph &graph) {
  SessionSchedulingOptions scheduling;
  return submit(backendKind, capabilities, graph, scheduling);
}

llvm::Expected<SessionHandle>
GlobalScheduler::submit(ExecutionBackendKind backendKind,
                        const BackendCapabilities &capabilities,
                        const TaskGraph &graph,
                        const SessionSchedulingOptions &scheduling) {
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
  session.scheduling = scheduling;
  sessions_.emplace(sessionId, std::move(session));
  sessionOrder_.push_back(sessionId);
  if (!lastAdmittedSessionId_.empty()) {
    for (size_t index = 0; index < sessionOrder_.size(); ++index) {
      if (sessionOrder_[index] != lastAdmittedSessionId_)
        continue;
      fairnessCursor_ = (index + 1) % sessionOrder_.size();
      break;
    }
  }

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
      record.waitingOnResources = false;
      record.waitingOnStreamResources = false;
      record.waitingOnQuota = false;
      record.blockedReason = ResourceBlockReason::None;
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

  releaseReservationLocked(sessionIt->second, *record);
  record->waitingOnResources = false;
  record->waitingOnStreamResources = false;
  record->waitingOnQuota = false;
  record->blockedReason = ResourceBlockReason::None;
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
        dependent->waitingOnResources = false;
        dependent->waitingOnStreamResources = false;
        dependent->waitingOnQuota = false;
        dependent->blockedReason = ResourceBlockReason::None;
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

  releaseReservationLocked(sessionIt->second, *record);
  record->waitingOnResources = false;
  record->waitingOnStreamResources = false;
  record->waitingOnQuota = false;
  record->blockedReason = ResourceBlockReason::None;
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
    releaseReservationLocked(sessionIt->second, record);
    record.waitingOnResources = false;
    record.waitingOnStreamResources = false;
    record.waitingOnQuota = false;
    record.blockedReason = ResourceBlockReason::None;
    it = tasks_.erase(it);
  }

  ++releasedSessionCount_;
  noteFairnessSessionRemovalLocked(sessionId);
  sessions_.erase(sessionIt);
  tryReserveReadyTasksLocked();
  schedulerCv_.notify_all();
}

} // namespace mlir::runtime
