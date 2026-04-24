#pragma once

#include "Runtime/Execution/BackendCapabilities.h"
#include "Runtime/Execution/SessionHandle.h"
#include "Runtime/Execution/ResourceScheduler.h"
#include "Runtime/Execution/TaskGraph.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <mutex>
#include <string>
#include <vector>

namespace mlir::runtime {

struct GlobalTaskRecord {
  enum class State {
    Submitted,
    Ready,
    Reserved,
    Running,
    Succeeded,
    Failed,
    Cancelled,
  };

  std::string sessionId;
  std::string taskId;
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  RuntimeTask task;
  TaskResourceRequirement resources;
  size_t remainingDependencies = 0;
  std::vector<std::string> dependents;
  State state = State::Submitted;
  std::optional<ResourceReservation> reservation;
  bool waitingOnResources = false;
  bool waitingOnStreamResources = false;
  ResourceBlockReason blockedReason = ResourceBlockReason::None;
};

struct SchedulerObservabilitySnapshot {
  std::map<std::string, std::string> attributes;
  std::map<std::string, int64_t> counters;
};

class GlobalScheduler {
public:
  GlobalScheduler();

  llvm::Expected<SessionHandle> submit(ExecutionBackendKind backendKind,
                                       const TaskGraph &graph);
  llvm::Expected<SessionHandle>
  submit(ExecutionBackendKind backendKind,
         const BackendCapabilities &capabilities, const TaskGraph &graph);

  llvm::Expected<std::optional<RuntimeTask>>
  waitAndAcquireTask(llvm::StringRef sessionId);
  llvm::Expected<size_t> completeTask(llvm::StringRef sessionId,
                                      llvm::StringRef taskId);
  llvm::Error failTask(llvm::StringRef sessionId, llvm::StringRef taskId);
  void releaseSession(llvm::StringRef sessionId);
  size_t sessionCount() const;
  void configureResourceScheduler(size_t simDispatchLanes, size_t deviceSlots,
                                  size_t workspaceBudget,
                                  size_t streamCapacity = 0);
  size_t taskCountInState(GlobalTaskRecord::State state) const;
  SchedulerObservabilitySnapshot observabilitySnapshot() const;

private:
  struct GlobalSessionRecord {
    ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
    size_t totalTasks = 0;
    size_t completedTasks = 0;
    size_t runningTasks = 0;
    bool failed = false;
  };

  static std::string taskKey(llvm::StringRef sessionId, llvm::StringRef taskId);
  void tryReserveReadyTasksLocked();
  bool sessionIsDrainedLocked(llvm::StringRef sessionId) const;
  void cancelPendingSessionTasksLocked(llvm::StringRef sessionId);
  GlobalTaskRecord *findTaskLocked(llvm::StringRef sessionId,
                                   llvm::StringRef taskId);
  size_t taskCountInStateLocked(GlobalTaskRecord::State state) const;

  size_t nextSessionOrdinal_ = 0;
  int64_t resourceBlockedAdmissionCount_ = 0;
  int64_t streamBlockedAdmissionCount_ = 0;
  int64_t successfulReservationCount_ = 0;
  int64_t failedTaskCount_ = 0;
  int64_t completedTaskCount_ = 0;
  int64_t releasedSessionCount_ = 0;
  std::map<std::string, GlobalSessionRecord> sessions_;
  std::map<std::string, GlobalTaskRecord> tasks_;
  ResourceScheduler resourceScheduler_;
  mutable std::mutex mutex_;
  std::condition_variable schedulerCv_;
};

} // namespace mlir::runtime
