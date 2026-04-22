#pragma once

#include "Runtime/Execution/SessionHandle.h"
#include "Runtime/Execution/ResourceScheduler.h"
#include "Runtime/Execution/TaskGraph.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <map>
#include <optional>
#include <mutex>
#include <string>

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
  TaskResourceRequirement resources;
  State state = State::Submitted;
  std::optional<ResourceReservation> reservation;
};

class GlobalScheduler {
public:
  GlobalScheduler();

  llvm::Expected<SessionHandle> submit(ExecutionBackendKind backendKind,
                                       const TaskGraph &graph);

  size_t sessionCount() const;
  ResourceScheduler &mutableResourceScheduler();
  void configureResourceScheduler(size_t simDispatchLanes, size_t deviceSlots,
                                  size_t workspaceBudget);
  size_t taskCountInState(GlobalTaskRecord::State state) const;

private:
  static std::string taskKey(llvm::StringRef sessionId, llvm::StringRef taskId);
  void tryReserveReadyTasks();

  size_t nextSessionOrdinal_ = 0;
  std::map<std::string, ExecutionBackendKind> sessions_;
  std::map<std::string, GlobalTaskRecord> tasks_;
  ResourceScheduler resourceScheduler_;
  mutable std::mutex mutex_;
};

} // namespace mlir::runtime
