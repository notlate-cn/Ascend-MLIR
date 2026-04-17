#pragma once

#include "Runtime/ExecutionBackend.h"

#include <cstddef>
#include <optional>
#include <string>

namespace mlir::runtime {

struct TaskResourceRequirement {
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  size_t workspaceBytes = 0;
  bool requiresSerializedLaunch = false;
  bool exclusiveDeviceAccess = false;
};

struct ResourceReservation {
  std::string sessionId;
  std::string taskId;
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  size_t workspaceBytes = 0;
  bool holdsSerializedLaunchLane = false;
  bool holdsDeviceSlot = false;
};

class ResourceScheduler {
public:
  void configureSimDispatchLanes(size_t count);
  void configureDeviceSlots(size_t count);
  void configureWorkspaceBudget(size_t bytes);

  std::optional<ResourceReservation>
  tryReserve(const std::string &sessionId, const std::string &taskId,
             const TaskResourceRequirement &requirement);

  void release(const ResourceReservation &reservation);

private:
  size_t availableSimDispatchLanes_ = 1;
  size_t availableDeviceSlots_ = 1;
  size_t availableWorkspaceBytes_ = 0;
};

} // namespace mlir::runtime
