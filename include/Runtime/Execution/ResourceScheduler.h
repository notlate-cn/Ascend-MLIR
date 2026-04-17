#pragma once

#include "Runtime/ExecutionBackend.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace mlir::runtime {

struct TaskResourceRequirement {
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  size_t workspaceBytes = 0;
  bool requiresSerializedLaunch = false;
};

struct ResourceReservation {
  uint64_t reservationId = 0;
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
  struct ActiveReservation {
    size_t workspaceBytes = 0;
    bool holdsSerializedLaunchLane = false;
    bool holdsDeviceSlot = false;
  };

  size_t configuredSimDispatchLanes_ = 1;
  size_t configuredDeviceSlots_ = 1;
  size_t configuredWorkspaceBudget_ = 0;
  size_t reservedSimDispatchLanes_ = 0;
  size_t reservedDeviceSlots_ = 0;
  size_t reservedWorkspaceBytes_ = 0;
  uint64_t nextReservationId_ = 1;
  std::unordered_map<uint64_t, ActiveReservation> activeReservations_;
};

} // namespace mlir::runtime
