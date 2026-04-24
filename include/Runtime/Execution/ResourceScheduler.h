#pragma once

#include "Runtime/ExecutionBackend.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace mlir::runtime {

enum class ResourceBlockReason {
  None,
  Workspace,
  SerializedLaunch,
  DeviceCapacity,
  StreamCapacity,
  ExclusiveStreamConflict,
};

struct TaskResourceRequirement {
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  size_t workspaceBytes = 0;
  bool requiresSerializedLaunch = false;
  bool exclusiveDeviceAccess = false;
  bool requiresStream = false;
  size_t streamUnits = 0;
  bool exclusiveStreamAccess = false;
};

struct ResourceReservation {
  std::string sessionId;
  std::string taskId;
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  size_t workspaceBytes = 0;
  bool holdsSerializedLaunchLane = false;
  bool holdsDeviceSlot = false;
  bool holdsStreamSlot = false;
  size_t reservedStreamUnits = 0;

private:
  friend class ResourceScheduler;
  uint64_t token_ = 0;
};

class ResourceScheduler {
public:
  void configureSimDispatchLanes(size_t count);
  void configureDeviceSlots(size_t count);
  void configureWorkspaceBudget(size_t bytes);
  void configureStreamCapacity(size_t count);
  ResourceBlockReason lastBlockReason() const { return lastBlockReason_; }

  std::optional<ResourceReservation>
  tryReserve(const std::string &sessionId, const std::string &taskId,
             const TaskResourceRequirement &requirement);

  void release(const ResourceReservation &reservation);

private:
  struct ActiveReservation {
    std::string sessionId;
    std::string taskId;
    ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
    size_t workspaceBytes = 0;
    size_t deviceSlots = 0;
    bool holdsSerializedLaunchLane = false;
    bool holdsDeviceSlot = false;
    bool holdsStreamSlot = false;
    size_t streamUnits = 0;
    bool exclusiveStreamAccess = false;
  };

  size_t configuredSimDispatchLanes_ = 1;
  size_t configuredDeviceSlots_ = 1;
  size_t configuredWorkspaceBudget_ = 0;
  size_t configuredStreamCapacity_ = 0;
  size_t reservedSimDispatchLanes_ = 0;
  size_t reservedDeviceSlots_ = 0;
  size_t reservedWorkspaceBytes_ = 0;
  size_t reservedStreamUnits_ = 0;
  bool hasExclusiveStreamReservation_ = false;
  ResourceBlockReason lastBlockReason_ = ResourceBlockReason::None;
  uint64_t nextReservationToken_ = 1;
  std::unordered_map<uint64_t, ActiveReservation>
      activeReservations_;
};

} // namespace mlir::runtime
