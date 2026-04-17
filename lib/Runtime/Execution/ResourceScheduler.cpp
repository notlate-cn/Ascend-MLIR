#include "Runtime/Execution/ResourceScheduler.h"

namespace mlir::runtime {

void ResourceScheduler::configureSimDispatchLanes(size_t count) {
  configuredSimDispatchLanes_ = count;
}

void ResourceScheduler::configureDeviceSlots(size_t count) {
  configuredDeviceSlots_ = count;
}

void ResourceScheduler::configureWorkspaceBudget(size_t bytes) {
  configuredWorkspaceBudget_ = bytes;
}

std::optional<ResourceReservation>
ResourceScheduler::tryReserve(const std::string &sessionId,
                              const std::string &taskId,
                              const TaskResourceRequirement &requirement) {
  if (configuredWorkspaceBudget_ < reservedWorkspaceBytes_)
    return std::nullopt;
  if (requirement.workspaceBytes >
      (configuredWorkspaceBudget_ - reservedWorkspaceBytes_))
    return std::nullopt;

  bool holdsSerializedLaunchLane = false;
  if (requirement.backendKind == ExecutionBackendKind::Simulation &&
      requirement.requiresSerializedLaunch) {
    if (configuredSimDispatchLanes_ <= reservedSimDispatchLanes_)
      return std::nullopt;
    holdsSerializedLaunchLane = true;
  }

  bool holdsDeviceSlot = false;
  size_t deviceSlotsToReserve = 0;
  if (requirement.backendKind == ExecutionBackendKind::Npu) {
    deviceSlotsToReserve = requirement.exclusiveDeviceAccess
                               ? configuredDeviceSlots_
                               : 1;
    if (deviceSlotsToReserve == 0 ||
        configuredDeviceSlots_ < reservedDeviceSlots_ + deviceSlotsToReserve)
      return std::nullopt;
    holdsDeviceSlot = true;
  }

  ResourceReservation reservation;
  reservation.sessionId = sessionId;
  reservation.taskId = taskId;
  reservation.backendKind = requirement.backendKind;
  reservation.workspaceBytes = requirement.workspaceBytes;
  reservation.holdsSerializedLaunchLane = holdsSerializedLaunchLane;
  reservation.holdsDeviceSlot = holdsDeviceSlot;
  reservation.token_ = nextReservationToken_++;

  activeReservations_.emplace(reservation.token_,
                              ActiveReservation{reservation.sessionId,
                                                reservation.taskId,
                                                reservation.backendKind,
                                                reservation.workspaceBytes,
                                                deviceSlotsToReserve,
                                                holdsSerializedLaunchLane,
                                                holdsDeviceSlot});
  reservedWorkspaceBytes_ += requirement.workspaceBytes;
  if (holdsSerializedLaunchLane)
    ++reservedSimDispatchLanes_;
  reservedDeviceSlots_ += deviceSlotsToReserve;
  return reservation;
}

void ResourceScheduler::release(const ResourceReservation &reservation) {
  auto it = activeReservations_.find(reservation.token_);
  if (it == activeReservations_.end())
    return;

  const ActiveReservation active = it->second;
  if (active.sessionId != reservation.sessionId ||
      active.taskId != reservation.taskId ||
      active.backendKind != reservation.backendKind ||
      active.workspaceBytes != reservation.workspaceBytes ||
      active.holdsSerializedLaunchLane != reservation.holdsSerializedLaunchLane ||
      active.holdsDeviceSlot != reservation.holdsDeviceSlot) {
    return;
  }

  activeReservations_.erase(it);

  reservedWorkspaceBytes_ -= active.workspaceBytes;
  if (active.holdsSerializedLaunchLane)
    --reservedSimDispatchLanes_;
  if (active.holdsDeviceSlot)
    reservedDeviceSlots_ -= active.deviceSlots;
}

} // namespace mlir::runtime
