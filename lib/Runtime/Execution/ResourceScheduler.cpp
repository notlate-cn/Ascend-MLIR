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
  if (requirement.backendKind == ExecutionBackendKind::Npu) {
    if (configuredDeviceSlots_ <= reservedDeviceSlots_)
      return std::nullopt;
    holdsDeviceSlot = true;
  }

  ResourceReservation reservation;
  reservation.reservationId = nextReservationId_++;
  reservation.sessionId = sessionId;
  reservation.taskId = taskId;
  reservation.backendKind = requirement.backendKind;
  reservation.workspaceBytes = requirement.workspaceBytes;
  reservation.holdsSerializedLaunchLane = holdsSerializedLaunchLane;
  reservation.holdsDeviceSlot = holdsDeviceSlot;

  activeReservations_.emplace(
      reservation.reservationId,
      ActiveReservation{requirement.workspaceBytes, holdsSerializedLaunchLane,
                        holdsDeviceSlot});
  reservedWorkspaceBytes_ += requirement.workspaceBytes;
  if (holdsSerializedLaunchLane)
    ++reservedSimDispatchLanes_;
  if (holdsDeviceSlot)
    ++reservedDeviceSlots_;
  return reservation;
}

void ResourceScheduler::release(const ResourceReservation &reservation) {
  auto it = activeReservations_.find(reservation.reservationId);
  if (it == activeReservations_.end())
    return;

  reservedWorkspaceBytes_ -= it->second.workspaceBytes;
  if (it->second.holdsSerializedLaunchLane)
    --reservedSimDispatchLanes_;
  if (it->second.holdsDeviceSlot)
    --reservedDeviceSlots_;
  activeReservations_.erase(it);
}

} // namespace mlir::runtime
