#include "Runtime/Execution/ResourceScheduler.h"

namespace mlir::runtime {

void ResourceScheduler::configureSimDispatchLanes(size_t count) {
  availableSimDispatchLanes_ = count;
}

void ResourceScheduler::configureDeviceSlots(size_t count) {
  availableDeviceSlots_ = count;
}

void ResourceScheduler::configureWorkspaceBudget(size_t bytes) {
  availableWorkspaceBytes_ = bytes;
}

std::optional<ResourceReservation>
ResourceScheduler::tryReserve(const std::string &sessionId,
                              const std::string &taskId,
                              const TaskResourceRequirement &requirement) {
  if (requirement.workspaceBytes > availableWorkspaceBytes_)
    return std::nullopt;

  ResourceReservation reservation;
  reservation.sessionId = sessionId;
  reservation.taskId = taskId;
  reservation.backendKind = requirement.backendKind;
  reservation.workspaceBytes = requirement.workspaceBytes;

  if (requirement.backendKind == ExecutionBackendKind::Simulation &&
      requirement.requiresSerializedLaunch) {
    if (availableSimDispatchLanes_ == 0)
      return std::nullopt;
    --availableSimDispatchLanes_;
    reservation.holdsSerializedLaunchLane = true;
  }

  if (requirement.backendKind == ExecutionBackendKind::Npu) {
    if (availableDeviceSlots_ == 0)
      return std::nullopt;
    --availableDeviceSlots_;
    reservation.holdsDeviceSlot = true;
  }

  availableWorkspaceBytes_ -= requirement.workspaceBytes;
  return reservation;
}

void ResourceScheduler::release(const ResourceReservation &reservation) {
  availableWorkspaceBytes_ += reservation.workspaceBytes;
  if (reservation.holdsSerializedLaunchLane)
    ++availableSimDispatchLanes_;
  if (reservation.holdsDeviceSlot)
    ++availableDeviceSlots_;
}

} // namespace mlir::runtime
