#include "Runtime/Execution/ResourceScheduler.h"

namespace mlir::runtime {

size_t ResourceScheduler::ReservationKeyHash::operator()(
    const ReservationKey &key) const {
  size_t hash = std::hash<std::string>{}(key.sessionId);
  hash ^= std::hash<std::string>{}(key.taskId) + 0x9e3779b9 + (hash << 6) +
          (hash >> 2);
  hash ^= std::hash<int>{}(static_cast<int>(key.backendKind)) + 0x9e3779b9 +
          (hash << 6) + (hash >> 2);
  hash ^= std::hash<size_t>{}(key.workspaceBytes) + 0x9e3779b9 + (hash << 6) +
          (hash >> 2);
  hash ^= std::hash<bool>{}(key.holdsSerializedLaunchLane) + 0x9e3779b9 +
          (hash << 6) + (hash >> 2);
  hash ^= std::hash<bool>{}(key.holdsDeviceSlot) + 0x9e3779b9 + (hash << 6) +
          (hash >> 2);
  return hash;
}

bool ResourceScheduler::ReservationKeyEq::operator()(
    const ReservationKey &lhs, const ReservationKey &rhs) const {
  return lhs.sessionId == rhs.sessionId && lhs.taskId == rhs.taskId &&
         lhs.backendKind == rhs.backendKind &&
         lhs.workspaceBytes == rhs.workspaceBytes &&
         lhs.holdsSerializedLaunchLane == rhs.holdsSerializedLaunchLane &&
         lhs.holdsDeviceSlot == rhs.holdsDeviceSlot;
}

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

  ReservationKey key{reservation.sessionId, reservation.taskId,
                     reservation.backendKind, reservation.workspaceBytes,
                     reservation.holdsSerializedLaunchLane,
                     reservation.holdsDeviceSlot};
  activeReservations_.emplace(
      std::move(key),
      ActiveReservation{requirement.workspaceBytes, deviceSlotsToReserve,
                        holdsSerializedLaunchLane, holdsDeviceSlot});
  reservedWorkspaceBytes_ += requirement.workspaceBytes;
  if (holdsSerializedLaunchLane)
    ++reservedSimDispatchLanes_;
  reservedDeviceSlots_ += deviceSlotsToReserve;
  return reservation;
}

void ResourceScheduler::release(const ResourceReservation &reservation) {
  ReservationKey key{reservation.sessionId, reservation.taskId,
                     reservation.backendKind, reservation.workspaceBytes,
                     reservation.holdsSerializedLaunchLane,
                     reservation.holdsDeviceSlot};
  auto range = activeReservations_.equal_range(key);
  if (range.first == range.second)
    return;

  auto it = range.first;
  const ActiveReservation active = it->second;
  activeReservations_.erase(it);

  reservedWorkspaceBytes_ -= active.workspaceBytes;
  if (active.holdsSerializedLaunchLane)
    --reservedSimDispatchLanes_;
  if (active.holdsDeviceSlot)
    reservedDeviceSlots_ -= active.deviceSlots;
}

} // namespace mlir::runtime
