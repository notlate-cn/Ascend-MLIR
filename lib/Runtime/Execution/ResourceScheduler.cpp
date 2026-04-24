#include "Runtime/Execution/ResourceScheduler.h"

#include <algorithm>

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

void ResourceScheduler::configureStreamCapacity(size_t count) {
  configuredStreamCapacity_ = count;
}

std::optional<ResourceReservation>
ResourceScheduler::tryReserve(const std::string &sessionId,
                              const std::string &taskId,
                              const TaskResourceRequirement &requirement) {
  lastBlockReason_ = ResourceBlockReason::None;

  if (configuredWorkspaceBudget_ < reservedWorkspaceBytes_) {
    lastBlockReason_ = ResourceBlockReason::Workspace;
    return std::nullopt;
  }
  if (requirement.workspaceBytes >
      (configuredWorkspaceBudget_ - reservedWorkspaceBytes_)) {
    lastBlockReason_ = ResourceBlockReason::Workspace;
    return std::nullopt;
  }

  bool holdsSerializedLaunchLane = false;
  if (requirement.backendKind == ExecutionBackendKind::Simulation &&
      requirement.requiresSerializedLaunch) {
    if (configuredSimDispatchLanes_ <= reservedSimDispatchLanes_) {
      lastBlockReason_ = ResourceBlockReason::SerializedLaunch;
      return std::nullopt;
    }
    holdsSerializedLaunchLane = true;
  }

  bool holdsDeviceSlot = false;
  size_t deviceSlotsToReserve = 0;
  if (requirement.backendKind == ExecutionBackendKind::Npu) {
    deviceSlotsToReserve = requirement.exclusiveDeviceAccess
                               ? configuredDeviceSlots_
                               : 1;
    if (deviceSlotsToReserve == 0 ||
        configuredDeviceSlots_ < reservedDeviceSlots_ + deviceSlotsToReserve) {
      lastBlockReason_ = ResourceBlockReason::DeviceCapacity;
      return std::nullopt;
    }
    holdsDeviceSlot = true;
  }

  bool holdsStreamSlot = false;
  size_t streamUnitsToReserve = requirement.requiresStream
                                    ? std::max<size_t>(requirement.streamUnits, 1)
                                    : 0;
  if (streamUnitsToReserve > 0) {
    if (requirement.exclusiveStreamAccess &&
        (reservedStreamUnits_ > 0 || hasExclusiveStreamReservation_)) {
      lastBlockReason_ = ResourceBlockReason::ExclusiveStreamConflict;
      return std::nullopt;
    }
    if (!requirement.exclusiveStreamAccess && hasExclusiveStreamReservation_) {
      lastBlockReason_ = ResourceBlockReason::ExclusiveStreamConflict;
      return std::nullopt;
    }
    if (configuredStreamCapacity_ <
        reservedStreamUnits_ + streamUnitsToReserve) {
      lastBlockReason_ = ResourceBlockReason::StreamCapacity;
      return std::nullopt;
    }
    holdsStreamSlot = true;
  }

  ResourceReservation reservation;
  reservation.sessionId = sessionId;
  reservation.taskId = taskId;
  reservation.backendKind = requirement.backendKind;
  reservation.workspaceBytes = requirement.workspaceBytes;
  reservation.holdsSerializedLaunchLane = holdsSerializedLaunchLane;
  reservation.holdsDeviceSlot = holdsDeviceSlot;
  reservation.holdsStreamSlot = holdsStreamSlot;
  reservation.reservedStreamUnits = streamUnitsToReserve;
  reservation.token_ = nextReservationToken_++;

  activeReservations_.emplace(reservation.token_,
                              ActiveReservation{reservation.sessionId,
                                                reservation.taskId,
                                                reservation.backendKind,
                                                reservation.workspaceBytes,
                                                deviceSlotsToReserve,
                                                holdsSerializedLaunchLane,
                                                holdsDeviceSlot,
                                                holdsStreamSlot,
                                                streamUnitsToReserve,
                                                requirement.exclusiveStreamAccess});
  reservedWorkspaceBytes_ += requirement.workspaceBytes;
  if (holdsSerializedLaunchLane)
    ++reservedSimDispatchLanes_;
  reservedDeviceSlots_ += deviceSlotsToReserve;
  if (holdsStreamSlot) {
    reservedStreamUnits_ += streamUnitsToReserve;
    if (requirement.exclusiveStreamAccess)
      hasExclusiveStreamReservation_ = true;
  }
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
      active.holdsDeviceSlot != reservation.holdsDeviceSlot ||
      active.holdsStreamSlot != reservation.holdsStreamSlot ||
      active.streamUnits != reservation.reservedStreamUnits) {
    return;
  }

  activeReservations_.erase(it);

  reservedWorkspaceBytes_ -= active.workspaceBytes;
  if (active.holdsSerializedLaunchLane)
    --reservedSimDispatchLanes_;
  if (active.holdsDeviceSlot)
    reservedDeviceSlots_ -= active.deviceSlots;
  if (active.holdsStreamSlot) {
    reservedStreamUnits_ -= active.streamUnits;
    if (active.exclusiveStreamAccess)
      hasExclusiveStreamReservation_ = false;
  }
  lastBlockReason_ = ResourceBlockReason::None;
}

} // namespace mlir::runtime
