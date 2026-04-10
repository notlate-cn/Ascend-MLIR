// include/Runtime/ProfileTrace.h
#pragma once

#include "Runtime/TaskGraph.h"

#include <string>
#include <vector>

namespace mlir::runtime {

struct ProfileEvent {
  std::string taskId;
  ExecutionBackendKind backend = ExecutionBackendKind::Simulation;
  std::string eventKind;
  std::string artifact;
};

struct ProfileTrace {
  std::string sessionId;
  std::vector<ProfileEvent> events;
};

} // namespace mlir::runtime
