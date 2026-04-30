#pragma once

#include <cstddef>

namespace mlir::runtime {

struct BackendCapabilities {
  bool supportsConcurrentDispatch = false;
  bool supportsConcurrentExecution = false;
  bool requiresSerializedLaunch = false;
  size_t maxConcurrentTasks = 1;
  // Zero means the backend did not advertise a stream-capacity contract.
  size_t maxConcurrentStreams = 1;
};

} // namespace mlir::runtime
