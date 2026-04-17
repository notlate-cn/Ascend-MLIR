#pragma once

#include <cstddef>

namespace mlir::runtime {

struct BackendCapabilities {
  bool supportsConcurrentDispatch = false;
  bool supportsConcurrentExecution = false;
  bool requiresSerializedLaunch = false;
  size_t maxConcurrentTasks = 1;
  size_t maxConcurrentStreams = 1;
};

} // namespace mlir::runtime
