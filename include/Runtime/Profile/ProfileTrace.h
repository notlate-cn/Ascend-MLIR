// include/Runtime/ProfileTrace.h
#pragma once

#include "Runtime/Execution/TaskGraph.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>
#include <vector>

namespace mlir::runtime {

struct ProfileEvent {
  std::string taskId;
  ExecutionBackendKind backend = ExecutionBackendKind::Simulation;
  std::string eventKind;
  std::string artifact;
  std::optional<int64_t> score;
  std::optional<int64_t> cycleCount;
};

struct ProfileTrace {
  std::string sessionId;
  std::vector<ProfileEvent> events;

  void addEvent(ProfileEvent event);
  void addProfileArtifact(llvm::StringRef taskId,
                          ExecutionBackendKind backend,
                          llvm::StringRef artifactPath,
                          std::optional<int64_t> score = std::nullopt,
                          std::optional<int64_t> cycleCount = std::nullopt);
  std::vector<std::string> profileArtifactPaths() const;
};

} // namespace mlir::runtime
