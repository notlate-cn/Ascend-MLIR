// include/Runtime/ProfileTrace.h
#pragma once

#include "Runtime/TaskGraph.h"
#include "llvm/ADT/StringRef.h"

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

  void addEvent(ProfileEvent event);
  void addProfileArtifact(llvm::StringRef taskId,
                          ExecutionBackendKind backend,
                          llvm::StringRef artifactPath);
  std::vector<std::string> profileArtifactPaths() const;
};

} // namespace mlir::runtime
