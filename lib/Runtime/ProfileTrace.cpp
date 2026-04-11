// lib/Runtime/ProfileTrace.cpp
#include "Runtime/ProfileTrace.h"

#include <utility>

namespace mlir::runtime {

void ProfileTrace::addEvent(ProfileEvent event) {
  events.push_back(std::move(event));
}

void ProfileTrace::addProfileArtifact(llvm::StringRef taskId,
                                      ExecutionBackendKind backend,
                                      llvm::StringRef artifactPath) {
  addEvent(ProfileEvent{taskId.str(), backend, "profile_artifact",
                        artifactPath.str()});
}

std::vector<std::string> ProfileTrace::profileArtifactPaths() const {
  std::vector<std::string> artifacts;
  artifacts.reserve(events.size());
  for (const ProfileEvent &event : events) {
    if (event.eventKind != "profile_artifact" || event.artifact.empty())
      continue;
    artifacts.push_back(event.artifact);
  }
  return artifacts;
}

} // namespace mlir::runtime
