// lib/Runtime/ProfileTrace.cpp
#include "Runtime/ProfileTrace.h"

#include <utility>

namespace mlir::runtime {

void ProfileTrace::addEvent(ProfileEvent event) {
  events.push_back(std::move(event));
}

void ProfileTrace::addProfileArtifact(llvm::StringRef taskId,
                                      ExecutionBackendKind backend,
                                      llvm::StringRef artifactPath,
                                      std::optional<int64_t> score,
                                      std::optional<int64_t> cycleCount) {
  addEvent(ProfileEvent{taskId.str(), backend, "profile_artifact",
                        artifactPath.str(), score, cycleCount});
}

void ProfileTrace::setAttribute(llvm::StringRef key, llvm::StringRef value) {
  attributes[key.str()] = value.str();
}

void ProfileTrace::addCounter(llvm::StringRef key, int64_t delta) {
  counters[key.str()] += delta;
}

void ProfileTrace::merge(const ProfileTrace &other) {
  events.insert(events.end(), other.events.begin(), other.events.end());
  for (const auto &[key, value] : other.attributes)
    attributes.try_emplace(key, value);
  for (const auto &[key, value] : other.counters)
    counters[key] += value;
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
