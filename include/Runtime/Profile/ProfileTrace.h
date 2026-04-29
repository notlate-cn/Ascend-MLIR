// include/Runtime/ProfileTrace.h
#pragma once

#include "Runtime/Execution/TaskGraph.h"
#include "llvm/ADT/StringRef.h"

#include <map>
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
  std::map<std::string, std::string> attributes;
  std::map<std::string, int64_t> counters;

  void addEvent(ProfileEvent event);
  void addProfileArtifact(llvm::StringRef taskId,
                          ExecutionBackendKind backend,
                          llvm::StringRef artifactPath,
                          std::optional<int64_t> score = std::nullopt,
                          std::optional<int64_t> cycleCount = std::nullopt);
  void setAttribute(llvm::StringRef key, llvm::StringRef value);
  void addCounter(llvm::StringRef key, int64_t delta = 1);
  void merge(const ProfileTrace &other);
  std::vector<std::string> profileArtifactPaths() const;
};

} // namespace mlir::runtime
