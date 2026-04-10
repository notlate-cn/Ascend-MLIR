#include "Runtime/ProfileUtils.h"

namespace mlir::runtime {

bool isSimulatorProfileArtifact(llvm::StringRef path) {
  return path.ends_with("/opprof/simulator/trace.json");
}

std::optional<ProfileTrace>
normalizeSimulatorProfileTrace(llvm::StringRef sessionId,
                               llvm::StringRef taskId,
                               llvm::ArrayRef<std::string> producedFiles) {
  ProfileTrace trace;
  trace.sessionId = sessionId.str();

  for (const std::string &file : producedFiles) {
    if (!isSimulatorProfileArtifact(file))
      continue;
    trace.addEvent(makeProfileArtifactEvent(taskId, ExecutionBackendKind::Simulation,
                                            file));
  }

  if (trace.events.empty())
    return std::nullopt;
  return trace;
}

ProfileEvent makeProfileArtifactEvent(llvm::StringRef taskId,
                                      ExecutionBackendKind backend,
                                      llvm::StringRef artifactPath) {
  return ProfileEvent{taskId.str(), backend, "profile_artifact",
                      artifactPath.str()};
}

void addProfileArtifact(ProfileTrace &trace, llvm::StringRef taskId,
                        ExecutionBackendKind backend,
                        llvm::StringRef artifactPath) {
  trace.addEvent(
      makeProfileArtifactEvent(taskId, backend, artifactPath));
}

} // namespace mlir::runtime
