#include "Runtime/ProfileUtils.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <filesystem>

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

llvm::Expected<ProfileTrace>
retainProfileArtifactsForCli(const ProfileTrace &trace,
                             llvm::StringRef destinationRoot) {
  ProfileTrace retained = trace;
  if (auto ec = llvm::sys::fs::create_directories(destinationRoot))
    return llvm::createStringError(
        ec, "cannot create retained profile directory: %s",
        destinationRoot.str().c_str());

  size_t artifactIndex = 0;
  for (ProfileEvent &event : retained.events) {
    if (event.eventKind != "profile_artifact" || event.artifact.empty())
      continue;

    llvm::StringRef sourcePath = event.artifact;
    llvm::StringRef extension = llvm::sys::path::extension(sourcePath);
    llvm::SmallString<256> retainedPath(destinationRoot);
    llvm::sys::path::append(retainedPath, event.taskId + "-" +
                                             std::to_string(artifactIndex++) +
                                             extension.str());

    std::error_code copyError;
    std::filesystem::copy_file(sourcePath.str(), retainedPath.str().str(),
                               std::filesystem::copy_options::overwrite_existing,
                               copyError);
    if (copyError) {
      return llvm::createStringError(copyError,
                                     "cannot retain profile artifact %s -> %s",
                                     sourcePath.str().c_str(),
                                     retainedPath.str().str().c_str());
    }
    event.artifact = retainedPath.str().str();
  }

  return retained;
}

} // namespace mlir::runtime
