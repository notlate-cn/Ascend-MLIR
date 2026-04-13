#include "Runtime/ProfileUtils.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <filesystem>
#include <vector>

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

namespace {

struct RetainedProfileDirectory {
  std::string path;
  std::filesystem::file_time_type mtime;
};

llvm::Error pruneRetainedProfileDirectoriesImpl(llvm::StringRef root,
                                                size_t keepCount) {
  const std::string rootPath = root.str();
  std::error_code ec;
  if (!std::filesystem::exists(rootPath, ec))
    return llvm::Error::success();
  if (ec)
    return llvm::createStringError(ec,
                                   "cannot inspect retained profile root: %s",
                                   rootPath.c_str());

  std::vector<RetainedProfileDirectory> directories;
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator(rootPath, ec)) {
    if (ec)
      return llvm::createStringError(ec,
                                     "cannot iterate retained profile root: %s",
                                     rootPath.c_str());

    std::error_code dirEc;
    if (!entry.is_directory(dirEc) || dirEc)
      continue;

    std::error_code timeEc;
    auto mtime = std::filesystem::last_write_time(entry.path(), timeEc);
    if (timeEc)
      mtime = std::filesystem::file_time_type::min();

    directories.push_back({entry.path().string(), mtime});
  }

  if (directories.size() <= keepCount)
    return llvm::Error::success();

  std::sort(directories.begin(), directories.end(),
            [](const RetainedProfileDirectory &lhs,
               const RetainedProfileDirectory &rhs) {
    if (lhs.mtime != rhs.mtime)
      return lhs.mtime > rhs.mtime;
    return lhs.path < rhs.path;
  });

  for (size_t index = keepCount; index < directories.size(); ++index) {
    std::error_code removeEc;
    (void)std::filesystem::remove_all(directories[index].path, removeEc);
  }

  return llvm::Error::success();
}

} // namespace

llvm::Error pruneRetainedProfileDirectories(llvm::StringRef root,
                                            size_t keepCount) {
  return pruneRetainedProfileDirectoriesImpl(root, keepCount);
}

llvm::Error pruneRetainedProfileDirectoriesForTest(llvm::StringRef root,
                                                   size_t keepCount) {
  return pruneRetainedProfileDirectoriesImpl(root, keepCount);
}

} // namespace mlir::runtime
