#include "Runtime/ProfileUtils.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <set>
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
  llvm::SmallString<256> retainedSessionDir(destinationRoot);
  llvm::sys::path::append(retainedSessionDir, trace.sessionId);
  llvm::SmallString<256> retainedTaskDir(retainedSessionDir);
  llvm::sys::path::append(retainedTaskDir, "tasks");

  if (auto ec = llvm::sys::fs::create_directories(retainedTaskDir))
    return llvm::createStringError(
        ec, "cannot create retained profile directory: %s",
        retainedTaskDir.str().str().c_str());

  struct RetainedTaskSummary {
    std::string taskId;
    std::string profilePath;
    int64_t score = 0;
    int64_t cycleCount = 0;
    ExecutionBackendKind backend = ExecutionBackendKind::Simulation;
  };

  auto backendName = [](ExecutionBackendKind backend) -> llvm::StringRef {
    switch (backend) {
    case ExecutionBackendKind::Simulation:
      return "simulation";
    case ExecutionBackendKind::Npu:
      return "npu";
    }
    return "unknown";
  };

  auto parseRetainedTaskMetrics =
      [](llvm::StringRef taskId, llvm::StringRef retainedPath)
      -> llvm::Expected<std::pair<int64_t, int64_t>> {
    auto bufferOr = llvm::MemoryBuffer::getFile(retainedPath);
    if (!bufferOr) {
      return llvm::createStringError(bufferOr.getError(),
                                     "cannot read retained profile artifact %s",
                                     retainedPath.str().c_str());
    }

    auto parsedOr = llvm::json::parse(bufferOr.get()->getBuffer());
    if (!parsedOr) {
      llvm::Error parseError = parsedOr.takeError();
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "cannot parse retained profile artifact for task %s: %s",
          taskId.str().c_str(), llvm::toString(std::move(parseError)).c_str());
    }

    const auto *object = parsedOr->getAsObject();
    if (!object) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "retained profile artifact for task %s must be a json object",
          taskId.str().c_str());
    }

    std::optional<int64_t> score = object->getInteger("score");
    std::optional<int64_t> cycleCount = object->getInteger("cycle_count");
    if (!score && cycleCount)
      score = cycleCount;
    if (!cycleCount && score)
      cycleCount = score;
    if (!score || !cycleCount) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "retained profile artifact for task %s must provide score or cycle_count",
          taskId.str().c_str());
    }

    return std::make_pair(*score, *cycleCount);
  };

  std::vector<RetainedTaskSummary> taskSummaries;
  taskSummaries.reserve(retained.events.size());
  std::set<std::string> retainedTaskIds;
  std::optional<ExecutionBackendKind> summaryBackend;

  for (ProfileEvent &event : retained.events) {
    if (event.eventKind != "profile_artifact" || event.artifact.empty())
      continue;

    if (!retainedTaskIds.insert(event.taskId).second) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "duplicate retained profile task id: %s", event.taskId.c_str());
    }

    llvm::StringRef sourcePath = event.artifact;
    llvm::SmallString<256> retainedPath(retainedTaskDir);
    llvm::sys::path::append(retainedPath, event.taskId + ".json");

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

    auto metricsOr = parseRetainedTaskMetrics(event.taskId, retainedPath.str());
    if (!metricsOr)
      return metricsOr.takeError();

    if (!summaryBackend)
      summaryBackend = event.backend;

    event.artifact = retainedPath.str().str();
    taskSummaries.push_back(
        {event.taskId, event.artifact, metricsOr->first, metricsOr->second,
         event.backend});
  }

  int64_t totalScore = 0;
  int64_t totalCycleCount = 0;
  llvm::json::Array tasksJson;
  for (const RetainedTaskSummary &task : taskSummaries) {
    totalScore += task.score;
    totalCycleCount += task.cycleCount;

    llvm::json::Object taskObject;
    taskObject["task_id"] = task.taskId;
    taskObject["profile_path"] = task.profilePath;
    taskObject["score"] = task.score;
    taskObject["cycle_count"] = task.cycleCount;
    tasksJson.emplace_back(std::move(taskObject));
  }

  llvm::json::Object summaryObject;
  summaryObject["schema_version"] = 1;
  summaryObject["session_id"] = trace.sessionId;
  summaryObject["backend"] =
      backendName(summaryBackend.value_or(ExecutionBackendKind::Simulation)).str();
  summaryObject["task_count"] = static_cast<int64_t>(taskSummaries.size());
  summaryObject["successful_task_count"] =
      static_cast<int64_t>(taskSummaries.size());
  summaryObject["failed_task_count"] = 0;
  summaryObject["tasks"] = std::move(tasksJson);
  summaryObject["total_score"] = totalScore;
  summaryObject["total_cycle_count"] = totalCycleCount;

  llvm::SmallString<256> summaryPath(retainedSessionDir);
  llvm::sys::path::append(summaryPath, "session_summary.json");
  std::error_code writeError;
  llvm::raw_fd_ostream summaryStream(summaryPath, writeError,
                                     llvm::sys::fs::OF_Text);
  if (writeError) {
    return llvm::createStringError(writeError,
                                   "cannot write retained session summary: %s",
                                   summaryPath.str().str().c_str());
  }
  summaryStream << llvm::formatv("{0:2}",
                                 llvm::json::Value(std::move(summaryObject)))
                << "\n";
  summaryStream.flush();
  if (summaryStream.has_error()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "cannot flush retained session summary: %s",
        summaryPath.str().str().c_str());
  }
  summaryStream.close();
  if (summaryStream.has_error()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "cannot finalize retained session summary: %s",
        summaryPath.str().str().c_str());
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
