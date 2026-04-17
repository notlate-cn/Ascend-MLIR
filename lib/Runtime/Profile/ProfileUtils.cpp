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

static std::pair<std::optional<int64_t>, std::optional<int64_t>>
readProfileMetricsFromArtifact(llvm::StringRef path) {
  auto bufferOr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOr)
    return {std::nullopt, std::nullopt};

  auto parsedOr = llvm::json::parse(bufferOr.get()->getBuffer());
  if (!parsedOr)
    return {std::nullopt, std::nullopt};

  const auto *object = parsedOr->getAsObject();
  if (!object)
    return {std::nullopt, std::nullopt};

  std::optional<int64_t> score = object->getInteger("score");
  std::optional<int64_t> cycleCount = object->getInteger("cycle_count");
  if (!score && cycleCount)
    score = cycleCount;
  if (!cycleCount && score)
    cycleCount = score;
  return {score, cycleCount};
}

static void writeJsonEscapedString(llvm::raw_ostream &os, llvm::StringRef value) {
  os << '"';
  for (char ch : value) {
    switch (ch) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\b':
      os << "\\b";
      break;
    case '\f':
      os << "\\f";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      os << ch;
      break;
    }
  }
  os << '"';
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
    auto [score, cycleCount] = readProfileMetricsFromArtifact(file);
    trace.addEvent(makeProfileArtifactEvent(taskId,
                                            ExecutionBackendKind::Simulation,
                                            file, score, cycleCount));
  }

  if (trace.events.empty())
    return std::nullopt;
  return trace;
}

ProfileEvent makeProfileArtifactEvent(llvm::StringRef taskId,
                                      ExecutionBackendKind backend,
                                      llvm::StringRef artifactPath,
                                      std::optional<int64_t> score,
                                      std::optional<int64_t> cycleCount) {
  return ProfileEvent{taskId.str(), backend, "profile_artifact",
                      artifactPath.str(), score, cycleCount};
}

void addProfileArtifact(ProfileTrace &trace, llvm::StringRef taskId,
                        ExecutionBackendKind backend,
                        llvm::StringRef artifactPath,
                        std::optional<int64_t> score,
                        std::optional<int64_t> cycleCount) {
  trace.addEvent(
      makeProfileArtifactEvent(taskId, backend, artifactPath, score,
                               cycleCount));
}

std::string retainedProfileSessionDirectory(llvm::StringRef destinationRoot,
                                            llvm::StringRef sessionId) {
  llvm::SmallString<256> retainedSessionDir(destinationRoot);
  llvm::sys::path::append(retainedSessionDir, sessionId);
  return retainedSessionDir.str().str();
}

std::string retainedProfileSessionSummaryPath(llvm::StringRef destinationRoot,
                                              llvm::StringRef sessionId) {
  llvm::SmallString<256> summaryPath(
      retainedProfileSessionDirectory(destinationRoot, sessionId));
  llvm::sys::path::append(summaryPath, "session_summary.json");
  return summaryPath.str().str();
}

llvm::Expected<ProfileTrace>
retainProfileArtifactsForCli(const ProfileTrace &trace,
                             llvm::StringRef destinationRoot) {
  ProfileTrace retained = trace;
  llvm::SmallString<256> retainedSessionDir(
      retainedProfileSessionDirectory(destinationRoot, trace.sessionId));
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

    std::pair<int64_t, int64_t> metrics;
    if (event.score && event.cycleCount) {
      metrics = std::make_pair(*event.score, *event.cycleCount);
    } else {
      auto metricsOr = parseRetainedTaskMetrics(event.taskId, retainedPath.str());
      if (!metricsOr)
        return metricsOr.takeError();
      metrics = *metricsOr;
    }

    if (!summaryBackend)
      summaryBackend = event.backend;

    event.artifact = retainedPath.str().str();
    taskSummaries.push_back(
        {event.taskId, event.artifact, metrics.first, metrics.second,
         event.backend});
  }

  int64_t totalScore = 0;
  int64_t totalCycleCount = 0;
  for (const RetainedTaskSummary &task : taskSummaries) {
    totalScore += task.score;
    totalCycleCount += task.cycleCount;
  }

  llvm::SmallString<256> summaryPath(
      retainedProfileSessionSummaryPath(destinationRoot, trace.sessionId));
  std::error_code writeError;
  llvm::raw_fd_ostream summaryStream(summaryPath, writeError,
                                     llvm::sys::fs::OF_Text);
  if (writeError) {
    return llvm::createStringError(writeError,
                                   "cannot write retained session summary: %s",
                                   summaryPath.str().str().c_str());
  }
  summaryStream << "{\n";
  summaryStream << "  \"schema_version\": 1,\n";
  summaryStream << "  \"session_id\": ";
  writeJsonEscapedString(summaryStream, trace.sessionId);
  summaryStream << ",\n";
  summaryStream << "  \"backend\": ";
  writeJsonEscapedString(
      summaryStream,
      backendName(summaryBackend.value_or(ExecutionBackendKind::Simulation)));
  summaryStream << ",\n";
  summaryStream << "  \"task_count\": " << taskSummaries.size() << ",\n";
  summaryStream << "  \"successful_task_count\": " << taskSummaries.size()
                << ",\n";
  summaryStream << "  \"failed_task_count\": 0,\n";
  summaryStream << "  \"tasks\": [\n";
  for (size_t index = 0; index < taskSummaries.size(); ++index) {
    const RetainedTaskSummary &task = taskSummaries[index];
    summaryStream << "    {\n";
    summaryStream << "      \"task_id\": ";
    writeJsonEscapedString(summaryStream, task.taskId);
    summaryStream << ",\n";
    summaryStream << "      \"profile_path\": ";
    writeJsonEscapedString(summaryStream, task.profilePath);
    summaryStream << ",\n";
    summaryStream << "      \"score\": " << task.score << ",\n";
    summaryStream << "      \"cycle_count\": " << task.cycleCount << "\n";
    summaryStream << "    }";
    if (index + 1 != taskSummaries.size())
      summaryStream << ",";
    summaryStream << "\n";
  }
  summaryStream << "  ],\n";
  summaryStream << "  \"runtime\": {\n";
  summaryStream << "    \"attributes\": {\n";
  for (auto it = retained.attributes.begin(); it != retained.attributes.end();
       ++it) {
    summaryStream << "      ";
    writeJsonEscapedString(summaryStream, it->first);
    summaryStream << ": ";
    writeJsonEscapedString(summaryStream, it->second);
    if (std::next(it) != retained.attributes.end())
      summaryStream << ",";
    summaryStream << "\n";
  }
  summaryStream << "    },\n";
  summaryStream << "    \"counters\": {\n";
  for (auto it = retained.counters.begin(); it != retained.counters.end();
       ++it) {
    summaryStream << "      ";
    writeJsonEscapedString(summaryStream, it->first);
    summaryStream << ": " << it->second;
    if (std::next(it) != retained.counters.end())
      summaryStream << ",";
    summaryStream << "\n";
  }
  summaryStream << "    }\n";
  summaryStream << "  },\n";
  summaryStream << "  \"total_score\": " << totalScore << ",\n";
  summaryStream << "  \"total_cycle_count\": " << totalCycleCount << "\n";
  summaryStream << "}\n";
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

llvm::Expected<RetainedProfileCliArtifacts>
retainProfileArtifactsForCliRun(const ProfileTrace &trace,
                                llvm::StringRef destinationRoot) {
  auto retainedTraceOr = retainProfileArtifactsForCli(trace, destinationRoot);
  if (!retainedTraceOr)
    return retainedTraceOr.takeError();

  return RetainedProfileCliArtifacts{
      std::move(*retainedTraceOr),
      retainedProfileSessionSummaryPath(destinationRoot, trace.sessionId),
  };
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

size_t retainedProfilePruneKeepCountForNewSession(size_t sessionLimit) {
  return sessionLimit > 0 ? sessionLimit - 1 : 0;
}

llvm::Expected<std::string>
prepareRetainedProfileRunRootForCli(llvm::StringRef destinationRoot,
                                    size_t sessionLimit) {
  if (auto ec = llvm::sys::fs::create_directories(destinationRoot))
    return llvm::createStringError(
        ec, "cannot create retained profile directory: %s",
        destinationRoot.str().c_str());

  if (auto pruneErr = pruneRetainedProfileDirectories(
          destinationRoot,
          retainedProfilePruneKeepCountForNewSession(sessionLimit)))
    return std::move(pruneErr);

  return destinationRoot.str();
}

} // namespace mlir::runtime
