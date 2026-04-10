// lib/Runtime/ExecutionSession.cpp
#include "Runtime/ExecutionSession.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <filesystem>
#include <utility>

namespace mlir::runtime {

namespace {

struct SessionRuntimePaths {
  std::string sessionId;
  std::string workingDirectory;
};

static llvm::Expected<SessionRuntimePaths> prepareSessionRuntimePaths() {
  std::error_code tempDirError;
  const std::filesystem::path sessionRoot =
      std::filesystem::temp_directory_path(tempDirError) / "ascendc-runtime";
  if (tempDirError)
    return llvm::createStringError(
        tempDirError, "Cannot determine temp directory for runtime sessions");

  if (auto ec = llvm::sys::fs::create_directories(sessionRoot.string()))
    return llvm::createStringError(ec, "Cannot create session root directory: %s",
                                   sessionRoot.string().c_str());

  llvm::SmallString<256> workingDirectory;
  if (auto ec = llvm::sys::fs::createUniqueDirectory(
          (sessionRoot.string() + "/runtime-session-"), workingDirectory))
    return llvm::createStringError(ec, "Cannot create session directory in %s",
                                   sessionRoot.string().c_str());

  return SessionRuntimePaths{
      llvm::sys::path::filename(workingDirectory).str(),
      workingDirectory.str().str(),
  };
}

} // namespace

ExecutionSession::ExecutionSession(ExecutionBackendKind backendKind)
    : backendKind_(backendKind) {}

ExecutionSession::ExecutionSession(ExecutionBackendKind backendKind,
                                   std::shared_ptr<ExecutionBackendDriver> driver)
    : backendKind_(backendKind), driver_(std::move(driver)) {}

ExecutionSession::~ExecutionSession() {
  for (const std::string &workingDirectory : workingDirectories_) {
    std::error_code ec;
    std::filesystem::remove_all(workingDirectory, ec);
  }
}

llvm::Expected<SessionPlan> ExecutionSession::plan(const TaskGraph &graph) const {
  auto orderOr = graph.topologicalOrder();
  if (!orderOr)
    return orderOr.takeError();
  return SessionPlan{std::move(*orderOr)};
}

llvm::Expected<ProfileTrace> ExecutionSession::run(const TaskGraph &graph) {
  auto orderedTasksOr = graph.executionOrder();
  if (!orderedTasksOr)
    return orderedTasksOr.takeError();

  auto backendOr = getOrCreateBackend();
  if (!backendOr)
    return backendOr.takeError();
  ExecutionBackend &backend = *backendOr;

  auto runtimePathsOr = prepareSessionRuntimePaths();
  if (!runtimePathsOr)
    return runtimePathsOr.takeError();

  ProfileTrace sessionTrace;
  sessionTrace.sessionId = runtimePathsOr->sessionId;
  const std::string &workingDirectory = runtimePathsOr->workingDirectory;
  workingDirectories_.push_back(workingDirectory);

  for (const RuntimeTask &task : *orderedTasksOr) {
    ExecutionRequest request;
    request.sessionId = sessionTrace.sessionId;
    request.task = task;
    request.workingDirectory = workingDirectory;

    auto resultOr = backend.run(request);
    if (!resultOr)
      return resultOr.takeError();

    if (resultOr->profileTrace) {
      for (ProfileEvent event : resultOr->profileTrace->events)
        sessionTrace.addEvent(std::move(event));
    }
  }

  return sessionTrace;
}

llvm::Expected<ExecutionBackend &> ExecutionSession::getOrCreateBackend() {
  if (!backend_) {
    auto backendOr = createExecutionBackend(backendKind_, driver_);
    if (!backendOr)
      return backendOr.takeError();
    backend_ = std::move(*backendOr);
  }
  return *backend_;
}

} // namespace mlir::runtime
