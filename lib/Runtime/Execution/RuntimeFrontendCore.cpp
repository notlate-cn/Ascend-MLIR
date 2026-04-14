#include "Runtime/Execution/RuntimeFrontendCore.h"

#include "Runtime/Artifact/RuntimeSessionRequestBuilder.h"
#include "Runtime/Profile/ProfileUtils.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Error.h"

namespace mlir::runtime {

llvm::Expected<ArtifactCompileRequest>
buildFrontendCompileRequest(const FrontendCompileInput &input) {
  ArtifactCompileRequest request;
  request.kernelSource = input.kernelSource;
  request.outputDir = input.outputDir;
  request.kernelName = input.kernelName;
  request.socVersion = input.socVersion;
  request.arch = input.arch;
  request.kernelKind = input.kernelKind;
  request.optLevel = input.optLevel;
  request.cannMlirPath = input.cannMlirPath;
  request.npyDir = input.npyDir;
  return request;
}

llvm::Expected<FrontendPreparedRun>
prepareFrontendSingleTaskRun(const FrontendSingleTaskRunRequest &request) {
  const bool hasArtifact = request.artifact.has_value();
  const bool hasArtifactRoot = !request.artifactRoot.empty();
  if (hasArtifact == hasArtifactRoot) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "provide exactly one of artifact or artifact root for frontend single-task run");
  }

  KernelArtifact artifact;
  if (hasArtifact) {
    artifact = *request.artifact;
  } else {
    auto artifactOr = loadRuntimeSessionArtifactFromRoot(request.artifactRoot);
    if (!artifactOr)
      return artifactOr.takeError();
    artifact = *artifactOr;
  }

  TaskGraph graph;
  RuntimeTask task;
  task.taskId = request.taskId;
  task.artifact = std::move(artifact);
  task.invocation = request.invocation;
  if (auto err = graph.addTask(task))
    return std::move(err);

  FrontendPreparedRun prepared;
  prepared.backendKind = request.backendKind;
  prepared.graph = std::move(graph);
  return prepared;
}

namespace {

std::pair<std::string, std::string> parseErrorStage(llvm::StringRef message) {
  if (!message.starts_with("["))
    return {"", message.str()};
  const size_t end = message.find(']');
  if (end == llvm::StringRef::npos || end <= 1)
    return {"", message.str()};
  llvm::StringRef prefix = message.slice(1, end);
  const size_t split = prefix.find(':');
  if (split == llvm::StringRef::npos || split + 1 >= prefix.size())
    return {"", message.str()};
  std::string stage = prefix.drop_front(split + 1).str();
  llvm::StringRef remainder = message.drop_front(end + 1).trim();
  return {std::move(stage), remainder.str()};
}

bool graphRequestsValidation(const TaskGraph &graph) {
  auto tasksOr = graph.orderedTasks();
  if (!tasksOr)
    return false;
  for (const RuntimeTask &task : *tasksOr) {
    if (!task.invocation.expectedOutputs.empty())
      return true;
  }
  return false;
}

} // namespace

FrontendRunSummary summarizeFrontendRunSuccess(ExecutionBackendKind backendKind,
                                              bool validationRan,
                                              const ProfileTrace &trace,
                                              llvm::StringRef retainedSummaryPath) {
  FrontendRunSummary summary;
  summary.backendKind = backendKind;
  summary.success = true;
  summary.validationStatus = validationRan ? FrontendValidationStatus::Passed
                                           : FrontendValidationStatus::NotRun;
  summary.profileTrace = trace;
  summary.profileArtifactPaths = trace.profileArtifactPaths();
  summary.retainedSummaryPath = retainedSummaryPath.str();
  return summary;
}

FrontendRunSummary summarizeFrontendRunError(ExecutionBackendKind backendKind,
                                            bool validationRan,
                                            llvm::StringRef message) {
  FrontendRunSummary summary;
  summary.backendKind = backendKind;
  summary.success = false;
  summary.rawErrorMessage = message.str();
  auto [stage, detail] = parseErrorStage(message);
  summary.errorStage = std::move(stage);
  summary.errorMessage = std::move(detail);
  if (validationRan && summary.errorStage == "validate")
    summary.validationStatus = FrontendValidationStatus::Failed;
  return summary;
}

FrontendRunSummary executeFrontendPreparedRun(ExecutionSession &session,
                                              const FrontendPreparedRun &prepared,
                                              const FrontendRunOptions &options) {
  const bool validationRan = graphRequestsValidation(prepared.graph);

  if (prepared.backendKind == ExecutionBackendKind::Simulation &&
      options.retainSimulationProfiles) {
    if (options.retainedProfileRoot.empty()) {
      return summarizeFrontendRunError(
          prepared.backendKind, validationRan,
          "retained profile root is required when simulation profile retention is enabled");
    }
    if (auto preparedOr = prepareRetainedProfileRunRootForCli(
            options.retainedProfileRoot, options.retainedProfileSessionLimit);
        !preparedOr) {
      return summarizeFrontendRunError(prepared.backendKind, validationRan,
                                       llvm::toString(preparedOr.takeError()));
    }
  }

  auto traceOr = session.run(prepared.graph);
  if (!traceOr) {
    return summarizeFrontendRunError(prepared.backendKind, validationRan,
                                     llvm::toString(traceOr.takeError()));
  }

  ProfileTrace trace = std::move(*traceOr);
  std::string retainedSummaryPath;
  if (prepared.backendKind == ExecutionBackendKind::Simulation &&
      options.retainSimulationProfiles) {
    auto retainedOr =
        retainProfileArtifactsForCliRun(trace, options.retainedProfileRoot);
    if (!retainedOr) {
      return summarizeFrontendRunError(prepared.backendKind, validationRan,
                                       llvm::toString(retainedOr.takeError()));
    }
    trace = std::move(retainedOr->trace);
    retainedSummaryPath = std::move(retainedOr->summaryPath);
  }

  return summarizeFrontendRunSuccess(prepared.backendKind, validationRan, trace,
                                     retainedSummaryPath);
}

size_t runtimeSessionWorkdirPruneKeepCountForNewRun(size_t sessionLimit) {
  return sessionLimit > 0 ? sessionLimit - 1 : 0;
}

llvm::Expected<std::string>
prepareRuntimeSessionWorkdirRootForCli(llvm::StringRef destinationRoot,
                                       size_t sessionLimit) {
  if (auto ec = llvm::sys::fs::create_directories(destinationRoot))
    return llvm::createStringError(
        ec, "cannot create runtime session directory: %s",
        destinationRoot.str().c_str());

  if (auto pruneErr =
          pruneRetainedProfileDirectories(destinationRoot,
                                          runtimeSessionWorkdirPruneKeepCountForNewRun(
                                              sessionLimit)))
    return std::move(pruneErr);

  return destinationRoot.str();
}

} // namespace mlir::runtime
