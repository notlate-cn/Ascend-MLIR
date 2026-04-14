#include "Runtime/Execution/RuntimeFrontendCore.h"

#include "Runtime/Artifact/RuntimeSessionRequestBuilder.h"

#include "llvm/ADT/StringRef.h"
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
  summary.retainedSummaryPath = retainedSummaryPath.str();
  return summary;
}

FrontendRunSummary summarizeFrontendRunError(ExecutionBackendKind backendKind,
                                            bool validationRan,
                                            llvm::StringRef message) {
  FrontendRunSummary summary;
  summary.backendKind = backendKind;
  summary.success = false;
  auto [stage, detail] = parseErrorStage(message);
  summary.errorStage = std::move(stage);
  summary.errorMessage = std::move(detail);
  if (validationRan && summary.errorStage == "validate")
    summary.validationStatus = FrontendValidationStatus::Failed;
  return summary;
}

} // namespace mlir::runtime
