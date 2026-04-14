#pragma once

#include "Runtime/Artifact/ArtifactCompiler.h"
#include "Runtime/Execution/TaskGraph.h"
#include "Runtime/Profile/ProfileTrace.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>

namespace mlir::runtime {

struct FrontendCompileInput {
  std::string kernelSource;
  std::string outputDir;
  std::string kernelName;
  std::string socVersion;
  std::string arch;
  KernelKind kernelKind = KernelKind::Mix;
  std::optional<std::string> cannMlirPath;
  std::optional<std::string> npyDir;
  int optLevel = 3;
};

llvm::Expected<ArtifactCompileRequest>
buildFrontendCompileRequest(const FrontendCompileInput &input);

struct FrontendSingleTaskRunRequest {
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  std::string taskId = "main";
  std::optional<KernelArtifact> artifact;
  std::string artifactRoot;
  ExecutionInvocation invocation;
};

struct FrontendPreparedRun {
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  TaskGraph graph;
};

llvm::Expected<FrontendPreparedRun>
prepareFrontendSingleTaskRun(const FrontendSingleTaskRunRequest &request);

enum class FrontendValidationStatus {
  NotRun,
  Passed,
  Failed,
};

struct FrontendRunSummary {
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  bool success = false;
  FrontendValidationStatus validationStatus = FrontendValidationStatus::NotRun;
  std::string errorStage;
  std::string errorMessage;
  ProfileTrace profileTrace;
  std::string retainedSummaryPath;
};

FrontendRunSummary summarizeFrontendRunSuccess(ExecutionBackendKind backendKind,
                                              bool validationRan,
                                              const ProfileTrace &trace,
                                              llvm::StringRef retainedSummaryPath = "");

FrontendRunSummary summarizeFrontendRunError(ExecutionBackendKind backendKind,
                                            bool validationRan,
                                            llvm::StringRef message);

} // namespace mlir::runtime
