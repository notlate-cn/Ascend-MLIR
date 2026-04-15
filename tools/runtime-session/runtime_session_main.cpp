#include "Runtime/RuntimeSessionRequestBuilder.h"
#include "Runtime/RuntimeFrontendCore.h"
#include "Runtime/ToolDiscovery.h"
#include "Runtime/ExecutionBackend.h"
#include "Runtime/ExecutionSession.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace mlir::runtime;

static constexpr size_t RetainedProfileSessionLimit = 20;
static constexpr size_t RuntimeSessionWorkdirLimit = 20;

llvm::cl::OptionCategory RuntimeSessionCategory("runtime-session options");

llvm::cl::opt<std::string> ArtifactRoot(
    "artifact-root",
    llvm::cl::desc("Use an existing artifact root with a real manifest"),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<std::string> KernelFile(
    "kernel",
    llvm::cl::desc("Kernel source to compile into a runtime artifact"),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<std::string> KernelName(
    "name",
    llvm::cl::desc("Kernel name override when compiling a new artifact"),
    llvm::cl::init(""),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<std::string> OutputDir(
    "output",
    llvm::cl::desc("Artifact output directory when compiling"),
    llvm::cl::init("./build/runtime-session-artifact"),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<std::string> SocVersion(
    "soc",
    llvm::cl::desc("Target SoC version when compiling a new artifact"),
    llvm::cl::init("Ascend910B1"),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<std::string> KernelKindName(
    "kernel-kind",
    llvm::cl::desc("Kernel kind when compiling: vec, cube, or mix"),
    llvm::cl::init("mix"),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<std::string> TaskId(
    "task-id",
    llvm::cl::desc("Task id to register in the task graph"),
    llvm::cl::init("main"),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<bool> RunSession(
    "run",
    llvm::cl::desc("Execute the prepared task graph runtime session"),
    llvm::cl::init(false),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<std::string> CannMlir(
    "cann-mlir",
    llvm::cl::desc("Path to step7_cann.mlir for mix artifact compilation"),
    llvm::cl::init(""),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<std::string> NpyDir(
    "npy-dir",
    llvm::cl::desc("Directory containing runtime .npy files for ABI shaping"),
    llvm::cl::init(""),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<std::string> RunManifestPath(
    "run-manifest",
    llvm::cl::desc("JSON manifest describing artifact root, bindings, and execution settings"),
    llvm::cl::init(""),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<std::string> TestingDriver(
    "testing-driver",
    llvm::cl::desc("Testing-only backend driver injection"),
    llvm::cl::init(""),
    llvm::cl::Hidden,
    llvm::cl::cat(RuntimeSessionCategory));

class SuccessfulNpuTestingDriver final : public ExecutionBackendDriver {
public:
  llvm::Expected<ExecutionResult>
  run(const ExecutionRequest &request) override {
    ExecutionResult result;
    result.taskId = request.task.taskId;
    for (const TensorBinding &binding : request.task.invocation.outputs)
      result.producedFiles.push_back(binding.path);
    ProfileTrace trace;
    trace.sessionId = request.sessionId;
    trace.addProfileArtifact(request.task.taskId, ExecutionBackendKind::Npu,
                             request.workingDirectory + "/" +
                                 request.task.taskId + ".npu-profile.json");
    result.profileTrace = std::move(trace);
    return result;
  }
};

llvm::Expected<KernelKind> parseKernelKind(llvm::StringRef name) {
  if (name == "vec")
    return KernelKind::Vec;
  if (name == "cube")
    return KernelKind::Cube;
  if (name == "mix")
    return KernelKind::Mix;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported kernel kind: %s",
                                 name.str().c_str());
}

void printArtifactSummary(const KernelArtifact &artifact) {
  llvm::outs() << "artifact.kernel_name=" << artifact.kernelName << "\n";
  llvm::outs() << "artifact.root=" << artifact.artifactRoot << "\n";
  llvm::outs() << "artifact.manifest=" << artifact.manifestPath << "\n";
  llvm::outs() << "artifact.soc=" << artifact.socVersion << "\n";
}

void printPlan(const SessionPlan &plan) {
  llvm::outs() << "session.plan.tasks=" << plan.orderedTaskIds.size() << "\n";
  for (size_t i = 0; i < plan.orderedTaskIds.size(); ++i)
    llvm::outs() << "session.plan[" << i << "]=" << plan.orderedTaskIds[i]
                 << "\n";
}

llvm::StringRef backendName(ExecutionBackendKind backendKind) {
  switch (backendKind) {
  case ExecutionBackendKind::Simulation:
    return "sim";
  case ExecutionBackendKind::Npu:
    return "npu";
  }
  return "unknown";
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

void printRunSuccessSummary(const FrontendRunSummary &summary) {
  llvm::outs() << "session.backend=" << backendName(summary.backendKind)
               << "\n";
  llvm::outs() << "session.result=success\n";
  if (summary.validationStatus == FrontendValidationStatus::Passed)
    llvm::outs() << "session.validation=pass\n";
  llvm::outs() << "session.profile.session_id=" << summary.profileTrace.sessionId
               << "\n";
  llvm::outs() << "session.profile.count=" << summary.profileArtifactPaths.size()
               << "\n";
  for (size_t i = 0; i < summary.profileArtifactPaths.size(); ++i)
    llvm::outs() << "session.profile[" << i << "]="
                 << summary.profileArtifactPaths[i] << "\n";
  if (!summary.retainedSummaryPath.empty() &&
      std::filesystem::exists(summary.retainedSummaryPath))
    llvm::outs() << "session.profile.summary=" << summary.retainedSummaryPath
                 << "\n";
}

void printRunErrorSummary(const FrontendRunSummary &summary) {
  llvm::errs() << "session.backend=" << backendName(summary.backendKind)
               << "\n";
  llvm::errs() << "session.result=error\n";
  if (summary.validationStatus == FrontendValidationStatus::Failed)
    llvm::errs() << "session.validation=fail\n";
  if (!summary.errorStage.empty())
    llvm::errs() << "session.error_stage=" << summary.errorStage << "\n";
  llvm::errs() << "session.error=" << summary.errorMessage << "\n";
}

llvm::Expected<std::shared_ptr<ExecutionBackendDriver>>
createTestingDriver(ExecutionBackendKind backendKind) {
  if (TestingDriver.empty())
    return std::shared_ptr<ExecutionBackendDriver>{};
  if (TestingDriver == "npu-success") {
    if (backendKind != ExecutionBackendKind::Npu) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "testing driver npu-success requires backend=npu");
    }
    return std::make_shared<SuccessfulNpuTestingDriver>();
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported testing driver: %s",
                                 TestingDriver.getValue().c_str());
}

llvm::Expected<std::string> retainedProfileBaseDirectory() {
  std::error_code tempDirError;
  const std::filesystem::path retainRoot =
      std::filesystem::temp_directory_path(tempDirError) /
      "ascendc-runtime-profiles";
  if (tempDirError) {
    return llvm::createStringError(
        tempDirError,
        "cannot determine temp directory for retained profile artifacts");
  }

  if (auto ec = llvm::sys::fs::create_directories(retainRoot.string()))
    return llvm::createStringError(ec,
                                   "cannot create retained profile directory: %s",
                                   retainRoot.string().c_str());
  return retainRoot.string();
}

llvm::Expected<std::string> runtimeSessionWorkdirBaseDirectory() {
  std::error_code tempDirError;
  const std::filesystem::path sessionRoot =
      std::filesystem::temp_directory_path(tempDirError) / "ascendc-runtime";
  if (tempDirError) {
    return llvm::createStringError(
        tempDirError, "cannot determine temp directory for runtime sessions");
  }

  return sessionRoot.string();
}

} // namespace

int main(int argc, char **argv) {
  configureSiblingToolPathEnv("AFIR_MIX_TILING_HELPER", argv[0],
                              "mix-tiling-helper");
  llvm::cl::HideUnrelatedOptions(RuntimeSessionCategory);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "task graph runtime planning and execution CLI for artifacts and session graphs\n");

  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  std::optional<KernelArtifact> artifact;
  std::optional<TaskGraph> graph;
  if (!RunManifestPath.empty()) {
    auto manifestGraphOr = prepareRuntimeSessionGraphFromManifest(RunManifestPath);
    if (!manifestGraphOr) {
      llvm::errs() << "Error: " << llvm::toString(manifestGraphOr.takeError())
                   << "\n";
      return 4;
    }
    backendKind = manifestGraphOr->first;
    graph = std::move(manifestGraphOr->second);
  } else {
    const bool hasArtifactRoot = !ArtifactRoot.empty();
    const bool hasKernelFile = !KernelFile.empty();
    if (hasArtifactRoot == hasKernelFile) {
      llvm::errs() << "Error: provide exactly one of --artifact-root or --kernel\n";
      return 4;
    }

    RuntimeSessionArtifactRequest request;
    if (hasArtifactRoot) {
      request.artifactRoot = ArtifactRoot;
    } else {
      auto kernelKindOr = parseKernelKind(KernelKindName);
      if (!kernelKindOr) {
        llvm::errs() << "Error: " << llvm::toString(kernelKindOr.takeError())
                     << "\n";
        return 4;
      }
      request.kernelSource = KernelFile;
      request.kernelName = KernelName;
      request.kernelKind = *kernelKindOr;
      request.outputDir = OutputDir;
      request.socVersion = SocVersion;
      if (!CannMlir.empty())
        request.cannMlirPath = CannMlir;
      if (!NpyDir.empty())
        request.npyDir = NpyDir;
    }

    auto artifactOr = prepareRuntimeSessionArtifact(request);
    if (!artifactOr) {
      llvm::errs() << "Error: " << llvm::toString(artifactOr.takeError()) << "\n";
      return 4;
    }
    artifact = *artifactOr;
    printArtifactSummary(*artifact);
    FrontendSingleTaskRunRequest runRequest;
    runRequest.backendKind = backendKind;
    runRequest.taskId = TaskId;
    runRequest.artifact = *artifact;
    auto preparedRunOr = prepareFrontendSingleTaskRun(runRequest);
    if (!preparedRunOr) {
      llvm::errs() << "Error: " << llvm::toString(preparedRunOr.takeError())
                   << "\n";
      return 4;
    }
    backendKind = preparedRunOr->backendKind;
    graph = std::move(preparedRunOr->graph);
  }

  auto testingDriverOr = createTestingDriver(backendKind);
  if (!testingDriverOr) {
    llvm::errs() << "Error: " << llvm::toString(testingDriverOr.takeError())
                 << "\n";
    return 4;
  }

  ExecutionSession session(backendKind, *testingDriverOr);
  auto planOr = session.plan(*graph);
  if (!planOr) {
    llvm::errs() << "Error: " << llvm::toString(planOr.takeError()) << "\n";
    return 2;
  }
  printPlan(*planOr);

  if (!RunSession)
    return 0;

  const bool validationRan = graphRequestsValidation(*graph);
  auto runSession =
      std::make_unique<ExecutionSession>(backendKind, *testingDriverOr);
  FrontendRunOptions runOptions;
  if (backendKind == ExecutionBackendKind::Simulation) {
    auto sessionRootOr = runtimeSessionWorkdirBaseDirectory();
    if (!sessionRootOr) {
      FrontendRunSummary summary = summarizeFrontendRunError(
          backendKind, validationRan, llvm::toString(sessionRootOr.takeError()));
      printRunErrorSummary(summary);
      llvm::errs() << "Error: " << summary.rawErrorMessage << "\n";
      return 2;
    }
    if (auto preparedSessionRootOr = prepareRuntimeSessionWorkdirRootForCli(
            *sessionRootOr, RuntimeSessionWorkdirLimit);
        !preparedSessionRootOr) {
      FrontendRunSummary summary =
          summarizeFrontendRunError(backendKind, validationRan,
                                    llvm::toString(preparedSessionRootOr.takeError()));
      printRunErrorSummary(summary);
      llvm::errs() << "Error: " << summary.rawErrorMessage << "\n";
      return 2;
    }

    auto retainBaseDirOr = retainedProfileBaseDirectory();
    if (!retainBaseDirOr) {
      FrontendRunSummary summary = summarizeFrontendRunError(
          backendKind, validationRan, llvm::toString(retainBaseDirOr.takeError()));
      printRunErrorSummary(summary);
      llvm::errs() << "Error: " << summary.rawErrorMessage << "\n";
      return 2;
    }
    runOptions.retainSimulationProfiles = true;
    runOptions.retainedProfileRoot = *retainBaseDirOr;
    runOptions.retainedProfileSessionLimit = RetainedProfileSessionLimit;
  }

  FrontendPreparedRun prepared;
  prepared.backendKind = backendKind;
  prepared.graph = std::move(*graph);
  FrontendRunSummary summary =
      executeFrontendPreparedRun(*runSession, prepared, runOptions);
  if (!summary.success) {
    printRunErrorSummary(summary);
    llvm::errs() << "Error: " << summary.rawErrorMessage << "\n";
    return 2;
  }

  printRunSuccessSummary(summary);
  if (backendKind == ExecutionBackendKind::Simulation) {
    runSession->releaseWorkingDirectoriesForProcessExit();
    runSession.reset();
    llvm::outs().flush();
    llvm::errs().flush();
    _Exit(0);
  }
  return 0;
}
