#include "Runtime/RuntimeSessionRequestBuilder.h"
#include "Runtime/RuntimeFrontendCore.h"
#include "Runtime/DebugCase.h"
#include "Runtime/RunManifest.h"
#include "Runtime/TensorDiff.h"
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

#ifndef _WIN32
#include <unistd.h>
#endif

namespace {

using namespace mlir::runtime;

static constexpr size_t RetainedProfileSessionLimit = 20;
static constexpr size_t RuntimeSessionWorkdirLimit = 20;

llvm::cl::OptionCategory RuntimeSessionCategory("runtime-session options");

#ifndef ASCEND_RUNTIME_SESSION_RUN_ONLY
llvm::cl::opt<std::string> ArtifactRoot(
    "artifact-root",
    llvm::cl::desc("Use an existing artifact root with a real manifest"),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<std::string> DebugCasePath(
    "case",
    llvm::cl::desc("User-level case.json to prepare and optionally run"),
    llvm::cl::init(""),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<std::string> ArtifactManifestPath(
    "artifact-manifest",
    llvm::cl::desc("Artifact manifest to prepare into a concrete run manifest"),
    llvm::cl::init(""),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<std::string> EmitRunManifestPath(
    "emit-run-manifest",
    llvm::cl::desc("Write a concrete run manifest prepared from --artifact-manifest"),
    llvm::cl::init(""),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::list<std::string> ShapeArgAssignments(
    "shape-arg",
    llvm::cl::desc("Concrete shape argument for artifact-manifest prepare, formatted as name=value"),
    llvm::cl::ZeroOrMore,
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::list<std::string> InputPathAssignments(
    "input",
    llvm::cl::desc("Concrete artifact-manifest input binding, formatted as name=path or task.name=path"),
    llvm::cl::ZeroOrMore,
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::list<std::string> ExpectedOutputPathAssignments(
    "expected-output",
    llvm::cl::desc("Concrete artifact-manifest expected output binding, formatted as name=path or task.name=path"),
    llvm::cl::ZeroOrMore,
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<bool> EnableProfiling(
    "profiling",
    llvm::cl::desc("Enable profiling in run manifests emitted from --artifact-manifest"),
    llvm::cl::init(false),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<double> PrepareAtol(
    "atol",
    llvm::cl::desc("Absolute validation tolerance for run manifests emitted from --artifact-manifest"),
    llvm::cl::init(1.0),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<double> PrepareRtol(
    "rtol",
    llvm::cl::desc("Relative validation tolerance for run manifests emitted from --artifact-manifest"),
    llvm::cl::init(1e-2),
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
llvm::cl::list<std::string> OutputArgs(
    "output",
    llvm::cl::desc("Artifact output directory when compiling, or artifact-manifest output binding as name=path or task.name=path"),
    llvm::cl::ZeroOrMore,
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
#endif
llvm::cl::opt<bool> RunSession(
    "run",
    llvm::cl::desc("Execute the prepared task graph runtime session"),
    llvm::cl::init(false),
    llvm::cl::cat(RuntimeSessionCategory));
#ifndef ASCEND_RUNTIME_SESSION_RUN_ONLY
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
#endif
llvm::cl::opt<std::string> RunManifestPath(
    "run-manifest",
    llvm::cl::desc("JSON manifest describing artifact root, bindings, and execution settings"),
    llvm::cl::init(""),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<std::string> CompareTensorsPath(
    "compare-tensors",
    llvm::cl::desc("Tensor comparison manifest to evaluate and summarize"),
    llvm::cl::init(""),
    llvm::cl::cat(RuntimeSessionCategory));
llvm::cl::opt<std::string> EmitValidationSummaryPath(
    "emit-validation-summary",
    llvm::cl::desc("Write tensor validation summary JSON"),
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

#ifndef ASCEND_RUNTIME_SESSION_RUN_ONLY
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

llvm::Expected<std::pair<std::string, int64_t>>
parseShapeArgAssignment(llvm::StringRef assignment) {
  auto [name, valueText] = assignment.split('=');
  name = name.trim();
  valueText = valueText.trim();
  if (name.empty() || valueText.empty() || assignment.find('=') == llvm::StringRef::npos) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid --shape-arg assignment: %s",
                                   assignment.str().c_str());
  }
  int64_t value = 0;
  if (valueText.getAsInteger(10, value)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid --shape-arg integer value: %s",
                                   assignment.str().c_str());
  }
  return std::make_pair(name.str(), value);
}

llvm::Expected<ArtifactManifestBindingPath>
parseBindingPathAssignment(llvm::StringRef optionName,
                           llvm::StringRef assignment) {
  auto [selector, path] = assignment.split('=');
  selector = selector.trim();
  path = path.trim();
  if (selector.empty() || path.empty() ||
      assignment.find('=') == llvm::StringRef::npos) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid --%s assignment: %s",
                                   optionName.str().c_str(),
                                   assignment.str().c_str());
  }

  ArtifactManifestBindingPath parsed;
  auto [taskId, bindingName] = selector.split('.');
  if (bindingName.empty()) {
    parsed.bindingName = selector.str();
  } else {
    taskId = taskId.trim();
    bindingName = bindingName.trim();
    if (taskId.empty() || bindingName.empty()) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "invalid --%s binding selector: %s",
                                     optionName.str().c_str(),
                                     selector.str().c_str());
    }
    parsed.taskId = taskId.str();
    parsed.bindingName = bindingName.str();
  }
  parsed.path = path.str();
  return parsed;
}

bool collectBindingAssignments(
    llvm::StringRef optionName, const llvm::cl::list<std::string> &assignments,
    std::vector<ArtifactManifestBindingPath> &out) {
  for (const std::string &assignment : assignments) {
    auto parsedOr = parseBindingPathAssignment(optionName, assignment);
    if (!parsedOr) {
      llvm::errs() << "Error: " << llvm::toString(parsedOr.takeError())
                   << "\n";
      return false;
    }
    out.push_back(std::move(*parsedOr));
  }
  return true;
}

void printArtifactSummary(const KernelArtifact &artifact) {
  llvm::outs() << "artifact.kernel_name=" << artifact.kernelName << "\n";
  llvm::outs() << "artifact.root=" << artifact.artifactRoot << "\n";
  llvm::outs() << "artifact.manifest=" << artifact.manifestPath << "\n";
  llvm::outs() << "artifact.soc=" << artifact.socVersion << "\n";
}
#endif

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
  // Keep observability generic so new runtime attributes/counters flow through unchanged.
  for (const auto &[key, value] : summary.runtimeAttributes)
    llvm::outs() << "session.runtime.attribute." << key << "=" << value << "\n";
  for (const auto &[key, value] : summary.runtimeCounters)
    llvm::outs() << "session.runtime.counter." << key << "=" << value << "\n";
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

#ifndef ASCEND_RUNTIME_SESSION_RUN_ONLY
std::string parentDirectoryOrCurrent(llvm::StringRef path) {
  llvm::SmallString<256> dir(path);
  llvm::sys::path::remove_filename(dir);
  if (dir.empty())
    return ".";
  return dir.str().str();
}

llvm::Expected<std::string> createTemporaryCaseRunManifestPath() {
  int fd = -1;
  llvm::SmallString<256> path;
  if (auto ec = llvm::sys::fs::createTemporaryFile(
          "runtime-session-case", "json", fd, path))
    return llvm::createStringError(ec,
                                   "cannot create temporary case run manifest");
#ifndef _WIN32
  if (fd >= 0)
    ::close(fd);
#endif
  return path.str().str();
}
#endif

} // namespace

int main(int argc, char **argv) {
  configureSiblingToolPathEnv("AFIR_MIX_TILING_HELPER", argv[0],
                              "mix-tiling-helper");
  llvm::cl::HideUnrelatedOptions(RuntimeSessionCategory);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "task graph runtime planning and execution CLI for artifacts and session graphs\n");

  if (!CompareTensorsPath.empty() || !EmitValidationSummaryPath.empty()) {
    if (CompareTensorsPath.empty() || EmitValidationSummaryPath.empty()) {
      llvm::errs() << "Error: --compare-tensors requires --emit-validation-summary\n";
      return 4;
    }
    if (RunSession || !RunManifestPath.empty()
#ifndef ASCEND_RUNTIME_SESSION_RUN_ONLY
        || !DebugCasePath.empty() || !ArtifactManifestPath.empty() ||
        !EmitRunManifestPath.empty() || !ArtifactRoot.empty() ||
        !KernelFile.empty() || !ShapeArgAssignments.empty() ||
        !InputPathAssignments.empty() || !ExpectedOutputPathAssignments.empty() ||
        !OutputArgs.empty() || !CannMlir.empty() || !NpyDir.empty()
#endif
        || !TestingDriver.empty()
    ) {
      llvm::errs() << "Error: --compare-tensors cannot be combined with runtime planning, compile, or run options\n";
      return 4;
    }

    TensorDiffRequest request;
    request.manifestPath = CompareTensorsPath;
    request.outputSummaryPath = EmitValidationSummaryPath;
    auto resultOr = emitTensorDiffSummaryFromManifest(request);
    if (!resultOr) {
      llvm::errs() << "Error: " << llvm::toString(resultOr.takeError()) << "\n";
      return 4;
    }
    llvm::outs() << "validation.comparisons=" << resultOr->comparisonCount << "\n";
    llvm::outs() << "validation.failed=" << resultOr->failedCount << "\n";
    llvm::outs() << "validation.status=" << (resultOr->passed ? "pass" : "fail") << "\n";
    llvm::outs() << "validation.summary=" << EmitValidationSummaryPath << "\n";
    return resultOr->passed ? 0 : 1;
  }

  std::string caseRunManifestPath;
#ifndef ASCEND_RUNTIME_SESSION_RUN_ONLY
  if (!DebugCasePath.empty()) {
    if (!RunManifestPath.empty() || !ArtifactManifestPath.empty() ||
        !ArtifactRoot.empty() || !KernelFile.empty()) {
      llvm::errs() << "Error: --case cannot be combined with --run-manifest, "
                      "--artifact-manifest, --artifact-root, or --kernel\n";
      return 4;
    }
    if (!ShapeArgAssignments.empty() || !InputPathAssignments.empty() ||
        !OutputArgs.empty() || !ExpectedOutputPathAssignments.empty()) {
      llvm::errs() << "Error: --case carries shape/input/output bindings; do "
                      "not combine it with --shape-arg, --input, --output, or "
                      "--expected-output\n";
      return 4;
    }
    if (EmitRunManifestPath.empty() && !RunSession) {
      llvm::errs() << "Error: --case requires --emit-run-manifest or --run\n";
      return 4;
    }

    if (!EmitRunManifestPath.empty()) {
      caseRunManifestPath = EmitRunManifestPath;
    } else {
      auto tempPathOr = createTemporaryCaseRunManifestPath();
      if (!tempPathOr) {
        llvm::errs() << "Error: " << llvm::toString(tempPathOr.takeError())
                     << "\n";
        return 4;
      }
      caseRunManifestPath = *tempPathOr;
    }

    llvm::SmallString<256> defaultOutputDir(
        parentDirectoryOrCurrent(caseRunManifestPath));
    llvm::sys::path::append(defaultOutputDir, "outputs");

    DebugCasePrepareRequest caseRequest;
    caseRequest.casePath = DebugCasePath;
    caseRequest.outputRunManifestPath = caseRunManifestPath;
    caseRequest.defaultOutputDirectory = defaultOutputDir.str().str();
    if (auto err = emitRunManifestFromDebugCase(caseRequest)) {
      llvm::errs() << "Error: " << llvm::toString(std::move(err)) << "\n";
      return 4;
    }
    llvm::outs() << "run_manifest.path=" << caseRunManifestPath << "\n";
    if (!RunSession)
      return 0;
  }

  if (caseRunManifestPath.empty() &&
      (!ArtifactManifestPath.empty() || !EmitRunManifestPath.empty())) {
    if (ArtifactManifestPath.empty()) {
      llvm::errs() << "Error: --emit-run-manifest requires --artifact-manifest\n";
      return 4;
    }
    if (EmitRunManifestPath.empty()) {
      llvm::errs() << "Error: --artifact-manifest requires --emit-run-manifest\n";
      return 4;
    }
    if (ArtifactRoot.empty()) {
      llvm::errs() << "Error: --artifact-manifest prepare requires --artifact-root\n";
      return 4;
    }
    if (!RunManifestPath.empty()) {
      llvm::errs()
          << "Error: cannot combine --emit-run-manifest with --run-manifest\n";
      return 4;
    }
    if (!KernelFile.empty()) {
      llvm::errs()
          << "Error: cannot combine --artifact-manifest prepare with --kernel\n";
      return 4;
    }
    if (RunSession) {
      llvm::errs() << "Error: --emit-run-manifest is a prepare step; run the "
                      "emitted manifest with --run-manifest --run\n";
      return 4;
    }

    ArtifactManifestPrepareRequest prepareRequest;
    prepareRequest.artifactManifestPath = ArtifactManifestPath;
    prepareRequest.artifactRoot = ArtifactRoot;
    prepareRequest.outputRunManifestPath = EmitRunManifestPath;
    for (const std::string &assignment : ShapeArgAssignments) {
      auto shapeArgOr = parseShapeArgAssignment(assignment);
      if (!shapeArgOr) {
        llvm::errs() << "Error: " << llvm::toString(shapeArgOr.takeError())
                     << "\n";
        return 4;
      }
      prepareRequest.shapeArgs.push_back(std::move(*shapeArgOr));
    }
    if (!collectBindingAssignments("input", InputPathAssignments,
                                   prepareRequest.inputPaths))
      return 4;
    if (!collectBindingAssignments("output", OutputArgs,
                                   prepareRequest.outputPaths))
      return 4;
    if (!collectBindingAssignments("expected-output",
                                   ExpectedOutputPathAssignments,
                                   prepareRequest.expectedOutputPaths))
      return 4;
    if (EnableProfiling)
      prepareRequest.enableProfiling = true;
    if (PrepareAtol.getNumOccurrences() > 0)
      prepareRequest.atol = PrepareAtol;
    if (PrepareRtol.getNumOccurrences() > 0)
      prepareRequest.rtol = PrepareRtol;
    if (auto err = emitRunManifestFromArtifactManifest(prepareRequest)) {
      llvm::errs() << "Error: " << llvm::toString(std::move(err)) << "\n";
      return 4;
    }
    llvm::outs() << "run_manifest.path=" << EmitRunManifestPath << "\n";
    return 0;
  }
#endif

  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
#ifndef ASCEND_RUNTIME_SESSION_RUN_ONLY
  std::optional<KernelArtifact> artifact;
#endif
  std::optional<TaskGraph> graph;
  std::string effectiveRunManifestPath =
      caseRunManifestPath.empty() ? RunManifestPath.getValue()
                                  : caseRunManifestPath;
  if (!effectiveRunManifestPath.empty()) {
    auto manifestGraphOr =
        prepareRuntimeSessionGraphFromManifest(effectiveRunManifestPath);
    if (!manifestGraphOr) {
      llvm::errs() << "Error: " << llvm::toString(manifestGraphOr.takeError())
                   << "\n";
      return 4;
    }
    backendKind = manifestGraphOr->first;
    graph = std::move(manifestGraphOr->second);
  } else {
#ifdef ASCEND_RUNTIME_SESSION_RUN_ONLY
    llvm::errs()
        << "Error: run-only runtime-session build requires --run-manifest\n";
    return 4;
#else
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
      request.socVersion = SocVersion;
      if (!CannMlir.empty())
        request.cannMlirPath = CannMlir;
      if (!NpyDir.empty())
        request.npyDir = NpyDir;
      if (OutputArgs.size() > 1) {
        llvm::errs()
            << "Error: compile path accepts at most one --output directory\n";
        return 4;
      }
      request.outputDir = OutputArgs.empty()
                              ? "./build/runtime-session-artifact"
                              : OutputArgs.front();
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
#endif
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
