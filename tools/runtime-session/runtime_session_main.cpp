#include "Runtime/ArtifactCompiler.h"
#include "Runtime/ExecutionBackend.h"
#include "Runtime/ExecutionSession.h"
#include "Runtime/RunManifest.h"
#include "Runtime/TaskGraph.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <map>
#include <optional>
#include <string>

namespace {

using namespace mlir::runtime;

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
    llvm::cl::desc("Traverse ExecutionSession::run and return the current task I/O binding limitation"),
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

llvm::Expected<KernelKind> parseManifestKernelKind(llvm::StringRef name) {
  if (name.empty())
    return KernelKind::Mix;
  return parseKernelKind(name);
}

std::string defaultKernelName(llvm::StringRef kernelFile,
                              llvm::StringRef artifactRoot) {
  if (!kernelFile.empty())
    return llvm::sys::path::stem(kernelFile).str();
  if (!artifactRoot.empty())
    return llvm::sys::path::filename(artifactRoot).str();
  return "kernel";
}

std::map<std::string, std::string> readManifest(const std::string &path) {
  std::map<std::string, std::string> out;
  auto bufferOr = llvm::MemoryBuffer::getFile(path, false);
  if (!bufferOr)
    return out;

  llvm::SmallVector<llvm::StringRef> lines;
  (*bufferOr)->getBuffer().split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (line.empty() || line.starts_with("#"))
      continue;
    size_t split = line.find('=');
    if (split == llvm::StringRef::npos)
      continue;
    out.emplace(line.substr(0, split).str(), line.substr(split + 1).str());
  }
  return out;
}

llvm::Expected<std::string> requireManifestValue(
    const std::map<std::string, std::string> &manifest, llvm::StringRef key) {
  auto it = manifest.find(key.str());
  if (it == manifest.end() || it->second.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "manifest is missing required field: %s",
                                   key.str().c_str());
  }
  return it->second;
}

std::string resolveArtifactPath(llvm::StringRef artifactRoot,
                                llvm::StringRef maybeRelativePath) {
  if (maybeRelativePath.empty())
    return "";
  if (llvm::sys::path::is_absolute(maybeRelativePath))
    return maybeRelativePath.str();

  llvm::SmallString<256> resolved(artifactRoot);
  llvm::sys::path::append(resolved, maybeRelativePath);
  return resolved.str().str();
}

llvm::Expected<std::string> locateManifestPath(llvm::StringRef artifactRoot) {
  llvm::SmallVector<llvm::SmallString<256>, 3> candidates;
  candidates.emplace_back(artifactRoot);
  llvm::sys::path::append(candidates.back(), "out", "manifest.txt");
  candidates.emplace_back(artifactRoot);
  llvm::sys::path::append(candidates.back(), "mix-artifact.txt");
  candidates.emplace_back(artifactRoot);
  llvm::sys::path::append(candidates.back(), "out", "mix-artifact.txt");

  for (const llvm::SmallString<256> &candidate : candidates) {
    if (llvm::sys::fs::exists(candidate))
      return candidate.str().str();
  }

  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "artifact root does not contain a supported manifest (expected out/manifest.txt, mix-artifact.txt, or out/mix-artifact.txt)");
}

llvm::Expected<KernelArtifact> loadArtifactFromRoot() {
  llvm::SmallString<256> artifactRoot(ArtifactRoot);
  llvm::sys::fs::make_absolute(artifactRoot);

  llvm::sys::fs::file_status status;
  if (auto ec = llvm::sys::fs::status(artifactRoot, status))
    return llvm::createStringError(ec, "cannot access artifact root: %s",
                                   artifactRoot.c_str());
  if (!llvm::sys::fs::is_directory(status)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "artifact root is not a directory: %s",
                                   artifactRoot.c_str());
  }

  auto manifestPathOr = locateManifestPath(artifactRoot.str());
  if (!manifestPathOr)
    return manifestPathOr.takeError();

  const std::string manifestPath = *manifestPathOr;
  const auto manifest = readManifest(manifestPath);
  if (manifest.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "manifest is empty or unreadable: %s",
                                   manifestPath.c_str());
  }

  auto kernelNameOr = requireManifestValue(manifest, "kernel_name");
  if (!kernelNameOr)
    return kernelNameOr.takeError();
  auto socVersionOr = requireManifestValue(manifest, "soc_version");
  if (!socVersionOr)
    return socVersionOr.takeError();

  KernelArtifact artifact;
  artifact.kernelName = *kernelNameOr;
  auto kernelKindIt = manifest.find("kernel_kind");
  auto parsedKernelKindOr = parseManifestKernelKind(
      kernelKindIt != manifest.end() ? kernelKindIt->second : "");
  if (!parsedKernelKindOr)
    return parsedKernelKindOr.takeError();
  artifact.kernelKind = *parsedKernelKindOr;
  artifact.mixResourceType = MixResourceType::Unknown;
  artifact.socVersion = *socVersionOr;
  artifact.artifactRoot = artifactRoot.str().str();

  auto manifestPathIt = manifest.find("manifest_path");
  if (manifestPathIt != manifest.end() && !manifestPathIt->second.empty()) {
    artifact.manifestPath =
        resolveArtifactPath(artifact.artifactRoot, manifestPathIt->second);
    if (!llvm::sys::fs::exists(artifact.manifestPath)) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "manifest_path from manifest does not exist: %s",
          artifact.manifestPath.c_str());
    }
  } else {
    artifact.manifestPath = manifestPath;
  }

  auto kernelSoIt = manifest.find("kernel_so_path");
  if (kernelSoIt != manifest.end() && !kernelSoIt->second.empty())
    artifact.packedSharedObjectPath =
        resolveArtifactPath(artifact.artifactRoot, kernelSoIt->second);

  auto deviceObjectIt = manifest.find("device_object_path");
  auto deviceBinaryIt = manifest.find("device_binary_path");
  if (deviceBinaryIt != manifest.end() && !deviceBinaryIt->second.empty()) {
    artifact.deviceBinaryPath =
        resolveArtifactPath(artifact.artifactRoot, deviceBinaryIt->second);
  } else if (deviceObjectIt != manifest.end() && !deviceObjectIt->second.empty()) {
    artifact.deviceBinaryPath =
        resolveArtifactPath(artifact.artifactRoot, deviceObjectIt->second);
  } else if (!artifact.packedSharedObjectPath.empty()) {
    artifact.deviceBinaryPath = artifact.packedSharedObjectPath;
  }

  return artifact;
}

llvm::Expected<KernelArtifact> prepareArtifact() {
  const bool hasArtifactRoot = !ArtifactRoot.empty();
  const bool hasKernelFile = !KernelFile.empty();
  if (hasArtifactRoot == hasKernelFile) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "provide exactly one of --artifact-root or --kernel");
  }

  if (hasArtifactRoot)
    return loadArtifactFromRoot();

  auto kernelKindOr = parseKernelKind(KernelKindName);
  if (!kernelKindOr)
    return kernelKindOr.takeError();
  const KernelKind kernelKind = *kernelKindOr;

  std::string effectiveKernelName = KernelName;
  if (effectiveKernelName.empty())
    effectiveKernelName = defaultKernelName(KernelFile, "");

  ArtifactCompileRequest request;
  request.kernelSource = KernelFile;
  request.kernelName = effectiveKernelName;
  request.kernelKind = kernelKind;
  request.socVersion = SocVersion;
  request.outputDir = OutputDir;
  if (!CannMlir.empty())
    request.cannMlirPath = CannMlir;
  if (!NpyDir.empty())
    request.npyDir = NpyDir;

  ArtifactCompiler compiler;
  return compiler.compile(request);
}

llvm::Expected<TaskGraph> buildGraph(const KernelArtifact &artifact,
                                     const ExecutionInvocation *invocation = nullptr) {
  TaskGraph graph;
  RuntimeTask task;
  task.taskId = TaskId;
  task.artifact = artifact;
  if (invocation)
    task.invocation = *invocation;
  if (auto err = graph.addTask(task))
    return std::move(err);
  return graph;
}

llvm::Expected<std::pair<ExecutionBackendKind, TaskGraph>>
prepareManifestGraph() {
  auto runSpecOr = loadRunManifest(RunManifestPath);
  if (!runSpecOr)
    return runSpecOr.takeError();

  ArtifactRoot = runSpecOr->artifactRoot;
  auto artifactOr = loadArtifactFromRoot();
  if (!artifactOr)
    return artifactOr.takeError();

  TaskGraph graph;
  RuntimeTask task;
  task.taskId = runSpecOr->taskId;
  task.artifact = *artifactOr;
  task.invocation = runSpecOr->invocation;
  if (auto err = graph.addTask(task))
    return std::move(err);
  return std::make_pair(runSpecOr->backendKind, std::move(graph));
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

} // namespace

int main(int argc, char **argv) {
  llvm::cl::HideUnrelatedOptions(RuntimeSessionCategory);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "task graph runtime planning CLI for artifacts and session plans, with a stubbed run path\n");

  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  std::optional<KernelArtifact> artifact;
  std::optional<TaskGraph> graph;
  if (!RunManifestPath.empty()) {
    auto manifestGraphOr = prepareManifestGraph();
    if (!manifestGraphOr) {
      llvm::errs() << "Error: " << llvm::toString(manifestGraphOr.takeError())
                   << "\n";
      return 4;
    }
    backendKind = manifestGraphOr->first;
    graph = std::move(manifestGraphOr->second);
  } else {
    auto artifactOr = prepareArtifact();
    if (!artifactOr) {
      llvm::errs() << "Error: " << llvm::toString(artifactOr.takeError()) << "\n";
      return 4;
    }
    artifact = *artifactOr;
    printArtifactSummary(*artifact);
    auto graphOr = buildGraph(*artifact);
    if (!graphOr) {
      llvm::errs() << "Error: " << llvm::toString(graphOr.takeError()) << "\n";
      return 4;
    }
    graph = std::move(*graphOr);
  }

  ExecutionSession session(backendKind);
  auto planOr = session.plan(*graph);
  if (!planOr) {
    llvm::errs() << "Error: " << llvm::toString(planOr.takeError()) << "\n";
    return 2;
  }
  printPlan(*planOr);

  if (!RunSession)
    return 0;

  ExecutionSession runSession(backendKind);
  auto traceOr = runSession.run(*graph);
  if (!traceOr) {
    llvm::errs() << "Error: " << llvm::toString(traceOr.takeError()) << "\n";
    return 2;
  }
  if (backendKind == ExecutionBackendKind::Simulation) {
    llvm::outs().flush();
    llvm::errs().flush();
    _Exit(0);
  }
  return 0;
}
