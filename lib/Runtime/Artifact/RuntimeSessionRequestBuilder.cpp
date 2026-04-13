#include "Runtime/Artifact/RuntimeSessionRequestBuilder.h"

#include "Runtime/Artifact/RunManifest.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <map>
#include <string>
#include <utility>

namespace mlir::runtime {

namespace {

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
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "manifest is missing required field: kernel_kind");
  return parseKernelKind(name);
}

llvm::Expected<MixResourceType>
parseManifestMixResourceType(llvm::StringRef name, KernelKind kind) {
  if (name.empty()) {
    if (kind == KernelKind::Mix)
      return MixResourceType::Mix1C1V;
    return MixResourceType::Unknown;
  }

  const std::string normalized = name.trim().lower();
  if (normalized == "unknown")
    return MixResourceType::Unknown;
  if (normalized == "aiv_only" || normalized == "aivonly")
    return MixResourceType::AIVOnly;
  if (normalized == "aic_only" || normalized == "aiconly")
    return MixResourceType::AICOnly;
  if (normalized == "mix_1c1v" || normalized == "mix1c1v")
    return MixResourceType::Mix1C1V;
  if (normalized == "mix_1c2v" || normalized == "mix1c2v")
    return MixResourceType::Mix1C2V;

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported mix resource type in manifest: %s",
                                 name.str().c_str());
}

std::string defaultKernelName(llvm::StringRef kernelSource) {
  if (!kernelSource.empty())
    return llvm::sys::path::stem(kernelSource).str();
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

llvm::Expected<KernelArtifact>
loadArtifactFromRoot(llvm::StringRef artifactRootInput) {
  llvm::SmallString<256> artifactRoot(artifactRootInput);
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
  auto kernelKindOr = requireManifestValue(manifest, "kernel_kind");
  if (!kernelKindOr)
    return kernelKindOr.takeError();

  KernelArtifact artifact;
  artifact.kernelName = *kernelNameOr;
  auto parsedKernelKindOr = parseManifestKernelKind(*kernelKindOr);
  if (!parsedKernelKindOr)
    return parsedKernelKindOr.takeError();
  artifact.kernelKind = *parsedKernelKindOr;
  auto mixResourceTypeIt = manifest.find("mix_resource_type");
  auto parsedMixResourceTypeOr = parseManifestMixResourceType(
      mixResourceTypeIt != manifest.end() ? mixResourceTypeIt->second : "",
      artifact.kernelKind);
  if (!parsedMixResourceTypeOr)
    return parsedMixResourceTypeOr.takeError();
  artifact.mixResourceType = *parsedMixResourceTypeOr;
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

} // namespace

llvm::Expected<KernelArtifact>
loadRuntimeSessionArtifactFromRoot(llvm::StringRef artifactRoot) {
  return loadArtifactFromRoot(artifactRoot);
}

llvm::Expected<KernelArtifact>
prepareRuntimeSessionArtifact(const RuntimeSessionArtifactRequest &request) {
  const bool hasArtifactRoot = !request.artifactRoot.empty();
  const bool hasKernelSource = !request.kernelSource.empty();
  if (hasArtifactRoot == hasKernelSource) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "provide exactly one of --artifact-root or --kernel");
  }

  if (hasArtifactRoot)
    return loadArtifactFromRoot(request.artifactRoot);

  KernelKind kernelKind = KernelKind::Vec;
  switch (request.kernelKind) {
  case KernelKind::Vec:
  case KernelKind::Cube:
  case KernelKind::Mix:
    kernelKind = request.kernelKind;
    break;
  default:
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "unsupported kernel kind: %d",
        static_cast<int>(request.kernelKind));
  }

  std::string effectiveKernelName = request.kernelName;
  if (effectiveKernelName.empty())
    effectiveKernelName = defaultKernelName(request.kernelSource);

  ArtifactCompileRequest compileRequest;
  compileRequest.kernelSource = request.kernelSource;
  compileRequest.kernelName = effectiveKernelName;
  compileRequest.kernelKind = kernelKind;
  compileRequest.socVersion = request.socVersion;
  compileRequest.outputDir = request.outputDir;
  if (request.cannMlirPath)
    compileRequest.cannMlirPath = request.cannMlirPath;
  if (request.npyDir)
    compileRequest.npyDir = request.npyDir;

  ArtifactCompiler compiler;
  return compiler.compile(compileRequest);
}

llvm::Expected<TaskGraph>
buildRuntimeSessionSingleTaskGraph(const KernelArtifact &artifact,
                                   llvm::StringRef taskId) {
  TaskGraph graph;
  RuntimeTask task;
  task.taskId = taskId.str();
  task.artifact = artifact;
  if (auto err = graph.addTask(task))
    return std::move(err);
  return graph;
}

llvm::Expected<std::pair<ExecutionBackendKind, TaskGraph>>
prepareRuntimeSessionGraphFromManifest(llvm::StringRef runManifestPath) {
  auto runSpecOr = loadRunManifest(runManifestPath.str());
  if (!runSpecOr)
    return runSpecOr.takeError();

  TaskGraph graph;
  for (const RunTaskSpec &taskSpec : runSpecOr->tasks) {
    auto artifactOr = loadArtifactFromRoot(taskSpec.artifactRoot);
    if (!artifactOr)
      return artifactOr.takeError();

    RuntimeTask task;
    task.taskId = taskSpec.taskId;
    task.artifact = *artifactOr;
    task.dependencies = taskSpec.dependencies;
    task.invocation = taskSpec.invocation;
    if (auto err = graph.addTask(task))
      return std::move(err);
  }
  return std::make_pair(runSpecOr->backendKind, std::move(graph));
}

} // namespace mlir::runtime
