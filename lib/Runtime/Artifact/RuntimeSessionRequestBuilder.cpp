#include "Runtime/Artifact/RuntimeSessionRequestBuilder.h"

#include "Runtime/Artifact/RunManifest.h"
#include "Runtime/MixAbi.h"
#include "Runtime/MixCompileMetadata.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <map>
#include <optional>
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

llvm::Expected<DType> parseDType(llvm::StringRef value) {
  if (value == "f16")
    return DType::F16;
  if (value == "bf16")
    return DType::BF16;
  if (value == "f32")
    return DType::F32;
  if (value == "int8")
    return DType::INT8;
  if (value == "int32")
    return DType::INT32;
  if (value == "int64")
    return DType::INT64;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported tensor dtype: %s",
                                 value.str().c_str());
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

llvm::Expected<uint32_t> readBlockDimFromLaunchInfo(llvm::StringRef path) {
  auto bufferOr = llvm::MemoryBuffer::getFile(path, false);
  if (!bufferOr)
    return llvm::createStringError(bufferOr.getError(),
                                   "cannot read launch info file: %s",
                                   path.str().c_str());

  llvm::SmallVector<llvm::StringRef> lines;
  (*bufferOr)->getBuffer().split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (!line.starts_with("block_dim="))
      continue;
    uint64_t value = 0;
    if (line.drop_front(strlen("block_dim=")).getAsInteger(10, value))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "invalid block_dim in launch info: %s",
                                     path.str().c_str());
    return static_cast<uint32_t>(value);
  }

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "missing block_dim in launch info: %s",
                                 path.str().c_str());
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

llvm::Error validateMixMetadataPath(llvm::StringRef metadataPath) {
  if (!llvm::sys::fs::exists(metadataPath)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "metadata_path from manifest does not exist: %s",
                                   metadataPath.str().c_str());
  }

  auto bufferOr = llvm::MemoryBuffer::getFile(metadataPath, false);
  if (!bufferOr) {
    return llvm::createStringError(bufferOr.getError(),
                                   "cannot read mix metadata file: %s",
                                   metadataPath.str().c_str());
  }

  auto metadataOr = parseMixCompileMetadataJson((*bufferOr)->getBuffer());
  if (!metadataOr) {
    const std::string diagnostic = llvm::toString(metadataOr.takeError());
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "invalid mix metadata file %s: %s", metadataPath.str().c_str(),
        diagnostic.c_str());
  }
  return llvm::Error::success();
}

llvm::Expected<MixCompileMetadata>
loadValidatedMixMetadata(llvm::StringRef metadataPath) {
  auto bufferOr = llvm::MemoryBuffer::getFile(metadataPath, false);
  if (!bufferOr) {
    return llvm::createStringError(bufferOr.getError(),
                                   "cannot read mix metadata file: %s",
                                   metadataPath.str().c_str());
  }
  return parseMixCompileMetadataJson((*bufferOr)->getBuffer());
}

MixAbiTensorDesc toMixAbiTensorDesc(const MixCompileMetadataTensorDesc &tensor) {
  MixAbiTensorDesc out;
  out.name = tensor.name;
  out.runtimeFile = tensor.runtimeFile;
  out.goldenFile = tensor.goldenFile;
  out.shape = tensor.shape;
  auto dtypeOr = parseDType(tensor.dtype);
  if (dtypeOr)
    out.dtype = *dtypeOr;
  else
    llvm::consumeError(dtypeOr.takeError());
  return out;
}

llvm::Expected<MixAbiMetadata>
buildMixAbiFromMetadata(const MixCompileMetadata &metadata,
                        llvm::StringRef artifactRoot) {
  MixAbiMetadata abi;
  abi.logicalKernelName = metadata.kernelName;
  abi.runtimeKernelName = metadata.runtimeKernelName;
  abi.workspaceMode = metadata.abi.workspaceMode;
  abi.workspaceBytes = static_cast<size_t>(metadata.abi.workspaceBytes);
  abi.tilingMode = metadata.abi.tilingMode;
  abi.tilingSource = metadata.abi.tilingSource;
  if (metadata.abi.hasWorkspaceArgIndex)
    abi.workspaceArgIndex =
        static_cast<size_t>(metadata.abi.workspaceArgIndex);
  if (metadata.abi.hasTilingArgIndex)
    abi.tilingArgIndex = static_cast<size_t>(metadata.abi.tilingArgIndex);
  abi.launcherSymbol = metadata.launcherSymbol;
  abi.aicEntry = metadata.entries.aic;
  abi.aivEntry = metadata.entries.aiv;

  abi.inputs.reserve(metadata.abi.inputs.size());
  for (const auto &tensor : metadata.abi.inputs) {
    auto parsedDTypeOr = parseDType(tensor.dtype);
    if (!parsedDTypeOr)
      return parsedDTypeOr.takeError();
    MixAbiTensorDesc desc = toMixAbiTensorDesc(tensor);
    desc.dtype = *parsedDTypeOr;
    abi.inputs.push_back(std::move(desc));
  }
  abi.outputs.reserve(metadata.abi.outputs.size());
  for (const auto &tensor : metadata.abi.outputs) {
    auto parsedDTypeOr = parseDType(tensor.dtype);
    if (!parsedDTypeOr)
      return parsedDTypeOr.takeError();
    MixAbiTensorDesc desc = toMixAbiTensorDesc(tensor);
    desc.dtype = *parsedDTypeOr;
    abi.outputs.push_back(std::move(desc));
  }

  llvm::StringRef launchInfoPath = metadata.artifacts.launchInfoFilePath;
  if (!launchInfoPath.empty()) {
    const std::string resolvedLaunchInfo =
        resolveArtifactPath(artifactRoot, launchInfoPath);
    auto blockDimOr = readBlockDimFromLaunchInfo(resolvedLaunchInfo);
    if (!blockDimOr)
      return blockDimOr.takeError();
    abi.blockDim = *blockDimOr;
  }

  return abi;
}

llvm::Expected<MixAbiMetadata>
loadMixAbiForArtifact(const KernelArtifact &artifact) {
  const auto manifest = readManifest(artifact.manifestPath);
  if (manifest.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "manifest is empty or unreadable: %s",
                                   artifact.manifestPath.c_str());
  }

  if (!artifact.metadataPath.empty()) {
    auto metadataOr = loadValidatedMixMetadata(artifact.metadataPath);
    if (!metadataOr)
      return metadataOr.takeError();
    return buildMixAbiFromMetadata(*metadataOr, artifact.artifactRoot);
  }

  return parseMixAbiManifest(manifest);
}

void enrichBindingFromAbi(TensorBinding &binding,
                          llvm::ArrayRef<MixAbiTensorDesc> tensors) {
  for (const MixAbiTensorDesc &tensor : tensors) {
    if (tensor.name != binding.name)
      continue;
    if (!binding.shape)
      binding.shape = tensor.shape;
    if (!binding.dtype)
      binding.dtype = tensor.dtype;
    break;
  }
}

void applyMixArtifactInvocationDefaults(const KernelArtifact &artifact,
                                        ExecutionInvocation &invocation) {
  if (artifact.kernelKind != KernelKind::Mix)
    return;

  auto abiOr = loadMixAbiForArtifact(artifact);
  if (!abiOr) {
    llvm::consumeError(abiOr.takeError());
    return;
  }
  const MixAbiMetadata &abi = *abiOr;

  if (invocation.blockDim == 1 && abi.blockDim != 0)
    invocation.blockDim = static_cast<int>(abi.blockDim);
  if (invocation.workspaceSize == 8192 && abi.workspaceBytes != 0)
    invocation.workspaceSize = abi.workspaceBytes;

  for (TensorBinding &binding : invocation.inputs)
    enrichBindingFromAbi(binding, abi.inputs);
  for (TensorBinding &binding : invocation.outputs)
    enrichBindingFromAbi(binding, abi.outputs);
  for (TensorBinding &binding : invocation.expectedOutputs)
    enrichBindingFromAbi(binding, abi.outputs);

  if (!invocation.tiling.has_value()) {
    if (abi.tilingMode == "generated_file" && !abi.tilingSource.empty()) {
      TilingBinding tiling;
      tiling.binaryPath = resolveArtifactPath(artifact.artifactRoot,
                                             abi.tilingSource);
      invocation.tiling = std::move(tiling);
    }
    return;
  }

  if (invocation.tiling->binaryPath.empty() && invocation.tiling->schemaPath.empty() &&
      abi.tilingMode == "generated_file" && !abi.tilingSource.empty()) {
    invocation.tiling->binaryPath =
        resolveArtifactPath(artifact.artifactRoot, abi.tilingSource);
  }
}

llvm::Error
validateMixArtifactInvocationBindings(const KernelArtifact &artifact,
                                      const ExecutionInvocation &invocation,
                                      llvm::StringRef taskId) {
  if (artifact.kernelKind != KernelKind::Mix)
    return llvm::Error::success();

  auto abiOr = loadMixAbiForArtifact(artifact);
  if (!abiOr)
    return abiOr.takeError();
  const MixAbiMetadata &abi = *abiOr;

  if (invocation.inputs.size() != abi.inputs.size()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "mix artifact ABI input count mismatch for task %s: run manifest has "
        "%zu input binding(s), artifact ABI has %zu",
        taskId.str().c_str(), invocation.inputs.size(), abi.inputs.size());
  }

  if (invocation.outputs.size() != abi.outputs.size()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "mix artifact ABI output count mismatch for task %s: run manifest has "
        "%zu output binding(s), artifact ABI has %zu",
        taskId.str().c_str(), invocation.outputs.size(), abi.outputs.size());
  }

  return llvm::Error::success();
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
    artifact.sharedLibraryPath =
        resolveArtifactPath(artifact.artifactRoot, kernelSoIt->second);

  auto deviceObjectIt = manifest.find("device_object_path");
  auto deviceBinaryIt = manifest.find("device_binary_path");
  if (deviceBinaryIt != manifest.end() && !deviceBinaryIt->second.empty()) {
    artifact.deviceBinaryPath =
        resolveArtifactPath(artifact.artifactRoot, deviceBinaryIt->second);
  } else if (deviceObjectIt != manifest.end() && !deviceObjectIt->second.empty()) {
    artifact.deviceBinaryPath =
        resolveArtifactPath(artifact.artifactRoot, deviceObjectIt->second);
  } else if (!artifact.sharedLibraryPath.empty()) {
    artifact.deviceBinaryPath = artifact.sharedLibraryPath;
  }

  auto metadataPathIt = manifest.find("metadata_path");
  if (metadataPathIt != manifest.end() && !metadataPathIt->second.empty()) {
    artifact.metadataPath =
        resolveArtifactPath(artifact.artifactRoot, metadataPathIt->second);
    if (artifact.kernelKind == KernelKind::Mix) {
      if (auto err = validateMixMetadataPath(artifact.metadataPath))
        return std::move(err);
      auto metadataOr = loadValidatedMixMetadata(artifact.metadataPath);
      if (!metadataOr)
        return metadataOr.takeError();
      artifact.sharedLibrarySymbol = metadataOr->launcherSymbol;
    } else if (!llvm::sys::fs::exists(artifact.metadataPath)) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "metadata_path from manifest does not exist: %s",
          artifact.metadataPath.c_str());
    }
  }

  if (artifact.kernelKind == KernelKind::Mix &&
      artifact.sharedLibrarySymbol.empty()) {
    artifact.sharedLibrarySymbol = "aclrtlaunch_" + artifact.kernelName;
  }

  return artifact;
}

} // namespace

llvm::Expected<KernelArtifact>
loadRuntimeSessionArtifactFromRoot(llvm::StringRef artifactRoot) {
  return loadArtifactFromRoot(artifactRoot);
}

llvm::Expected<TaskGraph>
buildRuntimeSessionSingleTaskGraph(const KernelArtifact &artifact,
                                   llvm::StringRef taskId) {
  TaskGraph graph;
  RuntimeTask task;
  task.taskId = taskId.str();
  task.artifact = artifact;
  applyMixArtifactInvocationDefaults(task.artifact, task.invocation);
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
    if (auto err = validateMixArtifactInvocationBindings(
            task.artifact, task.invocation, task.taskId))
      return std::move(err);
    applyMixArtifactInvocationDefaults(task.artifact, task.invocation);
    if (auto err = graph.addTask(task))
      return std::move(err);
  }
  return std::make_pair(runSpecOr->backendKind, std::move(graph));
}

} // namespace mlir::runtime
