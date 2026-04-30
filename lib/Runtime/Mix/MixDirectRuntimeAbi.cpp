#include "MixDirectCompileInternal.h"

#include "Runtime/Mix/MixAbiExtractor.h"
#include "Runtime/Mix/MixCompileMetadata.h"
#include "Runtime/NpyIO.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

#include <utility>

namespace mlir::runtime {
namespace {

static llvm::Expected<NDArray> loadNpyTensor(llvm::StringRef path) {
  return LoadNpy(path.str());
}

static bool hasDynamicShape(llvm::ArrayRef<int64_t> shape) {
  for (int64_t dim : shape)
    if (dim < 0)
      return true;
  return false;
}

static llvm::StringRef getDTypeName(DType dtype) {
  switch (dtype) {
  case DType::F16:
    return "f16";
  case DType::BF16:
    return "bf16";
  case DType::F32:
    return "f32";
  case DType::INT8:
    return "int8";
  case DType::INT32:
    return "int32";
  case DType::INT64:
    return "int64";
  default:
    return "unknown";
  }
}

static std::string buildOrdinalTensorNpyName(bool isOutput, size_t index) {
  return (llvm::Twine(isOutput ? "output" : "input") + llvm::Twine(index) +
          ".npy")
      .str();
}

static llvm::Expected<std::string>
resolveTensorNpyPath(llvm::StringRef npyDir, const MixAbiTensorDesc &tensor,
                     size_t index, bool isOutput) {
  const std::string namedPath = joinPath(npyDir, tensor.name + ".npy");
  if (llvm::sys::fs::exists(namedPath))
    return namedPath;
  const std::string ordinalPath =
      joinPath(npyDir, buildOrdinalTensorNpyName(isOutput, index));
  if (llvm::sys::fs::exists(ordinalPath))
    return ordinalPath;
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "RuntimeMix cannot resolve concrete %s tensor shape for ABI tensor '%s': "
      "expected either %s or %s",
      isOutput ? "output" : "input", tensor.name.c_str(), namedPath.c_str(),
      ordinalPath.c_str());
}

static llvm::Error reconcileTensorWithNpy(MixAbiTensorDesc &tensor,
                                          llvm::StringRef npyPath) {
  auto arrayOr = loadNpyTensor(npyPath);
  if (!arrayOr)
    return arrayOr.takeError();
  if (arrayOr->dtype != tensor.dtype)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix ABI dtype mismatch for tensor '%s': MLIR expects %s but %s "
        "contains %s",
        tensor.name.c_str(), getDTypeName(tensor.dtype).str().c_str(),
        npyPath.str().c_str(), getDTypeName(arrayOr->dtype).str().c_str());
  if (!tensor.shape.empty() && tensor.shape.size() != arrayOr->shape.size())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix ABI rank mismatch for tensor '%s': MLIR rank=%zu but %s "
        "rank=%zu",
        tensor.name.c_str(), tensor.shape.size(), npyPath.str().c_str(),
        arrayOr->shape.size());
  if (tensor.shape.empty()) {
    tensor.shape = arrayOr->shape;
    return llvm::Error::success();
  }
  for (size_t i = 0; i < arrayOr->shape.size(); ++i) {
    if (tensor.shape[i] >= 0 && tensor.shape[i] != arrayOr->shape[i])
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "RuntimeMix ABI shape mismatch for tensor '%s' dim %zu: MLIR expects "
          "%lld but %s provides %lld",
          tensor.name.c_str(), i, static_cast<long long>(tensor.shape[i]),
          npyPath.str().c_str(), static_cast<long long>(arrayOr->shape[i]));
    tensor.shape[i] = arrayOr->shape[i];
  }
  return llvm::Error::success();
}

static llvm::Error resolveDynamicShapesFromNpyDir(MixAbiMetadata &abi,
                                                  llvm::StringRef npyDir) {
  for (size_t i = 0; i < abi.inputs.size(); ++i) {
    if (!hasDynamicShape(abi.inputs[i].shape))
      continue;
    auto npyPathOr = resolveTensorNpyPath(npyDir, abi.inputs[i], i, false);
    if (!npyPathOr)
      return npyPathOr.takeError();
    if (auto err = reconcileTensorWithNpy(abi.inputs[i], *npyPathOr))
      return err;
  }
  for (size_t i = 0; i < abi.outputs.size(); ++i) {
    if (!hasDynamicShape(abi.outputs[i].shape))
      continue;
    auto npyPathOr = resolveTensorNpyPath(npyDir, abi.outputs[i], i, true);
    if (!npyPathOr)
      return npyPathOr.takeError();
    if (auto err = reconcileTensorWithNpy(abi.outputs[i], *npyPathOr))
      return err;
  }
  return llvm::Error::success();
}

static llvm::Expected<size_t>
findFirstDynamicTensorIndex(llvm::ArrayRef<MixAbiTensorDesc> tensors) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (hasDynamicShape(tensors[i].shape))
      return i;
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "no dynamic tensor found");
}

static MixCompileMetadataTensorDesc
makeMetadataTensorDesc(const MixAbiTensorDesc &tensor) {
  MixCompileMetadataTensorDesc out;
  out.name = tensor.name;
  out.dtype = getDTypeName(tensor.dtype).str();
  out.shape = tensor.shape;
  out.runtimeFile = tensor.runtimeFile;
  out.goldenFile = tensor.goldenFile;
  return out;
}

} // namespace

llvm::Expected<MixAbiMetadata>
loadMixDirectRuntimeAbi(llvm::StringRef cannMlirPath, llvm::StringRef npyDir,
                        llvm::StringRef runtimeKernelName) {
  if (cannMlirPath.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix direct backend requires --cann-mlir to derive ABI and "
        "canonical IO metadata for kernel '%s'",
        runtimeKernelName.str().c_str());

  auto abiOr = extractMixAbiFromCannMlir(cannMlirPath);
  if (!abiOr)
    return abiOr.takeError();
  MixAbiMetadata abi = std::move(*abiOr);

  if (!npyDir.empty()) {
    if (auto err = resolveDynamicShapesFromNpyDir(abi, npyDir))
      return std::move(err);
  }

  if (auto inputIndexOr = findFirstDynamicTensorIndex(abi.inputs))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix MLIR ABI extraction from %s produced unresolved dynamic "
        "input shape for tensor '%s'; pass --npy-dir with concrete IO data "
        "to resolve dynamic extents",
        cannMlirPath.str().c_str(), abi.inputs[*inputIndexOr].name.c_str());
  else
    llvm::consumeError(inputIndexOr.takeError());

  if (auto outputIndexOr = findFirstDynamicTensorIndex(abi.outputs))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix MLIR ABI extraction from %s produced unresolved dynamic "
        "output shape for tensor '%s'; pass --npy-dir with concrete IO data "
        "to resolve dynamic extents",
        cannMlirPath.str().c_str(), abi.outputs[*outputIndexOr].name.c_str());
  else
    llvm::consumeError(outputIndexOr.takeError());

  if (abi.logicalKernelName.empty())
    abi.logicalKernelName = runtimeKernelName.str();
  abi.runtimeKernelName = runtimeKernelName.str();
  if (abi.launcherSymbol.empty())
    abi.launcherSymbol = "aclrtlaunch_" + runtimeKernelName.str();
  if (abi.aicEntry.empty())
    abi.aicEntry = runtimeKernelName.str() + "_0_mix_aic";
  if (abi.aivEntry.empty())
    abi.aivEntry = runtimeKernelName.str() + "_0_mix_aiv";
  if (abi.workspaceBytes == 0)
    abi.workspaceBytes = 16777216ULL;
  abi.workspaceMode = "fixed";
  abi.tilingMode = "generated_file";
  abi.tilingSource = "out/tiling.bin";
  return abi;
}

llvm::Expected<std::string>
writeMixDirectCompileMetadataFile(llvm::StringRef metadataPath,
                                  llvm::StringRef runtimeKernelName,
                                  llvm::StringRef socVersion,
                                  llvm::StringRef mixKernelType,
                                  llvm::StringRef generatedSourcePath,
                                  llvm::ArrayRef<std::string> aicDefinitions,
                                  llvm::ArrayRef<std::string> aivDefinitions,
                                  llvm::StringRef deviceObjectPath,
                                  llvm::StringRef packedSharedObjectPath,
                                  const MixDirectTilingOutputs &tiling,
                                  const MixAbiMetadata &abi) {
  MixCompileMetadata metadata;
  metadata.schemaVersion = 1;
  metadata.kernelKind = "mix";
  metadata.kernelName = runtimeKernelName.str();
  metadata.runtimeKernelName = runtimeKernelName.str();
  metadata.socVersion = socVersion.str();
  metadata.mixKernelType = mixKernelType.str();
  metadata.launcherSymbol = abi.launcherSymbol;
  metadata.entries.aic = abi.aicEntry;
  metadata.entries.aiv = abi.aivEntry;
  metadata.generated.sourcePath = generatedSourcePath.str();
  metadata.deviceCompile.aicArch = "dav-c220-cube";
  metadata.deviceCompile.aivArch = "dav-c220-vec";
  metadata.deviceCompile.aicDefinitions.assign(aicDefinitions.begin(),
                                               aicDefinitions.end());
  metadata.deviceCompile.aivDefinitions.assign(aivDefinitions.begin(),
                                               aivDefinitions.end());
  metadata.artifacts.deviceObjectPath = deviceObjectPath.str();
  metadata.artifacts.packedSharedObjectPath = packedSharedObjectPath.str();
  metadata.artifacts.tilingFilePath = tiling.tilingArtifactPath;
  metadata.artifacts.launchInfoFilePath = tiling.launchInfoPath;
  metadata.abi.workspaceMode = abi.workspaceMode;
  metadata.abi.workspaceBytes = abi.workspaceBytes;
  metadata.abi.tilingMode = abi.tilingMode;
  metadata.abi.tilingSource = abi.tilingSource;
  if (abi.workspaceArgIndex) {
    metadata.abi.workspaceArgIndex = *abi.workspaceArgIndex;
    metadata.abi.hasWorkspaceArgIndex = true;
  }
  if (abi.tilingArgIndex) {
    metadata.abi.tilingArgIndex = *abi.tilingArgIndex;
    metadata.abi.hasTilingArgIndex = true;
  }
  metadata.abi.inputs.reserve(abi.inputs.size());
  for (const auto &tensor : abi.inputs)
    metadata.abi.inputs.push_back(makeMetadataTensorDesc(tensor));
  metadata.abi.outputs.reserve(abi.outputs.size());
  for (const auto &tensor : abi.outputs)
    metadata.abi.outputs.push_back(makeMetadataTensorDesc(tensor));
  metadata.hostLaunch.mode =
      tiling.backendKind == "helper" ? "helper" : "runtime-native";
  metadata.hostLaunch.helperKind =
      tiling.backendKind == "helper" ? "mix-tiling-helper"
                                     : "in-process-mix-tiling";
  llvm::json::Object helperInputs;
  if (!tiling.backendKind.empty())
    helperInputs["tiling_backend"] = tiling.backendKind;
  if (!tiling.strategyName.empty())
    helperInputs["tiling_strategy"] = tiling.strategyName;
  if (!tiling.debugNote.empty())
    helperInputs["tiling_debug_note"] = tiling.debugNote;
  metadata.hostLaunch.helperInputsJson =
      llvm::formatv("{0:2}", llvm::json::Value(std::move(helperInputs))).str();

  auto jsonOr = serializeMixCompileMetadataJson(metadata);
  if (!jsonOr)
    return jsonOr.takeError();
  if (auto err = writeTextFile(metadataPath, *jsonOr))
    return std::move(err);
  return metadataPath.str();
}

} // namespace mlir::runtime
