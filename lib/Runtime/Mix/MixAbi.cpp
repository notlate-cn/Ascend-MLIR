#include "Runtime/MixAbi.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

namespace mlir::runtime {

namespace {

llvm::Expected<std::string>
getRequiredManifestValue(const std::map<std::string, std::string> &manifest,
                         llvm::StringRef key) {
  auto it = manifest.find(key.str());
  if (it == manifest.end() || it->second.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Missing ABI manifest key: %s",
                                   key.str().c_str());
  return it->second;
}

std::optional<std::string>
getOptionalManifestValue(const std::map<std::string, std::string> &manifest,
                         llvm::StringRef key) {
  auto it = manifest.find(key.str());
  if (it == manifest.end() || it->second.empty())
    return std::nullopt;
  return it->second;
}

llvm::Expected<size_t> parseSizeValue(llvm::StringRef value,
                                      llvm::StringRef field) {
  uint64_t parsed = 0;
  if (value.getAsInteger(10, parsed))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Invalid %s: %s", field.str().c_str(),
                                   value.str().c_str());
  return static_cast<size_t>(parsed);
}

llvm::Expected<bool> parseBoolValue(llvm::StringRef value,
                                    llvm::StringRef field) {
  value = value.trim();
  if (value == "0" || value.equals_insensitive("false"))
    return false;
  if (value == "1" || value.equals_insensitive("true"))
    return true;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "Invalid %s: %s", field.str().c_str(),
                                 value.str().c_str());
}

llvm::Expected<DType> parseDType(llvm::StringRef dtype) {
  if (dtype == "f16")
    return DType::F16;
  if (dtype == "f32")
    return DType::F32;
  if (dtype == "bf16")
    return DType::BF16;
  if (dtype == "int8")
    return DType::INT8;
  if (dtype == "int32")
    return DType::INT32;
  if (dtype == "int64")
    return DType::INT64;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "Unsupported ABI dtype: %s",
                                 dtype.str().c_str());
}

llvm::Expected<std::vector<int64_t>>
parseShapeList(llvm::StringRef value) {
  std::vector<int64_t> shape;
  llvm::SmallVector<llvm::StringRef> dims;
  value.split(dims, ',', -1, false);
  for (llvm::StringRef dim : dims) {
    int64_t parsed = 0;
    if (dim.trim().getAsInteger(10, parsed))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Invalid ABI shape dim: %s",
                                     dim.str().c_str());
    shape.push_back(parsed);
  }
  return shape;
}

void appendLine(std::string &out, llvm::StringRef key, llvm::StringRef value) {
  out += key.str();
  out += '=';
  out += value.str();
  out += '\n';
}

void appendSizeLine(std::string &out, llvm::StringRef key, size_t value) {
  appendLine(out, key, std::to_string(value));
}

void appendShapeLine(std::string &out, llvm::StringRef key,
                     llvm::ArrayRef<int64_t> shape) {
  std::string rendered;
  llvm::raw_string_ostream os(rendered);
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i)
      os << ',';
    os << shape[i];
  }
  os.flush();
  appendLine(out, key, rendered);
}

std::string defaultRuntimeFileName(const MixAbiMetadata &abi,
                                   llvm::StringRef tensorName,
                                   bool isOutput) {
  llvm::StringRef kernelName = !abi.runtimeKernelName.empty()
                                   ? llvm::StringRef(abi.runtimeKernelName)
                                   : llvm::StringRef(abi.logicalKernelName);
  if (kernelName.empty())
    return std::string();
  return isOutput ? buildCanonicalOutputFileName(kernelName, tensorName)
                  : buildCanonicalInputFileName(kernelName, tensorName);
}

std::string defaultGoldenFileName(const MixAbiMetadata &abi,
                                  llvm::StringRef tensorName) {
  llvm::StringRef kernelName = !abi.runtimeKernelName.empty()
                                   ? llvm::StringRef(abi.runtimeKernelName)
                                   : llvm::StringRef(abi.logicalKernelName);
  if (kernelName.empty())
    return std::string();
  return buildCanonicalGoldenFileName(kernelName, tensorName);
}

llvm::Expected<MixAbiTensorDesc>
parseTensorDesc(const std::map<std::string, std::string> &manifest,
                llvm::StringRef prefix, size_t index,
                const MixAbiMetadata &abi, bool isOutput) {
  const std::string base = (prefix + std::to_string(index)).str();
  auto getRequired = [&](llvm::StringRef suffix)
      -> llvm::Expected<std::string> {
    const std::string key = base + suffix.str();
    return getRequiredManifestValue(manifest, key);
  };

  MixAbiTensorDesc tensor;
  auto nameOr = getRequired("_name");
  if (!nameOr)
    return nameOr.takeError();
  tensor.name = *nameOr;

  if (auto fileOr = getOptionalManifestValue(manifest, base + "_file")) {
    tensor.runtimeFile = *fileOr;
  } else if (auto runtimeFileOr =
                 getOptionalManifestValue(manifest, base + "_runtime_file")) {
    tensor.runtimeFile = *runtimeFileOr;
  }
  if (tensor.runtimeFile.empty())
    tensor.runtimeFile = defaultRuntimeFileName(abi, tensor.name, isOutput);

  if (auto goldenOr = getOptionalManifestValue(manifest, base + "_golden_file"))
    tensor.goldenFile = *goldenOr;
  if (isOutput && tensor.goldenFile.empty())
    tensor.goldenFile = defaultGoldenFileName(abi, tensor.name);

  auto dtypeOr = getRequired("_dtype");
  if (!dtypeOr)
    return dtypeOr.takeError();
  auto parsedDType = parseDType(*dtypeOr);
  if (!parsedDType)
    return parsedDType.takeError();
  tensor.dtype = *parsedDType;

  auto shapeOr = getRequired("_shape");
  if (!shapeOr)
    return shapeOr.takeError();
  auto parsedShape = parseShapeList(*shapeOr);
  if (!parsedShape)
    return parsedShape.takeError();
  tensor.shape = std::move(*parsedShape);
  return tensor;
}

llvm::Error appendTensorDesc(std::string &out, llvm::StringRef prefix,
                             size_t index, const MixAbiTensorDesc &tensor,
                             llvm::StringRef defaultKernelName,
                             bool isOutput) {
  const std::string base = (prefix + std::to_string(index)).str();
  if (tensor.name.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Missing ABI tensor name for %s",
                                   base.c_str());
  llvm::StringRef runtimeFile = tensor.runtimeFile;
  std::string synthesizedRuntimeFile;
  if (runtimeFile.empty() && !defaultKernelName.empty()) {
    synthesizedRuntimeFile = isOutput
                                ? buildCanonicalOutputFileName(defaultKernelName,
                                                               tensor.name)
                                : buildCanonicalInputFileName(defaultKernelName,
                                                              tensor.name);
    runtimeFile = synthesizedRuntimeFile;
  }
  if (runtimeFile.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Missing ABI tensor file for %s",
                                   base.c_str());

  appendLine(out, base + "_name", tensor.name);
  appendLine(out, base + "_file", runtimeFile);
  appendLine(out, base + "_dtype",
             tensor.dtype == DType::F16   ? "f16"
             : tensor.dtype == DType::F32 ? "f32"
             : tensor.dtype == DType::BF16 ? "bf16"
             : tensor.dtype == DType::INT8 ? "int8"
             : tensor.dtype == DType::INT32 ? "int32"
             : tensor.dtype == DType::INT64 ? "int64"
                                           : "unknown");
  appendShapeLine(out, base + "_shape", tensor.shape);
  if (!tensor.goldenFile.empty())
    appendLine(out, base + "_golden_file", tensor.goldenFile);
  return llvm::Error::success();
}

} // namespace

std::string buildCanonicalInputFileName(llvm::StringRef kernelName,
                                        llvm::StringRef tensorName) {
  return (kernelName + "." + tensorName + ".input.bin").str();
}

std::string buildCanonicalOutputFileName(llvm::StringRef kernelName,
                                         llvm::StringRef tensorName) {
  return (kernelName + "." + tensorName + ".output.bin").str();
}

std::string buildCanonicalGoldenFileName(llvm::StringRef kernelName,
                                         llvm::StringRef tensorName) {
  return (kernelName + "." + tensorName + ".golden.bin").str();
}

std::string normalizeMixKernelName(llvm::StringRef kernelName) {
  llvm::StringRef name = kernelName.trim();
  if (name.starts_with("auto_gen_"))
    name = name.drop_front(strlen("auto_gen_"));
  if (name.ends_with("_kernel"))
    name = name.drop_back(strlen("_kernel"));
  if (name.ends_with("_wrapperless"))
    name = name.drop_back(strlen("_wrapperless"));
  if (name.ends_with("_official_style"))
    name = name.drop_back(strlen("_official_style"));
  if (name.ends_with("_mix"))
    name = name.drop_back(strlen("_mix"));
  return name.str();
}

llvm::Expected<std::string>
serializeMixAbiManifest(const MixAbiMetadata &abi) {
  std::string out;
  if (!abi.runtimeKernelName.empty())
    appendLine(out, "abi_runtime_kernel_name", abi.runtimeKernelName);
  if (!abi.logicalKernelName.empty())
    appendLine(out, "abi_logical_kernel_name", abi.logicalKernelName);
  appendSizeLine(out, "abi_input_count", abi.inputs.size());
  for (size_t i = 0; i < abi.inputs.size(); ++i) {
    if (auto err = appendTensorDesc(
            out, "abi_input", i, abi.inputs[i],
            !abi.runtimeKernelName.empty()
                ? llvm::StringRef(abi.runtimeKernelName)
                : llvm::StringRef(abi.logicalKernelName),
            false))
      return err;
  }
  appendSizeLine(out, "abi_output_count", abi.outputs.size());
  for (size_t i = 0; i < abi.outputs.size(); ++i) {
    if (auto err = appendTensorDesc(
            out, "abi_output", i, abi.outputs[i],
            !abi.runtimeKernelName.empty()
                ? llvm::StringRef(abi.runtimeKernelName)
                : llvm::StringRef(abi.logicalKernelName),
            true))
      return err;
  }
  appendSizeLine(out, "abi_workspace_bytes", abi.workspaceBytes);
  appendSizeLine(out, "abi_block_dim", abi.blockDim);
  appendLine(out, "abi_workspace_mode", abi.workspaceMode);
  appendLine(out, "abi_tiling_mode", abi.tilingMode);
  appendLine(out, "abi_tiling_source", abi.tilingSource);
  appendSizeLine(out, "abi_inputs", abi.inputs.size());
  appendSizeLine(out, "abi_outputs", abi.outputs.size());
  if (abi.workspaceArgIndex)
    appendSizeLine(out, "abi_workspace_arg_index", *abi.workspaceArgIndex);
  if (abi.tilingArgIndex)
    appendSizeLine(out, "abi_tiling_arg_index", *abi.tilingArgIndex);
  if (abi.matmul) {
    appendLine(out, "abi_matmul_op_kind", abi.matmul->opKind);
    appendLine(out, "abi_matmul_trans_a", abi.matmul->transA ? "1" : "0");
    appendLine(out, "abi_matmul_trans_b", abi.matmul->transB ? "1" : "0");
    appendLine(out, "abi_matmul_has_bias", abi.matmul->hasBias ? "1" : "0");
    if (!abi.matmul->layoutA.empty())
      appendLine(out, "abi_matmul_layout_a", abi.matmul->layoutA);
    if (!abi.matmul->layoutB.empty())
      appendLine(out, "abi_matmul_layout_b", abi.matmul->layoutB);
    if (!abi.matmul->layoutC.empty())
      appendLine(out, "abi_matmul_layout_c", abi.matmul->layoutC);
    if (!abi.matmul->epilogueKind.empty())
      appendLine(out, "abi_matmul_epilogue_kind", abi.matmul->epilogueKind);
    if (!abi.matmul->batchShape.empty())
      appendShapeLine(out, "abi_matmul_batch_shape", abi.matmul->batchShape);
  }
  return out;
}

llvm::Expected<MixAbiMetadata>
parseMixAbiManifest(const std::map<std::string, std::string> &manifest) {
  MixAbiMetadata abi;

  if (auto value = getOptionalManifestValue(manifest,
                                            "abi_runtime_kernel_name"))
    abi.runtimeKernelName = *value;
  if (abi.runtimeKernelName.empty()) {
    if (auto value = getOptionalManifestValue(manifest, "kernel_name"))
      abi.runtimeKernelName = *value;
  }

  if (auto value = getOptionalManifestValue(manifest,
                                            "abi_logical_kernel_name"))
    abi.logicalKernelName = *value;
  if (abi.logicalKernelName.empty()) {
    if (auto value = getOptionalManifestValue(manifest, "requested_kernel_name"))
      abi.logicalKernelName = *value;
  }
  if (abi.logicalKernelName.empty())
    abi.logicalKernelName = abi.runtimeKernelName;
  if (abi.runtimeKernelName.empty())
    abi.runtimeKernelName = abi.logicalKernelName;

  auto inputCountOr = getRequiredManifestValue(manifest, "abi_input_count");
  if (!inputCountOr)
    return inputCountOr.takeError();
  auto inputCount = parseSizeValue(*inputCountOr, "abi_input_count");
  if (!inputCount)
    return inputCount.takeError();
  for (size_t i = 0; i < *inputCount; ++i) {
    auto tensorOr = parseTensorDesc(manifest, "abi_input", i, abi, false);
    if (!tensorOr)
      return tensorOr.takeError();
    abi.inputs.push_back(std::move(*tensorOr));
  }

  auto outputCountOr = getRequiredManifestValue(manifest, "abi_output_count");
  if (!outputCountOr)
    return outputCountOr.takeError();
  auto outputCount = parseSizeValue(*outputCountOr, "abi_output_count");
  if (!outputCount)
    return outputCount.takeError();
  for (size_t i = 0; i < *outputCount; ++i) {
    auto tensorOr = parseTensorDesc(manifest, "abi_output", i, abi, true);
    if (!tensorOr)
      return tensorOr.takeError();
    abi.outputs.push_back(std::move(*tensorOr));
  }

  auto workspaceBytesOr =
      getRequiredManifestValue(manifest, "abi_workspace_bytes");
  if (!workspaceBytesOr)
    return workspaceBytesOr.takeError();
  auto workspaceBytes =
      parseSizeValue(*workspaceBytesOr, "abi_workspace_bytes");
  if (!workspaceBytes)
    return workspaceBytes.takeError();
  abi.workspaceBytes = *workspaceBytes;

  if (auto it = manifest.find("abi_block_dim");
      it != manifest.end() && !it->second.empty()) {
    auto blockDim = parseSizeValue(it->second, "abi_block_dim");
    if (!blockDim)
      return blockDim.takeError();
    abi.blockDim = static_cast<uint32_t>(*blockDim);
  }

  auto workspaceModeOr =
      getRequiredManifestValue(manifest, "abi_workspace_mode");
  if (!workspaceModeOr)
    return workspaceModeOr.takeError();
  abi.workspaceMode = *workspaceModeOr;

  auto tilingModeOr = getRequiredManifestValue(manifest, "abi_tiling_mode");
  if (!tilingModeOr)
    return tilingModeOr.takeError();
  abi.tilingMode = *tilingModeOr;

  auto tilingSourceOr =
      getRequiredManifestValue(manifest, "abi_tiling_source");
  if (!tilingSourceOr)
    return tilingSourceOr.takeError();
  abi.tilingSource = *tilingSourceOr;

  if (auto value = getOptionalManifestValue(manifest,
                                            "abi_workspace_arg_index")) {
    auto parsed = parseSizeValue(*value, "abi_workspace_arg_index");
    if (!parsed)
      return parsed.takeError();
    abi.workspaceArgIndex = *parsed;
  }
  if (auto value = getOptionalManifestValue(manifest, "abi_tiling_arg_index")) {
    auto parsed = parseSizeValue(*value, "abi_tiling_arg_index");
    if (!parsed)
      return parsed.takeError();
    abi.tilingArgIndex = *parsed;
  }

  if (auto value = getOptionalManifestValue(manifest, "abi_launcher_symbol"))
    abi.launcherSymbol = *value;

  if (auto value = getOptionalManifestValue(manifest, "abi_aic_entry"))
    abi.aicEntry = *value;

  if (auto value = getOptionalManifestValue(manifest, "abi_aiv_entry"))
    abi.aivEntry = *value;

  if (auto value = getOptionalManifestValue(manifest, "abi_matmul_op_kind")) {
    MixAbiMatmulDesc matmul;
    matmul.opKind = *value;
    if (auto boolValue =
            getOptionalManifestValue(manifest, "abi_matmul_trans_a")) {
      auto parsed = parseBoolValue(*boolValue, "abi_matmul_trans_a");
      if (!parsed)
        return parsed.takeError();
      matmul.transA = *parsed;
    }
    if (auto boolValue =
            getOptionalManifestValue(manifest, "abi_matmul_trans_b")) {
      auto parsed = parseBoolValue(*boolValue, "abi_matmul_trans_b");
      if (!parsed)
        return parsed.takeError();
      matmul.transB = *parsed;
    }
    if (auto boolValue =
            getOptionalManifestValue(manifest, "abi_matmul_has_bias")) {
      auto parsed = parseBoolValue(*boolValue, "abi_matmul_has_bias");
      if (!parsed)
        return parsed.takeError();
      matmul.hasBias = *parsed;
    }
    if (auto layout = getOptionalManifestValue(manifest, "abi_matmul_layout_a"))
      matmul.layoutA = *layout;
    if (auto layout = getOptionalManifestValue(manifest, "abi_matmul_layout_b"))
      matmul.layoutB = *layout;
    if (auto layout = getOptionalManifestValue(manifest, "abi_matmul_layout_c"))
      matmul.layoutC = *layout;
    if (auto epilogue =
            getOptionalManifestValue(manifest, "abi_matmul_epilogue_kind"))
      matmul.epilogueKind = *epilogue;
    if (auto batchShape =
            getOptionalManifestValue(manifest, "abi_matmul_batch_shape")) {
      auto parsed = parseShapeList(*batchShape);
      if (!parsed)
        return parsed.takeError();
      matmul.batchShape = std::move(*parsed);
    }
    abi.matmul = std::move(matmul);
  }

  return abi;
}

} // namespace mlir::runtime
