#include "Runtime/MixCompileMetadata.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

namespace mlir::runtime {

namespace {

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Value *value, const char *fieldName) {
  if (!value)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "missing required object field: %s",
                                   fieldName);
  if (const auto *object = value->getAsObject())
    return object;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "field must be an object: %s", fieldName);
}

llvm::Expected<std::string>
requireString(const llvm::json::Object &object, const char *fieldName) {
  if (auto value = object.getString(fieldName))
    return value->str();
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "missing required string field: %s",
                                 fieldName);
}

llvm::Expected<uint64_t>
requireUInt64(const llvm::json::Object &object, const char *fieldName) {
  if (auto value = object.getInteger(fieldName))
    return static_cast<uint64_t>(*value);
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "missing required integer field: %s",
                                 fieldName);
}

llvm::Expected<std::vector<std::string>>
parseStringArray(const llvm::json::Object &object, const char *fieldName) {
  std::vector<std::string> values;
  auto *array = object.getArray(fieldName);
  if (!array)
    return values;
  values.reserve(array->size());
  for (const llvm::json::Value &value : *array) {
    auto string = value.getAsString();
    if (!string)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "field must be an array of strings: %s",
                                     fieldName);
    values.push_back(string->str());
  }
  return values;
}

llvm::Expected<std::vector<int64_t>>
parseShapeArray(const llvm::json::Object &object, const char *fieldName) {
  std::vector<int64_t> shape;
  auto *array = object.getArray(fieldName);
  if (!array)
    return shape;
  shape.reserve(array->size());
  for (const llvm::json::Value &value : *array) {
    auto integer = value.getAsInteger();
    if (!integer)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "field must be an array of integers: %s",
                                     fieldName);
    shape.push_back(static_cast<int64_t>(*integer));
  }
  return shape;
}

llvm::Expected<std::vector<MixCompileMetadataTensorDesc>>
parseTensorArray(const llvm::json::Object &object, const char *fieldName) {
  std::vector<MixCompileMetadataTensorDesc> tensors;
  auto *array = object.getArray(fieldName);
  if (!array)
    return tensors;
  tensors.reserve(array->size());
  for (const llvm::json::Value &value : *array) {
    auto tensorObjectOr = requireObject(&value, fieldName);
    if (!tensorObjectOr)
      return tensorObjectOr.takeError();
    MixCompileMetadataTensorDesc tensor;
    auto nameOr = requireString(**tensorObjectOr, "name");
    if (!nameOr)
      return nameOr.takeError();
    tensor.name = *nameOr;
    auto dtypeOr = requireString(**tensorObjectOr, "dtype");
    if (!dtypeOr)
      return dtypeOr.takeError();
    tensor.dtype = *dtypeOr;
    auto runtimeFileOr = requireString(**tensorObjectOr, "runtime_file");
    if (!runtimeFileOr)
      return runtimeFileOr.takeError();
    tensor.runtimeFile = *runtimeFileOr;
    if (auto goldenFile = (*tensorObjectOr)->getString("golden_file"))
      tensor.goldenFile = goldenFile->str();
    auto shapeOr = parseShapeArray(**tensorObjectOr, "shape");
    if (!shapeOr)
      return shapeOr.takeError();
    tensor.shape = std::move(*shapeOr);
    tensors.push_back(std::move(tensor));
  }
  return tensors;
}

llvm::json::Array toJsonStringArray(llvm::ArrayRef<std::string> values) {
  llvm::json::Array array;
  for (const std::string &value : values)
    array.push_back(value);
  return array;
}

llvm::json::Array toJsonShape(llvm::ArrayRef<int64_t> shape) {
  llvm::json::Array array;
  for (int64_t dim : shape)
    array.push_back(dim);
  return array;
}

llvm::json::Array
toJsonTensorArray(llvm::ArrayRef<MixCompileMetadataTensorDesc> tensors) {
  llvm::json::Array array;
  for (const auto &tensor : tensors) {
    llvm::json::Object object;
    object["name"] = tensor.name;
    object["dtype"] = tensor.dtype;
    object["shape"] = toJsonShape(tensor.shape);
    object["runtime_file"] = tensor.runtimeFile;
    if (!tensor.goldenFile.empty())
      object["golden_file"] = tensor.goldenFile;
    array.push_back(std::move(object));
  }
  return array;
}

} // namespace

llvm::Expected<MixCompileMetadata>
parseMixCompileMetadataJson(llvm::StringRef jsonText) {
  auto jsonOr = llvm::json::parse(jsonText);
  if (!jsonOr)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to parse mix metadata JSON: %s",
                                   llvm::toString(jsonOr.takeError()).c_str());

  auto *root = jsonOr->getAsObject();
  if (!root)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "mix metadata root must be an object");

  MixCompileMetadata metadata;
  auto schemaVersionOr = requireUInt64(*root, "schema_version");
  if (!schemaVersionOr)
    return schemaVersionOr.takeError();
  metadata.schemaVersion = *schemaVersionOr;

  auto kernelKindOr = requireString(*root, "kernel_kind");
  if (!kernelKindOr)
    return kernelKindOr.takeError();
  metadata.kernelKind = *kernelKindOr;
  if (metadata.kernelKind != "mix")
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "mix metadata kernel_kind must be 'mix'");

  auto kernelNameOr = requireString(*root, "kernel_name");
  if (!kernelNameOr)
    return kernelNameOr.takeError();
  metadata.kernelName = *kernelNameOr;

  auto runtimeKernelNameOr = requireString(*root, "runtime_kernel_name");
  if (!runtimeKernelNameOr)
    return runtimeKernelNameOr.takeError();
  metadata.runtimeKernelName = *runtimeKernelNameOr;

  auto socVersionOr = requireString(*root, "soc_version");
  if (!socVersionOr)
    return socVersionOr.takeError();
  metadata.socVersion = *socVersionOr;

  auto mixKernelTypeOr = requireString(*root, "mix_kernel_type");
  if (!mixKernelTypeOr)
    return mixKernelTypeOr.takeError();
  metadata.mixKernelType = *mixKernelTypeOr;

  auto launcherSymbolOr = requireString(*root, "launcher_symbol");
  if (!launcherSymbolOr)
    return launcherSymbolOr.takeError();
  metadata.launcherSymbol = *launcherSymbolOr;

  auto entriesOr = requireObject(root->get("entries"), "entries");
  if (!entriesOr)
    return entriesOr.takeError();
  auto aicOr = requireString(**entriesOr, "aic");
  if (!aicOr)
    return aicOr.takeError();
  metadata.entries.aic = *aicOr;
  auto aivOr = requireString(**entriesOr, "aiv");
  if (!aivOr)
    return aivOr.takeError();
  metadata.entries.aiv = *aivOr;

  auto generatedOr = requireObject(root->get("generated"), "generated");
  if (!generatedOr)
    return generatedOr.takeError();
  auto sourcePathOr = requireString(**generatedOr, "source_path");
  if (!sourcePathOr)
    return sourcePathOr.takeError();
  metadata.generated.sourcePath = *sourcePathOr;

  auto deviceCompileOr =
      requireObject(root->get("device_compile"), "device_compile");
  if (!deviceCompileOr)
    return deviceCompileOr.takeError();
  auto aicArchOr = requireString(**deviceCompileOr, "aic_arch");
  if (!aicArchOr)
    return aicArchOr.takeError();
  metadata.deviceCompile.aicArch = *aicArchOr;
  auto aivArchOr = requireString(**deviceCompileOr, "aiv_arch");
  if (!aivArchOr)
    return aivArchOr.takeError();
  metadata.deviceCompile.aivArch = *aivArchOr;
  auto aicDefsOr = parseStringArray(**deviceCompileOr, "aic_definitions");
  if (!aicDefsOr)
    return aicDefsOr.takeError();
  metadata.deviceCompile.aicDefinitions = std::move(*aicDefsOr);
  auto aivDefsOr = parseStringArray(**deviceCompileOr, "aiv_definitions");
  if (!aivDefsOr)
    return aivDefsOr.takeError();
  metadata.deviceCompile.aivDefinitions = std::move(*aivDefsOr);

  auto artifactsOr = requireObject(root->get("artifacts"), "artifacts");
  if (!artifactsOr)
    return artifactsOr.takeError();
  auto deviceObjectPathOr = requireString(**artifactsOr, "device_object_path");
  if (!deviceObjectPathOr)
    return deviceObjectPathOr.takeError();
  metadata.artifacts.deviceObjectPath = *deviceObjectPathOr;
  auto packedSharedObjectPathOr =
      requireString(**artifactsOr, "packed_shared_object_path");
  if (!packedSharedObjectPathOr)
    return packedSharedObjectPathOr.takeError();
  metadata.artifacts.packedSharedObjectPath = *packedSharedObjectPathOr;
  auto tilingFilePathOr = requireString(**artifactsOr, "tiling_file_path");
  if (!tilingFilePathOr)
    return tilingFilePathOr.takeError();
  metadata.artifacts.tilingFilePath = *tilingFilePathOr;
  auto launchInfoFilePathOr =
      requireString(**artifactsOr, "launch_info_file_path");
  if (!launchInfoFilePathOr)
    return launchInfoFilePathOr.takeError();
  metadata.artifacts.launchInfoFilePath = *launchInfoFilePathOr;

  auto abiOr = requireObject(root->get("abi"), "abi");
  if (!abiOr)
    return abiOr.takeError();
  auto workspaceModeOr = requireString(**abiOr, "workspace_mode");
  if (!workspaceModeOr)
    return workspaceModeOr.takeError();
  metadata.abi.workspaceMode = *workspaceModeOr;
  auto workspaceBytesOr = requireUInt64(**abiOr, "workspace_bytes");
  if (!workspaceBytesOr)
    return workspaceBytesOr.takeError();
  metadata.abi.workspaceBytes = *workspaceBytesOr;
  auto tilingModeOr = requireString(**abiOr, "tiling_mode");
  if (!tilingModeOr)
    return tilingModeOr.takeError();
  metadata.abi.tilingMode = *tilingModeOr;
  auto tilingSourceOr = requireString(**abiOr, "tiling_source");
  if (!tilingSourceOr)
    return tilingSourceOr.takeError();
  metadata.abi.tilingSource = *tilingSourceOr;
  if (auto workspaceArgIndex = (*abiOr)->getInteger("workspace_arg_index")) {
    metadata.abi.workspaceArgIndex = static_cast<uint64_t>(*workspaceArgIndex);
    metadata.abi.hasWorkspaceArgIndex = true;
  }
  if (auto tilingArgIndex = (*abiOr)->getInteger("tiling_arg_index")) {
    metadata.abi.tilingArgIndex = static_cast<uint64_t>(*tilingArgIndex);
    metadata.abi.hasTilingArgIndex = true;
  }
  auto inputsOr = parseTensorArray(**abiOr, "inputs");
  if (!inputsOr)
    return inputsOr.takeError();
  metadata.abi.inputs = std::move(*inputsOr);
  auto outputsOr = parseTensorArray(**abiOr, "outputs");
  if (!outputsOr)
    return outputsOr.takeError();
  metadata.abi.outputs = std::move(*outputsOr);

  auto hostLaunchOr = requireObject(root->get("host_launch"), "host_launch");
  if (!hostLaunchOr)
    return hostLaunchOr.takeError();
  auto modeOr = requireString(**hostLaunchOr, "mode");
  if (!modeOr)
    return modeOr.takeError();
  metadata.hostLaunch.mode = *modeOr;
  auto helperKindOr = requireString(**hostLaunchOr, "helper_kind");
  if (!helperKindOr)
    return helperKindOr.takeError();
  metadata.hostLaunch.helperKind = *helperKindOr;
  if (const llvm::json::Value *helperInputsValue =
          (*hostLaunchOr)->get("helper_inputs")) {
    std::string helperInputsText;
    llvm::raw_string_ostream os(helperInputsText);
    os << llvm::formatv("{0:2}", *helperInputsValue);
    os.flush();
    metadata.hostLaunch.helperInputsJson = std::move(helperInputsText);
  } else {
    metadata.hostLaunch.helperInputsJson = "{}";
  }

  return metadata;
}

llvm::Expected<std::string>
serializeMixCompileMetadataJson(const MixCompileMetadata &metadata) {
  llvm::json::Object root;
  root["schema_version"] = static_cast<int64_t>(metadata.schemaVersion);
  root["kernel_kind"] = metadata.kernelKind;
  root["kernel_name"] = metadata.kernelName;
  root["runtime_kernel_name"] = metadata.runtimeKernelName;
  root["soc_version"] = metadata.socVersion;
  root["mix_kernel_type"] = metadata.mixKernelType;
  root["launcher_symbol"] = metadata.launcherSymbol;

  llvm::json::Object entries;
  entries["aic"] = metadata.entries.aic;
  entries["aiv"] = metadata.entries.aiv;
  root["entries"] = std::move(entries);

  llvm::json::Object generated;
  generated["source_path"] = metadata.generated.sourcePath;
  root["generated"] = std::move(generated);

  llvm::json::Object deviceCompile;
  deviceCompile["aic_arch"] = metadata.deviceCompile.aicArch;
  deviceCompile["aiv_arch"] = metadata.deviceCompile.aivArch;
  deviceCompile["aic_definitions"] =
      toJsonStringArray(metadata.deviceCompile.aicDefinitions);
  deviceCompile["aiv_definitions"] =
      toJsonStringArray(metadata.deviceCompile.aivDefinitions);
  root["device_compile"] = std::move(deviceCompile);

  llvm::json::Object artifacts;
  artifacts["device_object_path"] = metadata.artifacts.deviceObjectPath;
  artifacts["packed_shared_object_path"] =
      metadata.artifacts.packedSharedObjectPath;
  artifacts["tiling_file_path"] = metadata.artifacts.tilingFilePath;
  artifacts["launch_info_file_path"] = metadata.artifacts.launchInfoFilePath;
  root["artifacts"] = std::move(artifacts);

  llvm::json::Object abi;
  abi["workspace_mode"] = metadata.abi.workspaceMode;
  abi["workspace_bytes"] = static_cast<int64_t>(metadata.abi.workspaceBytes);
  abi["tiling_mode"] = metadata.abi.tilingMode;
  abi["tiling_source"] = metadata.abi.tilingSource;
  if (metadata.abi.hasWorkspaceArgIndex)
    abi["workspace_arg_index"] =
        static_cast<int64_t>(metadata.abi.workspaceArgIndex);
  if (metadata.abi.hasTilingArgIndex)
    abi["tiling_arg_index"] =
        static_cast<int64_t>(metadata.abi.tilingArgIndex);
  abi["inputs"] = toJsonTensorArray(metadata.abi.inputs);
  abi["outputs"] = toJsonTensorArray(metadata.abi.outputs);
  root["abi"] = std::move(abi);

  llvm::json::Object hostLaunch;
  hostLaunch["mode"] = metadata.hostLaunch.mode;
  hostLaunch["helper_kind"] = metadata.hostLaunch.helperKind;
  auto helperInputsOr = llvm::json::parse(metadata.hostLaunch.helperInputsJson);
  if (!helperInputsOr)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to parse helper_inputs JSON: %s",
                                   llvm::toString(helperInputsOr.takeError())
                                       .c_str());
  hostLaunch["helper_inputs"] = std::move(*helperInputsOr);
  root["host_launch"] = std::move(hostLaunch);

  std::string out;
  llvm::raw_string_ostream os(out);
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(root)));
  os.flush();
  out.push_back('\n');
  return out;
}

} // namespace mlir::runtime
