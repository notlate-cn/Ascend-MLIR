#include "Runtime/RunManifest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

namespace mlir::runtime {

namespace {

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Value *value, const char *fieldName) {
  if (!value)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "missing required object field: %s",
                                   fieldName);
  if (auto *object = value->getAsObject())
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

llvm::Expected<ExecutionBackendKind> parseBackendKind(llvm::StringRef value) {
  if (value == "sim" || value == "simulation")
    return ExecutionBackendKind::Simulation;
  if (value == "npu")
    return ExecutionBackendKind::Npu;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported backend kind: %s",
                                 value.str().c_str());
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

llvm::Expected<std::vector<int64_t>>
parseShape(const llvm::json::Array &array) {
  std::vector<int64_t> shape;
  shape.reserve(array.size());
  for (const llvm::json::Value &value : array) {
    auto integer = value.getAsInteger();
    if (!integer)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "shape entries must be integers");
    shape.push_back(static_cast<int64_t>(*integer));
  }
  return shape;
}

llvm::Expected<TensorBinding> parseTensorBinding(const llvm::json::Object &obj) {
  TensorBinding binding;
  auto nameOr = requireString(obj, "name");
  if (!nameOr)
    return nameOr.takeError();
  binding.name = *nameOr;

  llvm::StringRef source = "external_file";
  if (auto sourceValue = obj.getString("source"))
    source = *sourceValue;

  if (source == "external_file") {
    auto pathOr = requireString(obj, "path");
    if (!pathOr)
      return pathOr.takeError();
    binding.path = *pathOr;
    binding.sourceKind = BindingSourceKind::ExternalFile;
  } else if (source == "task_output") {
    auto upstreamTaskOr = requireString(obj, "upstream_task");
    if (!upstreamTaskOr)
      return upstreamTaskOr.takeError();
    auto upstreamOutputOr = requireString(obj, "upstream_output");
    if (!upstreamOutputOr)
      return upstreamOutputOr.takeError();
    binding.sourceKind = BindingSourceKind::TaskOutput;
    binding.upstreamTaskId = *upstreamTaskOr;
    binding.upstreamOutputName = *upstreamOutputOr;
  } else {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unsupported tensor binding source: %s",
                                   source.str().c_str());
  }

  if (auto *shape = obj.getArray("shape")) {
    auto shapeOr = parseShape(*shape);
    if (!shapeOr)
      return shapeOr.takeError();
    binding.shape = std::move(*shapeOr);
  }
  if (auto dtype = obj.getString("dtype")) {
    auto dtypeOr = parseDType(*dtype);
    if (!dtypeOr)
      return dtypeOr.takeError();
    binding.dtype = *dtypeOr;
  }
  return binding;
}

llvm::Expected<std::vector<TensorBinding>>
parseTensorBindings(const llvm::json::Object &root, const char *fieldName) {
  std::vector<TensorBinding> bindings;
  auto *array = root.getArray(fieldName);
  if (!array)
    return bindings;
  for (const llvm::json::Value &value : *array) {
    auto objectOr = requireObject(&value, fieldName);
    if (!objectOr)
      return objectOr.takeError();
    auto bindingOr = parseTensorBinding(**objectOr);
    if (!bindingOr)
      return bindingOr.takeError();
    bindings.push_back(std::move(*bindingOr));
  }
  return bindings;
}

llvm::Expected<std::optional<TilingBinding>>
parseTilingBinding(const llvm::json::Object &root) {
  const llvm::json::Value *tilingValue = root.get("tiling");
  if (!tilingValue)
    return std::nullopt;
  auto tilingObjectOr = requireObject(tilingValue, "tiling");
  if (!tilingObjectOr)
    return tilingObjectOr.takeError();

  TilingBinding tiling;
  if (auto schema = (*tilingObjectOr)->getString("schema"))
    tiling.schemaPath = schema->str();
  if (auto params = (*tilingObjectOr)->getString("params"))
    tiling.params = params->str();
  if (auto binary = (*tilingObjectOr)->getString("binary"))
    tiling.binaryPath = binary->str();
  return tiling;
}

} // namespace

llvm::Expected<RunManifestSpec> loadRunManifest(const std::string &path) {
  auto bufferOr = llvm::MemoryBuffer::getFile(path, /*IsText=*/true);
  if (!bufferOr)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot read run manifest: %s",
                                   path.c_str());

  auto jsonOr = llvm::json::parse((*bufferOr)->getBuffer());
  if (!jsonOr)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid JSON in run manifest: %s",
                                   path.c_str());

  auto *root = jsonOr->getAsObject();
  if (!root)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "run manifest root must be an object: %s",
                                   path.c_str());

  RunManifestSpec spec;
  auto taskIdOr = requireString(*root, "task_id");
  if (!taskIdOr)
    return taskIdOr.takeError();
  spec.taskId = *taskIdOr;

  auto artifactRootOr = requireString(*root, "artifact_root");
  if (!artifactRootOr)
    return artifactRootOr.takeError();
  spec.artifactRoot = *artifactRootOr;

  auto backendNameOr = requireString(*root, "backend");
  if (!backendNameOr)
    return backendNameOr.takeError();
  auto backendKindOr = parseBackendKind(*backendNameOr);
  if (!backendKindOr)
    return backendKindOr.takeError();
  spec.backendKind = *backendKindOr;

  auto inputsOr = parseTensorBindings(*root, "inputs");
  if (!inputsOr)
    return inputsOr.takeError();
  spec.invocation.inputs = std::move(*inputsOr);

  auto outputsOr = parseTensorBindings(*root, "outputs");
  if (!outputsOr)
    return outputsOr.takeError();
  spec.invocation.outputs = std::move(*outputsOr);

  auto expectedOutputsOr = parseTensorBindings(*root, "expected_outputs");
  if (!expectedOutputsOr)
    return expectedOutputsOr.takeError();
  spec.invocation.expectedOutputs = std::move(*expectedOutputsOr);

  auto tilingOr = parseTilingBinding(*root);
  if (!tilingOr)
    return tilingOr.takeError();
  spec.invocation.tiling = std::move(*tilingOr);

  if (auto blockDim = root->getInteger("block_dim"))
    spec.invocation.blockDim = static_cast<int>(*blockDim);
  if (auto workspaceSize = root->getInteger("workspace_size"))
    spec.invocation.workspaceSize = static_cast<size_t>(*workspaceSize);
  if (auto profiling = root->getBoolean("profiling"))
    spec.invocation.enableProfiling = *profiling;

  return spec;
}

} // namespace mlir::runtime
