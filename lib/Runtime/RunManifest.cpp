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

llvm::Expected<TensorBinding> parseTensorBinding(const llvm::json::Object &obj) {
  TensorBinding binding;
  auto nameOr = requireString(obj, "name");
  if (!nameOr)
    return nameOr.takeError();
  auto pathOr = requireString(obj, "path");
  if (!pathOr)
    return pathOr.takeError();
  binding.name = *nameOr;
  binding.path = *pathOr;
  binding.sourceKind = BindingSourceKind::ExternalFile;
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
