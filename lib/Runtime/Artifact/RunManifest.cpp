#include "Runtime/RunManifest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

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

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, const char *fieldName) {
  if (auto *array = object.getArray(fieldName))
    return array;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "missing required array field: %s",
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

llvm::StringRef stringifyBackendKind(ExecutionBackendKind backendKind) {
  switch (backendKind) {
  case ExecutionBackendKind::Simulation:
    return "sim";
  case ExecutionBackendKind::Npu:
    return "npu";
  }
  return "unknown";
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
  if (auto path = obj.getString("path"))
    binding.path = path->str();

  if (source == "external_file") {
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
  } else if (source == "input_alias") {
    auto inputOr = requireString(obj, "input");
    if (!inputOr)
      return inputOr.takeError();
    binding.sourceKind = BindingSourceKind::InputAlias;
    binding.aliasedInputName = *inputOr;
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

llvm::Expected<RunTaskSpec>
parseTaskSpec(const llvm::json::Object &root,
              std::optional<llvm::StringRef> defaultArtifactRoot) {
  RunTaskSpec spec;
  auto taskIdOr = requireString(root, "task_id");
  if (!taskIdOr)
    return taskIdOr.takeError();
  spec.taskId = *taskIdOr;

  if (auto artifactRoot = root.getString("artifact_root")) {
    spec.artifactRoot = artifactRoot->str();
  } else if (defaultArtifactRoot) {
    spec.artifactRoot = defaultArtifactRoot->str();
  } else {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "missing required string field: artifact_root");
  }

  auto dependenciesOr = parseStringArray(root, "dependencies");
  if (!dependenciesOr)
    return dependenciesOr.takeError();
  spec.dependencies = std::move(*dependenciesOr);

  auto inputsOr = parseTensorBindings(root, "inputs");
  if (!inputsOr)
    return inputsOr.takeError();
  spec.invocation.inputs = std::move(*inputsOr);

  auto outputsOr = parseTensorBindings(root, "outputs");
  if (!outputsOr)
    return outputsOr.takeError();
  spec.invocation.outputs = std::move(*outputsOr);

  auto expectedOutputsOr = parseTensorBindings(root, "expected_outputs");
  if (!expectedOutputsOr)
    return expectedOutputsOr.takeError();
  spec.invocation.expectedOutputs = std::move(*expectedOutputsOr);

  auto tilingOr = parseTilingBinding(root);
  if (!tilingOr)
    return tilingOr.takeError();
  spec.invocation.tiling = std::move(*tilingOr);

  if (auto blockDim = root.getInteger("block_dim"))
    spec.invocation.blockDim = static_cast<int>(*blockDim);
  if (auto workspaceSize = root.getInteger("workspace_size"))
    spec.invocation.workspaceSize = static_cast<size_t>(*workspaceSize);
  if (auto profiling = root.getBoolean("profiling"))
    spec.invocation.enableProfiling = *profiling;
  if (auto atol = root.getNumber("atol"))
    spec.invocation.atol = *atol;
  if (auto rtol = root.getNumber("rtol"))
    spec.invocation.rtol = *rtol;

  return spec;
}

llvm::Expected<int64_t>
parseOptionalInteger(const llvm::json::Object &object, const char *fieldName,
                     int64_t defaultValue) {
  if (auto value = object.getInteger(fieldName))
    return *value;
  return defaultValue;
}

llvm::Expected<const llvm::json::Object *>
selectStaticScheduleEntry(const llvm::json::Object &kernelEntry,
                          llvm::StringRef kernelId) {
  auto scheduleEntriesOr = requireArray(kernelEntry, "scheduleEntries");
  if (!scheduleEntriesOr)
    return scheduleEntriesOr.takeError();
  if ((*scheduleEntriesOr)->empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "artifact manifest kernel %s has no schedule entries",
                                   kernelId.str().c_str());
  }

  for (const llvm::json::Value &value : **scheduleEntriesOr) {
    auto entryOr = requireObject(&value, "scheduleEntries");
    if (!entryOr)
      return entryOr.takeError();
    if (auto guard = (*entryOr)->getString("guard")) {
      if (*guard == "true")
        return *entryOr;
      continue;
    }
    if (auto fallback = (*entryOr)->getBoolean("fallback")) {
      if (*fallback)
        return *entryOr;
      continue;
    }
    return *entryOr;
  }

  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "artifact manifest kernel %s has no static schedule entry",
      kernelId.str().c_str());
}

std::string renderJsonScalar(const llvm::json::Value &value) {
  if (auto string = value.getAsString())
    return string->str();
  if (auto integer = value.getAsInteger())
    return std::to_string(*integer);
  if (auto boolean = value.getAsBoolean())
    return *boolean ? "true" : "false";
  if (auto number = value.getAsNumber())
    return llvm::formatv("{0}", *number).str();
  return llvm::formatv("{0:0}", value).str();
}

std::string renderTilingParamValue(const llvm::json::Value &value) {
  if (auto *array = value.getAsArray()) {
    std::string rendered;
    llvm::raw_string_ostream os(rendered);
    for (auto [index, element] : llvm::enumerate(*array)) {
      if (index != 0)
        os << ",";
      os << renderJsonScalar(element);
    }
    return os.str();
  }
  return renderJsonScalar(value);
}

std::string renderTilingParams(const llvm::json::Object &scheduleEntry) {
  const llvm::json::Object *params = scheduleEntry.getObject("tilingParams");
  if (!params || params->empty())
    return "";

  std::vector<std::pair<std::string, std::string>> fields;
  fields.reserve(params->size());
  for (const auto &field : *params)
    fields.emplace_back(field.getFirst().str(),
                        renderTilingParamValue(field.getSecond()));
  std::sort(fields.begin(), fields.end());

  std::string rendered;
  llvm::raw_string_ostream os(rendered);
  for (auto [index, field] : llvm::enumerate(fields)) {
    if (index != 0)
      os << ",";
    os << field.first << "=" << field.second;
  }
  return os.str();
}

llvm::json::Array toJsonStringArray(llvm::ArrayRef<std::string> values) {
  llvm::json::Array array;
  for (const std::string &value : values)
    array.push_back(value);
  return array;
}

llvm::Expected<llvm::StringMap<llvm::SmallVector<std::string, 2>>>
parseKernelGraphDependencies(const llvm::json::Object &root,
                             llvm::ArrayRef<std::string> kernelIds) {
  llvm::StringMap<bool> knownKernels;
  for (const std::string &kernelId : kernelIds)
    knownKernels[kernelId] = true;

  llvm::StringMap<llvm::SmallVector<std::string, 2>> dependencies;
  const llvm::json::Object *graph = root.getObject("kernelGraph");
  if (!graph)
    return dependencies;

  const llvm::json::Array *edges = graph->getArray("edges");
  if (!edges)
    return dependencies;

  for (const llvm::json::Value &value : *edges) {
    auto edgeOr = requireObject(&value, "kernelGraph.edges");
    if (!edgeOr)
      return edgeOr.takeError();
    auto fromOr = requireString(**edgeOr, "from");
    if (!fromOr)
      return fromOr.takeError();
    auto toOr = requireString(**edgeOr, "to");
    if (!toOr)
      return toOr.takeError();
    if (!knownKernels.count(*fromOr)) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "artifact manifest kernelGraph edge references unknown source kernel: %s",
          fromOr->c_str());
    }
    if (!knownKernels.count(*toOr)) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "artifact manifest kernelGraph edge references unknown target kernel: %s",
          toOr->c_str());
    }
    llvm::SmallVector<std::string, 2> &deps = dependencies[*toOr];
    if (!llvm::is_contained(deps, *fromOr))
      deps.push_back(*fromOr);
  }
  return dependencies;
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
  auto backendNameOr = requireString(*root, "backend");
  if (!backendNameOr)
    return backendNameOr.takeError();
  auto backendKindOr = parseBackendKind(*backendNameOr);
  if (!backendKindOr)
    return backendKindOr.takeError();
  spec.backendKind = *backendKindOr;

  std::optional<llvm::StringRef> defaultArtifactRoot;
  if (auto artifactRoot = root->getString("artifact_root"))
    defaultArtifactRoot = *artifactRoot;

  if (auto *tasks = root->getArray("tasks")) {
    spec.tasks.reserve(tasks->size());
    for (const llvm::json::Value &value : *tasks) {
      auto taskObjectOr = requireObject(&value, "tasks");
      if (!taskObjectOr)
        return taskObjectOr.takeError();
      auto taskOr = parseTaskSpec(**taskObjectOr, defaultArtifactRoot);
      if (!taskOr)
        return taskOr.takeError();
      spec.tasks.push_back(std::move(*taskOr));
    }
  } else {
    auto taskOr = parseTaskSpec(*root, defaultArtifactRoot);
    if (!taskOr)
      return taskOr.takeError();
    spec.tasks.push_back(std::move(*taskOr));
  }

  if (spec.tasks.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "run manifest must contain at least one task");
  }

  return spec;
}

llvm::Error emitRunManifestFromArtifactManifest(
    const ArtifactManifestPrepareRequest &request) {
  if (request.artifactManifestPath.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "artifact manifest path is required");
  }
  if (request.artifactRoot.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "artifact root is required");
  }
  if (request.outputRunManifestPath.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "output run manifest path is required");
  }

  auto bufferOr =
      llvm::MemoryBuffer::getFile(request.artifactManifestPath, /*IsText=*/true);
  if (!bufferOr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot read artifact manifest: %s",
                                   request.artifactManifestPath.c_str());
  }

  auto jsonOr = llvm::json::parse((*bufferOr)->getBuffer());
  if (!jsonOr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid JSON in artifact manifest: %s",
                                   request.artifactManifestPath.c_str());
  }

  const llvm::json::Object *root = jsonOr->getAsObject();
  if (!root) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "artifact manifest root must be an object: %s",
        request.artifactManifestPath.c_str());
  }

  auto kernelEntriesOr = requireArray(*root, "kernel_entries");
  if (!kernelEntriesOr)
    return kernelEntriesOr.takeError();
  if ((*kernelEntriesOr)->empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "artifact manifest must contain at least one kernel entry");
  }

  struct PreparedKernel {
    std::string kernelId;
    int64_t workspaceSize = 8192;
    std::string tilingParams;
  };

  llvm::SmallVector<PreparedKernel, 8> kernels;
  llvm::SmallVector<std::string, 8> kernelIds;
  for (const llvm::json::Value &value : **kernelEntriesOr) {
    auto entryOr = requireObject(&value, "kernel_entries");
    if (!entryOr)
      return entryOr.takeError();
    auto kernelIdOr = requireString(**entryOr, "kernel_id");
    if (!kernelIdOr)
      return kernelIdOr.takeError();
    auto scheduleEntryOr = selectStaticScheduleEntry(**entryOr, *kernelIdOr);
    if (!scheduleEntryOr)
      return scheduleEntryOr.takeError();
    auto workspaceSizeOr =
        parseOptionalInteger(**entryOr, "workspaceSizeBytes", 8192);
    if (!workspaceSizeOr)
      return workspaceSizeOr.takeError();

    PreparedKernel kernel;
    kernel.kernelId = *kernelIdOr;
    kernel.workspaceSize = *workspaceSizeOr;
    kernel.tilingParams = renderTilingParams(**scheduleEntryOr);
    kernels.push_back(std::move(kernel));
    kernelIds.push_back(*kernelIdOr);
  }

  auto dependenciesOr = parseKernelGraphDependencies(*root, kernelIds);
  if (!dependenciesOr)
    return dependenciesOr.takeError();

  llvm::json::Object runManifest;
  runManifest["backend"] = stringifyBackendKind(request.backendKind).str();
  runManifest["artifact_root"] = request.artifactRoot;
  llvm::json::Array tasks;
  for (const PreparedKernel &kernel : kernels) {
    llvm::json::Object task;
    task["task_id"] = kernel.kernelId;
    auto depsIt = dependenciesOr->find(kernel.kernelId);
    if (depsIt != dependenciesOr->end()) {
      std::vector<std::string> deps(depsIt->second.begin(),
                                    depsIt->second.end());
      task["dependencies"] = toJsonStringArray(deps);
    }
    task["inputs"] = llvm::json::Array{};
    task["outputs"] = llvm::json::Array{};
    if (!kernel.tilingParams.empty()) {
      llvm::json::Object tiling;
      tiling["params"] = kernel.tilingParams;
      task["tiling"] = std::move(tiling);
    }
    task["block_dim"] = 1;
    task["workspace_size"] = kernel.workspaceSize;
    tasks.push_back(std::move(task));
  }
  runManifest["tasks"] = std::move(tasks);

  std::error_code error;
  llvm::raw_fd_ostream os(request.outputRunManifestPath, error,
                          llvm::sys::fs::OF_None);
  if (error) {
    return llvm::createStringError(error, "cannot open run manifest for writing: %s",
                                   request.outputRunManifestPath.c_str());
  }
  llvm::json::OStream json(os, /*IndentSize=*/2);
  json.value(llvm::json::Value(std::move(runManifest)));
  os << "\n";
  os.flush();
  if (os.has_error()) {
    std::error_code writeError = os.error();
    os.clear_error();
    return llvm::createStringError(writeError, "cannot write run manifest: %s",
                                   request.outputRunManifestPath.c_str());
  }
  return llvm::Error::success();
}

} // namespace mlir::runtime
