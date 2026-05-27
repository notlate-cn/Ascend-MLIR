#include "Runtime/RunManifest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <algorithm>
#include <cctype>
#include <optional>
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

llvm::StringRef dtypeToString(DType dtype) {
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
  }
  return "unknown";
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

llvm::json::Array toJsonShape(llvm::ArrayRef<int64_t> shape) {
  llvm::json::Array array;
  for (int64_t dim : shape)
    array.push_back(dim);
  return array;
}

llvm::json::Object toJsonTensorBinding(const TensorBinding &binding) {
  llvm::json::Object object;
  object["name"] = binding.name;
  switch (binding.sourceKind) {
  case BindingSourceKind::ExternalFile:
    if (!binding.path.empty())
      object["path"] = binding.path;
    break;
  case BindingSourceKind::TaskOutput:
    object["source"] = "task_output";
    object["upstream_task"] = binding.upstreamTaskId;
    object["upstream_output"] = binding.upstreamOutputName;
    break;
  case BindingSourceKind::InputAlias:
    object["source"] = "input_alias";
    object["input"] = binding.aliasedInputName;
    break;
  }
  if (binding.shape)
    object["shape"] = toJsonShape(*binding.shape);
  if (binding.dtype)
    object["dtype"] = dtypeToString(*binding.dtype).str();
  return object;
}

llvm::json::Array toJsonTensorBindings(llvm::ArrayRef<TensorBinding> bindings) {
  llvm::json::Array array;
  for (const TensorBinding &binding : bindings)
    array.push_back(toJsonTensorBinding(binding));
  return array;
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

struct ShapeArgDesc {
  std::string name;
  std::string shapeKey;
};

struct KernelGraphEdge {
  std::string from;
  std::string to;
  std::vector<std::string> carriedBuffers;
};

struct HostTilingBindingDesc {
  std::string id;
  std::string library;
  std::string getTilingSize;
  std::string getTiling;
  std::string getBlockDim;
  std::string getWorkspaceSize;
};

llvm::Expected<std::vector<ShapeArgDesc>>
parseShapeArgOrder(const llvm::json::Object &object) {
  std::vector<ShapeArgDesc> shapeArgs;
  const llvm::json::Array *array = object.getArray("shapeArgOrder");
  if (!array)
    return shapeArgs;
  shapeArgs.reserve(array->size());
  for (const llvm::json::Value &value : *array) {
    auto objectOr = requireObject(&value, "shapeArgOrder");
    if (!objectOr)
      return objectOr.takeError();
    auto nameOr = requireString(**objectOr, "name");
    if (!nameOr)
      return nameOr.takeError();
    auto shapeKeyOr = requireString(**objectOr, "shapeKey");
    if (!shapeKeyOr)
      return shapeKeyOr.takeError();
    shapeArgs.push_back({std::move(*nameOr), std::move(*shapeKeyOr)});
  }
  return shapeArgs;
}

llvm::Expected<TensorBinding>
parseAbiBindingDescriptor(const llvm::json::Object &object,
                          llvm::StringRef defaultName) {
  TensorBinding binding;
  if (auto name = object.getString("name"))
    binding.name = name->str();
  else
    binding.name = defaultName.str();

  if (auto path = object.getString("path"))
    binding.path = path->str();
  else if (auto runtimeFile = object.getString("runtime_file"))
    binding.path = runtimeFile->str();
  else if (auto file = object.getString("file"))
    binding.path = file->str();

  if (auto *shape = object.getArray("shape")) {
    auto shapeOr = parseShape(*shape);
    if (!shapeOr)
      return shapeOr.takeError();
    binding.shape = std::move(*shapeOr);
  }
  if (auto dtype = object.getString("dtype")) {
    auto dtypeOr = parseDType(*dtype);
    if (!dtypeOr)
      return dtypeOr.takeError();
    binding.dtype = *dtypeOr;
  }
  return binding;
}

llvm::Expected<std::vector<TensorBinding>>
parseAbiBindingList(const llvm::json::Object *abi, const char *fieldName,
                    int64_t fallbackCount, llvm::StringRef fallbackPrefix) {
  std::vector<TensorBinding> bindings;
  if (!abi) {
    for (int64_t i = 0; i < fallbackCount; ++i) {
      TensorBinding binding;
      binding.name = (fallbackPrefix + llvm::Twine(i)).str();
      bindings.push_back(std::move(binding));
    }
    return bindings;
  }

  if (const llvm::json::Array *array = abi->getArray(fieldName)) {
    bindings.reserve(array->size());
    for (auto [index, value] : llvm::enumerate(*array)) {
      auto objectOr = requireObject(&value, fieldName);
      if (!objectOr)
        return objectOr.takeError();
      auto bindingOr = parseAbiBindingDescriptor(
          **objectOr, (fallbackPrefix + llvm::Twine(index)).str());
      if (!bindingOr)
        return bindingOr.takeError();
      bindings.push_back(std::move(*bindingOr));
    }
    return bindings;
  }

  for (int64_t i = 0; i < fallbackCount; ++i) {
    TensorBinding binding;
    binding.name = (fallbackPrefix + llvm::Twine(i)).str();
    bindings.push_back(std::move(binding));
  }
  return bindings;
}

void addShapeValueAliases(llvm::StringMap<int64_t> &shapeValues,
                          llvm::StringRef lhs, llvm::StringRef rhs) {
  auto lhsIt = shapeValues.find(lhs);
  auto rhsIt = shapeValues.find(rhs);
  if (lhsIt == shapeValues.end() && rhsIt == shapeValues.end())
    return;
  if (lhsIt == shapeValues.end()) {
    shapeValues[lhs] = rhsIt->second;
    return;
  }
  if (rhsIt == shapeValues.end())
    shapeValues[rhs] = lhsIt->second;
}

void concretizeBindingShape(TensorBinding &binding,
                            const llvm::StringMap<int64_t> &shapeValues,
                            llvm::StringRef prefix, size_t bindingIndex) {
  if (!binding.shape)
    return;
  for (auto [dimIndex, dim] : llvm::enumerate(*binding.shape)) {
    if (dim >= 0)
      continue;
    const std::string key =
        (prefix + llvm::Twine(bindingIndex) + "_dim" + llvm::Twine(dimIndex)).str();
    auto it = shapeValues.find(key);
    if (it != shapeValues.end())
      dim = it->second;
  }
}

void resolveBindingPath(TensorBinding &binding, llvm::StringRef artifactRoot) {
  if (binding.path.empty() || llvm::sys::path::is_absolute(binding.path))
    return;
  llvm::SmallString<256> resolved(artifactRoot);
  llvm::sys::path::append(resolved, binding.path);
  binding.path = resolved.str().str();
}

llvm::Expected<int64_t>
evalGuardValue(llvm::StringRef expr,
               const llvm::StringMap<int64_t> &shapeValues) {
  expr = expr.trim();
  if (expr.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "empty guard expression");
  size_t modPos = expr.find('%');
  if (modPos != llvm::StringRef::npos) {
    auto lhsOr = evalGuardValue(expr.substr(0, modPos), shapeValues);
    if (!lhsOr)
      return lhsOr.takeError();
    auto rhsOr = evalGuardValue(expr.substr(modPos + 1), shapeValues);
    if (!rhsOr)
      return rhsOr.takeError();
    if (*rhsOr == 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "guard expression modulo by zero");
    return *lhsOr % *rhsOr;
  }

  int64_t integer = 0;
  if (!expr.getAsInteger(10, integer))
    return integer;
  auto it = shapeValues.find(expr);
  if (it != shapeValues.end())
    return it->second;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "guard expression references unknown shape value: %s",
                                 expr.str().c_str());
}

llvm::Expected<bool>
evalGuardComparison(llvm::StringRef expr,
                    const llvm::StringMap<int64_t> &shapeValues) {
  expr = expr.trim();
  static constexpr llvm::StringLiteral ops[] = {"==", "!=", "<=", ">=", "<", ">"};
  for (llvm::StringRef op : ops) {
    size_t pos = expr.find(op);
    if (pos == llvm::StringRef::npos)
      continue;
    auto lhsOr = evalGuardValue(expr.substr(0, pos), shapeValues);
    if (!lhsOr)
      return lhsOr.takeError();
    auto rhsOr = evalGuardValue(expr.substr(pos + op.size()), shapeValues);
    if (!rhsOr)
      return rhsOr.takeError();
    if (op == "==")
      return *lhsOr == *rhsOr;
    if (op == "!=")
      return *lhsOr != *rhsOr;
    if (op == "<=")
      return *lhsOr <= *rhsOr;
    if (op == ">=")
      return *lhsOr >= *rhsOr;
    if (op == "<")
      return *lhsOr < *rhsOr;
    if (op == ">")
      return *lhsOr > *rhsOr;
  }
  auto valueOr = evalGuardValue(expr, shapeValues);
  if (!valueOr)
    return valueOr.takeError();
  return *valueOr != 0;
}

llvm::Expected<bool> evalGuardExpr(llvm::StringRef expr,
                                   const llvm::StringMap<int64_t> &shapeValues) {
  expr = expr.trim();
  if (expr == "true")
    return true;
  if (expr == "false")
    return false;

  llvm::SmallVector<llvm::StringRef, 4> disjunctions;
  expr.split(disjunctions, "||");
  for (llvm::StringRef disjunction : disjunctions) {
    bool all = true;
    llvm::SmallVector<llvm::StringRef, 4> conjunctions;
    disjunction.split(conjunctions, "&&");
    for (llvm::StringRef conjunction : conjunctions) {
      auto resultOr = evalGuardComparison(conjunction, shapeValues);
      if (!resultOr)
        return resultOr.takeError();
      all = all && *resultOr;
      if (!all)
        break;
    }
    if (all)
      return true;
  }
  return false;
}

llvm::Expected<const llvm::json::Object *>
selectScheduleEntry(const llvm::json::Object &kernelEntry,
                    llvm::StringRef kernelId,
                    const llvm::StringMap<int64_t> &shapeValues) {
  auto scheduleEntriesOr = requireArray(kernelEntry, "scheduleEntries");
  if (!scheduleEntriesOr)
    return scheduleEntriesOr.takeError();
  if ((*scheduleEntriesOr)->empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "artifact manifest kernel %s has no schedule entries",
                                   kernelId.str().c_str());
  }

  struct Candidate {
    const llvm::json::Object *entry = nullptr;
    int64_t priority = 0;
    size_t order = 0;
    bool fallback = false;
  };
  std::vector<Candidate> candidates;
  candidates.reserve((*scheduleEntriesOr)->size());
  for (auto [order, value] : llvm::enumerate(**scheduleEntriesOr)) {
    auto entryOr = requireObject(&value, "scheduleEntries");
    if (!entryOr)
      return entryOr.takeError();
    Candidate candidate;
    candidate.entry = *entryOr;
    candidate.priority = (*entryOr)->getInteger("priority").value_or(
        static_cast<int64_t>(order));
    candidate.order = order;
    candidate.fallback = (*entryOr)->getBoolean("fallback").value_or(false);
    candidates.push_back(candidate);
  }

  llvm::stable_sort(candidates, [](const Candidate &lhs, const Candidate &rhs) {
    if (lhs.priority != rhs.priority)
      return lhs.priority < rhs.priority;
    return lhs.order < rhs.order;
  });

  for (const Candidate &candidate : candidates) {
    if (candidate.fallback)
      continue;
    llvm::StringRef guard = "true";
    if (auto guardValue = candidate.entry->getString("guard"))
      guard = *guardValue;
    auto matchesOr = evalGuardExpr(guard, shapeValues);
    if (!matchesOr)
      return matchesOr.takeError();
    if (*matchesOr)
      return candidate.entry;
  }

  for (const Candidate &candidate : candidates) {
    if (!candidate.fallback)
      continue;
    llvm::StringRef guard = "true";
    if (auto guardValue = candidate.entry->getString("guard"))
      guard = *guardValue;
    auto matchesOr = evalGuardExpr(guard, shapeValues);
    if (!matchesOr)
      return matchesOr.takeError();
    if (*matchesOr)
      return candidate.entry;
  }

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "artifact manifest kernel %s has no matching schedule entry",
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

llvm::Expected<std::vector<KernelGraphEdge>>
parseKernelGraphEdges(const llvm::json::Object &root,
                      llvm::ArrayRef<std::string> kernelIds) {
  llvm::StringMap<bool> knownKernels;
  for (const std::string &kernelId : kernelIds)
    knownKernels[kernelId] = true;

  std::vector<KernelGraphEdge> parsedEdges;
  const llvm::json::Object *graph = root.getObject("kernelGraph");
  if (!graph)
    return parsedEdges;

  const llvm::json::Array *edges = graph->getArray("edges");
  if (!edges)
    return parsedEdges;

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
    KernelGraphEdge edge;
    edge.from = std::move(*fromOr);
    edge.to = std::move(*toOr);
    if (auto carriedBuffersOr = parseStringArray(**edgeOr, "carriedBuffers"))
      edge.carriedBuffers = std::move(*carriedBuffersOr);
    else
      return carriedBuffersOr.takeError();
    parsedEdges.push_back(std::move(edge));
  }
  return parsedEdges;
}

llvm::StringMap<llvm::SmallVector<std::string, 2>>
buildKernelGraphDependencies(llvm::ArrayRef<KernelGraphEdge> edges) {
  llvm::StringMap<llvm::SmallVector<std::string, 2>> dependencies;
  for (const KernelGraphEdge &edge : edges) {
    llvm::SmallVector<std::string, 2> &deps = dependencies[edge.to];
    if (!llvm::is_contained(deps, edge.from))
      deps.push_back(edge.from);
  }
  return dependencies;
}

llvm::Expected<std::vector<HostTilingBindingDesc>>
parseHostTilingBindings(const llvm::json::Object &object) {
  std::vector<HostTilingBindingDesc> bindings;
  const llvm::json::Array *array = object.getArray("hostTilingBindings");
  if (!array)
    return bindings;
  bindings.reserve(array->size());
  for (const llvm::json::Value &value : *array) {
    auto bindingObjectOr = requireObject(&value, "hostTilingBindings");
    if (!bindingObjectOr)
      return bindingObjectOr.takeError();
    auto idOr = requireString(**bindingObjectOr, "id");
    if (!idOr)
      return idOr.takeError();
    auto libraryOr = requireString(**bindingObjectOr, "library");
    if (!libraryOr)
      return libraryOr.takeError();
    auto symbolsOr =
        requireObject((*bindingObjectOr)->get("symbols"), "symbols");
    if (!symbolsOr)
      return symbolsOr.takeError();
    auto getTilingSizeOr = requireString(**symbolsOr, "getTilingSize");
    if (!getTilingSizeOr)
      return getTilingSizeOr.takeError();
    auto getTilingOr = requireString(**symbolsOr, "getTiling");
    if (!getTilingOr)
      return getTilingOr.takeError();
    auto getBlockDimOr = requireString(**symbolsOr, "getBlockDim");
    if (!getBlockDimOr)
      return getBlockDimOr.takeError();
    auto getWorkspaceSizeOr = requireString(**symbolsOr, "getWorkspaceSize");
    if (!getWorkspaceSizeOr)
      return getWorkspaceSizeOr.takeError();

    HostTilingBindingDesc desc;
    desc.id = std::move(*idOr);
    desc.library = std::move(*libraryOr);
    desc.getTilingSize = std::move(*getTilingSizeOr);
    desc.getTiling = std::move(*getTilingOr);
    desc.getBlockDim = std::move(*getBlockDimOr);
    desc.getWorkspaceSize = std::move(*getWorkspaceSizeOr);
    bindings.push_back(std::move(desc));
  }
  return bindings;
}

const HostTilingBindingDesc *
findHostTilingBinding(llvm::ArrayRef<HostTilingBindingDesc> rootBindings,
                      llvm::ArrayRef<HostTilingBindingDesc> kernelBindings,
                      llvm::StringRef id) {
  for (const HostTilingBindingDesc &binding : kernelBindings)
    if (binding.id == id)
      return &binding;
  for (const HostTilingBindingDesc &binding : rootBindings)
    if (binding.id == id)
      return &binding;
  return nullptr;
}

std::string resolvePath(llvm::StringRef base, llvm::StringRef path) {
  if (path.empty() || llvm::sys::path::is_absolute(path))
    return path.str();
  llvm::SmallString<256> resolved(base);
  llvm::sys::path::append(resolved, path);
  return resolved.str().str();
}

bool hasSupportedArtifactManifest(llvm::StringRef artifactRoot) {
  llvm::SmallString<256> candidate(artifactRoot);
  llvm::sys::path::append(candidate, "out", "manifest.txt");
  if (llvm::sys::fs::exists(candidate))
    return true;
  candidate = artifactRoot;
  llvm::sys::path::append(candidate, "mix-artifact.txt");
  if (llvm::sys::fs::exists(candidate))
    return true;
  candidate = artifactRoot;
  llvm::sys::path::append(candidate, "out", "mix-artifact.txt");
  return llvm::sys::fs::exists(candidate);
}

std::string resolveTaskArtifactRoot(llvm::StringRef artifactRoot,
                                    llvm::StringRef kernelId) {
  llvm::SmallString<256> perKernelRoot(artifactRoot);
  llvm::sys::path::append(perKernelRoot, kernelId);
  if (hasSupportedArtifactManifest(perKernelRoot))
    return perKernelRoot.str().str();
  return artifactRoot.str();
}

std::string sanitizePathComponent(llvm::StringRef value) {
  std::string sanitized;
  sanitized.reserve(value.size());
  for (char c : value) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' ||
        c == '.')
      sanitized.push_back(c);
    else
      sanitized.push_back('_');
  }
  return sanitized.empty() ? "kernel" : sanitized;
}

std::string makeTilingBinaryPath(llvm::StringRef runManifestPath,
                                 llvm::StringRef kernelId) {
  llvm::SmallString<256> dir(runManifestPath);
  llvm::SmallString<128> stem(llvm::sys::path::stem(runManifestPath));
  llvm::sys::path::remove_filename(dir);
  if (dir.empty())
    dir = ".";
  std::string filename =
      (llvm::Twine(stem) + "." + sanitizePathComponent(kernelId) + ".tiling.bin")
          .str();
  llvm::sys::path::append(dir, filename);
  return dir.str().str();
}

std::string makeDefaultOutputPath(llvm::StringRef outputDir,
                                  llvm::StringRef kernelId,
                                  llvm::StringRef outputName) {
  llvm::SmallString<256> path(outputDir);
  std::string filename =
      (sanitizePathComponent(kernelId) + "." +
       sanitizePathComponent(outputName) + ".actual.npy");
  llvm::sys::path::append(path, filename);
  return path.str().str();
}

using GetTilingSizeFn = int32_t (*)();
using GetTilingFn = int32_t (*)(const int64_t *, int32_t, void *);
using GetBlockDimFn = int64_t (*)(const int64_t *, int32_t);
using GetWorkspaceSizeFn = int64_t (*)(const int64_t *, int32_t);

template <typename FnTy>
llvm::Expected<FnTy> loadHostTilingSymbol(void *handle, llvm::StringRef symbol,
                                          llvm::StringRef library) {
  dlerror();
  void *address = dlsym(handle, symbol.str().c_str());
  const char *error = dlerror();
  if (error || !address) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot resolve host tiling symbol %s from %s: %s",
                                   symbol.str().c_str(), library.str().c_str(),
                                   error ? error : "symbol not found");
  }
  return reinterpret_cast<FnTy>(address);
}

llvm::Error writeBinaryFile(llvm::StringRef path,
                            llvm::ArrayRef<uint8_t> bytes) {
  std::error_code error;
  llvm::raw_fd_ostream os(path, error, llvm::sys::fs::OF_None);
  if (error)
    return llvm::createStringError(error, "cannot open tiling binary for writing: %s",
                                   path.str().c_str());
  os.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  os.flush();
  if (os.has_error()) {
    std::error_code writeError = os.error();
    os.clear_error();
    return llvm::createStringError(writeError, "cannot write tiling binary: %s",
                                   path.str().c_str());
  }
  return llvm::Error::success();
}

llvm::Error materializeHostTiling(const HostTilingBindingDesc &binding,
                                  llvm::StringRef artifactRoot,
                                  llvm::StringRef outputRunManifestPath,
                                  llvm::StringRef kernelId,
                                  llvm::ArrayRef<int64_t> shapeArgs,
                                  TilingBinding &tiling,
                                  int64_t &blockDim,
                                  int64_t &workspaceSize) {
  const std::string libraryPath = resolvePath(artifactRoot, binding.library);
  void *handle = dlopen(libraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot open host tiling library %s: %s",
                                   libraryPath.c_str(), dlerror());
  }
  struct ScopedDlClose {
    void *handle = nullptr;
    ~ScopedDlClose() {
      if (handle)
        dlclose(handle);
    }
  } scoped{handle};

  auto getTilingSizeOr =
      loadHostTilingSymbol<GetTilingSizeFn>(handle, binding.getTilingSize,
                                            libraryPath);
  if (!getTilingSizeOr)
    return getTilingSizeOr.takeError();
  auto getTilingOr =
      loadHostTilingSymbol<GetTilingFn>(handle, binding.getTiling, libraryPath);
  if (!getTilingOr)
    return getTilingOr.takeError();
  auto getBlockDimOr =
      loadHostTilingSymbol<GetBlockDimFn>(handle, binding.getBlockDim,
                                          libraryPath);
  if (!getBlockDimOr)
    return getBlockDimOr.takeError();
  auto getWorkspaceSizeOr =
      loadHostTilingSymbol<GetWorkspaceSizeFn>(handle, binding.getWorkspaceSize,
                                               libraryPath);
  if (!getWorkspaceSizeOr)
    return getWorkspaceSizeOr.takeError();

  int32_t tilingSize = (*getTilingSizeOr)();
  if (tilingSize < 0) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "host tiling size query failed for kernel %s",
                                   kernelId.str().c_str());
  }
  std::vector<uint8_t> tilingBytes(static_cast<size_t>(tilingSize));
  int32_t shapeCount = static_cast<int32_t>(shapeArgs.size());
  int32_t rc = (*getTilingOr)(shapeArgs.data(), shapeCount, tilingBytes.data());
  if (rc != 0) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "host tiling query failed for kernel %s with code %d",
                                   kernelId.str().c_str(), rc);
  }

  blockDim = (*getBlockDimOr)(shapeArgs.data(), shapeCount);
  if (blockDim < 0) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "host block_dim query failed for kernel %s",
                                   kernelId.str().c_str());
  }
  workspaceSize = (*getWorkspaceSizeOr)(shapeArgs.data(), shapeCount);
  if (workspaceSize < 0) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "host workspace size query failed for kernel %s",
                                   kernelId.str().c_str());
  }

  tiling.binaryPath = makeTilingBinaryPath(outputRunManifestPath, kernelId);
  return writeBinaryFile(tiling.binaryPath, tilingBytes);
}

bool bindingsCompatible(const TensorBinding &producer,
                        const TensorBinding &consumer) {
  if (producer.dtype && consumer.dtype && *producer.dtype != *consumer.dtype)
    return false;
  if (producer.shape && consumer.shape && *producer.shape != *consumer.shape)
    return false;
  return true;
}

std::optional<size_t> parseOperandHint(llvm::StringRef carriedBuffer) {
  size_t pos = carriedBuffer.rfind("_operand");
  if (pos == llvm::StringRef::npos)
    return std::nullopt;
  llvm::StringRef suffix = carriedBuffer.substr(pos + strlen("_operand"));
  uint64_t value = 0;
  if (suffix.getAsInteger(10, value))
    return std::nullopt;
  return static_cast<size_t>(value);
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
    std::string artifactRoot;
    std::vector<TensorBinding> inputs;
    std::vector<TensorBinding> outputs;
    std::vector<TensorBinding> expectedOutputs;
    std::vector<int64_t> shapeArgs;
    TilingBinding tiling;
    bool hasTiling = false;
    int64_t blockDim = 1;
    int64_t workspaceSize = 8192;
    std::string tilingParams;
  };

  auto rootHostTilingBindingsOr = parseHostTilingBindings(*root);
  if (!rootHostTilingBindingsOr)
    return rootHostTilingBindingsOr.takeError();
  std::vector<HostTilingBindingDesc> rootHostTilingBindings =
      std::move(*rootHostTilingBindingsOr);

  llvm::StringMap<int64_t> requestShapeValues;
  for (const auto &shapeArg : request.shapeArgs)
    requestShapeValues[shapeArg.first] = shapeArg.second;

  std::vector<PreparedKernel> kernels;
  llvm::SmallVector<std::string, 8> kernelIds;
  for (const llvm::json::Value &value : **kernelEntriesOr) {
    auto entryOr = requireObject(&value, "kernel_entries");
    if (!entryOr)
      return entryOr.takeError();
    auto kernelIdOr = requireString(**entryOr, "kernel_id");
    if (!kernelIdOr)
      return kernelIdOr.takeError();

    const llvm::json::Object *abi = (*entryOr)->getObject("abi");
    int64_t numInputs = abi ? abi->getInteger("numInputs").value_or(0) : 0;
    int64_t numOutputs = abi ? abi->getInteger("numOutputs").value_or(0) : 0;
    auto inputsOr = parseAbiBindingList(abi, "inputs", numInputs, "arg");
    if (!inputsOr)
      return inputsOr.takeError();
    auto outputsOr = parseAbiBindingList(abi, "outputs", numOutputs, "out");
    if (!outputsOr)
      return outputsOr.takeError();

    llvm::StringMap<int64_t> shapeValues = requestShapeValues;
    for (auto [index, binding] : llvm::enumerate(*inputsOr)) {
      if (!binding.shape)
        continue;
      for (auto [dimIndex, dim] : llvm::enumerate(*binding.shape)) {
        if (dim >= 0) {
          shapeValues[(llvm::Twine("arg") + llvm::Twine(index) + "_dim" +
                       llvm::Twine(dimIndex))
                          .str()] = dim;
        }
      }
    }

    auto shapeArgsOr = parseShapeArgOrder(**entryOr);
    if (!shapeArgsOr)
      return shapeArgsOr.takeError();
    if (shapeArgsOr->empty()) {
      auto rootShapeArgsOr = parseShapeArgOrder(*root);
      if (!rootShapeArgsOr)
        return rootShapeArgsOr.takeError();
      *shapeArgsOr = std::move(*rootShapeArgsOr);
    }
    for (const ShapeArgDesc &shapeArg : *shapeArgsOr)
      addShapeValueAliases(shapeValues, shapeArg.name, shapeArg.shapeKey);

    for (auto [index, binding] : llvm::enumerate(*inputsOr)) {
      concretizeBindingShape(binding, shapeValues, "arg", index);
      if (!binding.shape)
        continue;
      for (auto [dimIndex, dim] : llvm::enumerate(*binding.shape)) {
        if (dim >= 0) {
          shapeValues[(llvm::Twine("arg") + llvm::Twine(index) + "_dim" +
                       llvm::Twine(dimIndex))
                          .str()] = dim;
        }
      }
    }
    for (auto [index, binding] : llvm::enumerate(*outputsOr))
      concretizeBindingShape(binding, shapeValues, "out", index);
    for (const ShapeArgDesc &shapeArg : *shapeArgsOr)
      addShapeValueAliases(shapeValues, shapeArg.name, shapeArg.shapeKey);

    auto scheduleEntryOr = selectScheduleEntry(**entryOr, *kernelIdOr, shapeValues);
    if (!scheduleEntryOr)
      return scheduleEntryOr.takeError();
    auto workspaceSizeOr =
        parseOptionalInteger(**entryOr, "workspaceSizeBytes", 8192);
    if (!workspaceSizeOr)
      return workspaceSizeOr.takeError();

    PreparedKernel kernel;
    kernel.kernelId = *kernelIdOr;
    kernel.artifactRoot = resolveTaskArtifactRoot(request.artifactRoot, kernel.kernelId);
    kernel.inputs = std::move(*inputsOr);
    kernel.outputs = std::move(*outputsOr);
    for (TensorBinding &binding : kernel.inputs)
      resolveBindingPath(binding, request.artifactRoot);
    for (TensorBinding &binding : kernel.outputs)
      resolveBindingPath(binding, request.artifactRoot);

    for (const ShapeArgDesc &shapeArg : *shapeArgsOr) {
      auto it = shapeValues.find(shapeArg.shapeKey);
      if (it == shapeValues.end())
        it = shapeValues.find(shapeArg.name);
      if (it == shapeValues.end()) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "artifact manifest kernel %s missing concrete shape argument: %s",
            kernel.kernelId.c_str(), shapeArg.shapeKey.c_str());
      }
      kernel.shapeArgs.push_back(it->second);
    }

    kernel.workspaceSize = *workspaceSizeOr;
    if (auto scheduleWorkspaceSize =
            (*scheduleEntryOr)->getInteger("workspaceSizeBytes"))
      kernel.workspaceSize = *scheduleWorkspaceSize;
    if (auto blockDim = (*scheduleEntryOr)->getInteger("blockDim"))
      kernel.blockDim = *blockDim;
    kernel.tilingParams = renderTilingParams(**scheduleEntryOr);
    if (!kernel.tilingParams.empty()) {
      kernel.tiling.params = kernel.tilingParams;
      kernel.hasTiling = true;
    }

    if (auto hostTilingId = (*scheduleEntryOr)->getString("hostTilingId")) {
      auto kernelHostTilingBindingsOr = parseHostTilingBindings(**entryOr);
      if (!kernelHostTilingBindingsOr)
        return kernelHostTilingBindingsOr.takeError();
      const HostTilingBindingDesc *binding = findHostTilingBinding(
          rootHostTilingBindings, *kernelHostTilingBindingsOr, *hostTilingId);
      if (!binding) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "artifact manifest schedule entry references unknown hostTilingId: %s",
            hostTilingId->str().c_str());
      }
      if (auto err = materializeHostTiling(
              *binding, request.artifactRoot, request.outputRunManifestPath,
              kernel.kernelId, kernel.shapeArgs, kernel.tiling, kernel.blockDim,
              kernel.workspaceSize))
        return err;
      kernel.hasTiling = true;
    }

    kernels.push_back(std::move(kernel));
    kernelIds.push_back(*kernelIdOr);
  }

  auto graphEdgesOr = parseKernelGraphEdges(*root, kernelIds);
  if (!graphEdgesOr)
    return graphEdgesOr.takeError();
  auto dependencies = buildKernelGraphDependencies(*graphEdgesOr);

  llvm::StringMap<size_t> kernelIndex;
  for (auto [index, kernel] : llvm::enumerate(kernels))
    kernelIndex[kernel.kernelId] = index;

  for (const KernelGraphEdge &edge : *graphEdgesOr) {
    auto producerIt = kernelIndex.find(edge.from);
    auto consumerIt = kernelIndex.find(edge.to);
    if (producerIt == kernelIndex.end() || consumerIt == kernelIndex.end())
      continue;
    PreparedKernel &producer = kernels[producerIt->second];
    PreparedKernel &consumer = kernels[consumerIt->second];
    llvm::SmallVector<std::string, 1> carriedBuffers(edge.carriedBuffers.begin(),
                                                     edge.carriedBuffers.end());
    if (carriedBuffers.empty())
      carriedBuffers.push_back("");
    for (llvm::StringRef carriedBuffer : carriedBuffers) {
      std::optional<size_t> hintedInput = parseOperandHint(carriedBuffer);
      size_t selectedInput = consumer.inputs.size();
      size_t selectedOutput = producer.outputs.size();
      auto tryAssign = [&](size_t inputIndex) -> bool {
        if (inputIndex >= consumer.inputs.size() ||
            consumer.inputs[inputIndex].sourceKind !=
                BindingSourceKind::ExternalFile)
          return false;
        for (auto [outputIndex, output] : llvm::enumerate(producer.outputs)) {
          if (!bindingsCompatible(output, consumer.inputs[inputIndex]))
            continue;
          selectedInput = inputIndex;
          selectedOutput = outputIndex;
          return true;
        }
        return false;
      };

      if (hintedInput)
        (void)tryAssign(*hintedInput);
      if (selectedInput == consumer.inputs.size()) {
        for (size_t inputIndex = 0; inputIndex < consumer.inputs.size();
             ++inputIndex) {
          if (tryAssign(inputIndex))
            break;
        }
      }
      if (selectedInput == consumer.inputs.size() ||
          selectedOutput == producer.outputs.size()) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "cannot map carried buffer from %s to %s in artifact manifest",
            edge.from.c_str(), edge.to.c_str());
      }

      TensorBinding &input = consumer.inputs[selectedInput];
      input.sourceKind = BindingSourceKind::TaskOutput;
      input.path.clear();
      input.upstreamTaskId = producer.kernelId;
      input.upstreamOutputName = producer.outputs[selectedOutput].name;
    }
  }

  auto describeBindingAssignment =
      [](const ArtifactManifestBindingPath &assignment) {
        if (assignment.taskId.empty())
          return assignment.bindingName;
        return (llvm::Twine(assignment.taskId) + "." + assignment.bindingName)
            .str();
      };

  auto collectBindingMatches =
      [&](const ArtifactManifestBindingPath &assignment, bool inputs) {
        std::vector<std::pair<size_t, size_t>> matches;
        for (auto [kernelIndexValue, kernel] : llvm::enumerate(kernels)) {
          if (!assignment.taskId.empty() &&
              kernel.kernelId != assignment.taskId)
            continue;
          llvm::ArrayRef<TensorBinding> bindings =
              inputs ? llvm::ArrayRef<TensorBinding>(kernel.inputs)
                     : llvm::ArrayRef<TensorBinding>(kernel.outputs);
          for (auto [bindingIndex, binding] : llvm::enumerate(bindings)) {
            if (binding.name != assignment.bindingName)
              continue;
            if (inputs &&
                binding.sourceKind != BindingSourceKind::ExternalFile)
              continue;
            matches.emplace_back(kernelIndexValue, bindingIndex);
          }
        }
        return matches;
      };

  auto applyBindingPathAssignments =
      [&](llvm::ArrayRef<ArtifactManifestBindingPath> assignments,
          bool inputs, llvm::StringRef optionName) -> llvm::Error {
    for (const ArtifactManifestBindingPath &assignment : assignments) {
      std::vector<std::pair<size_t, size_t>> matches =
          collectBindingMatches(assignment, inputs);
      const std::string selector = describeBindingAssignment(assignment);
      if (matches.empty()) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "artifact manifest prepare --%s references unknown %s binding: %s",
            optionName.str().c_str(),
            inputs ? "external input" : "output", selector.c_str());
      }
      if (matches.size() > 1) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "artifact manifest prepare --%s binding is ambiguous; use "
            "task.binding selector: %s",
            optionName.str().c_str(), selector.c_str());
      }
      auto [kernelIndexValue, bindingIndex] = matches.front();
      TensorBinding &binding =
          inputs ? kernels[kernelIndexValue].inputs[bindingIndex]
                 : kernels[kernelIndexValue].outputs[bindingIndex];
      binding.path = assignment.path;
      if (assignment.shape)
        binding.shape = *assignment.shape;
      if (assignment.dtype)
        binding.dtype = *assignment.dtype;
    }
    return llvm::Error::success();
  };

  if (auto err =
          applyBindingPathAssignments(request.inputPaths, true, "input"))
    return err;
  if (auto err =
          applyBindingPathAssignments(request.outputPaths, false, "output"))
    return err;

  for (const ArtifactManifestBindingPath &assignment :
       request.expectedOutputPaths) {
    std::vector<std::pair<size_t, size_t>> matches =
        collectBindingMatches(assignment, /*inputs=*/false);
    const std::string selector = describeBindingAssignment(assignment);
    if (matches.empty()) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "artifact manifest prepare --expected-output references unknown "
          "output binding: %s",
          selector.c_str());
    }
    if (matches.size() > 1) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "artifact manifest prepare --expected-output binding is ambiguous; "
          "use task.binding selector: %s",
          selector.c_str());
    }

    auto [kernelIndexValue, bindingIndex] = matches.front();
    const TensorBinding &output = kernels[kernelIndexValue].outputs[bindingIndex];
    auto &expectedOutputs = kernels[kernelIndexValue].expectedOutputs;
    auto duplicate = std::find_if(
        expectedOutputs.begin(), expectedOutputs.end(),
        [&](const TensorBinding &binding) { return binding.name == output.name; });
    if (duplicate != expectedOutputs.end()) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "artifact manifest prepare --expected-output repeats binding: %s",
          selector.c_str());
    }

    TensorBinding expected;
    expected.name = output.name;
    expected.path = assignment.path;
    expected.shape = output.shape;
    expected.dtype = output.dtype;
    if (assignment.shape)
      expected.shape = *assignment.shape;
    if (assignment.dtype)
      expected.dtype = *assignment.dtype;
    expectedOutputs.push_back(std::move(expected));
  }

  if (!request.defaultOutputDirectory.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(
            request.defaultOutputDirectory)) {
      return llvm::createStringError(
          ec, "cannot create default output directory: %s",
          request.defaultOutputDirectory.c_str());
    }
    for (PreparedKernel &kernel : kernels) {
      for (TensorBinding &output : kernel.outputs) {
        if (output.sourceKind == BindingSourceKind::ExternalFile &&
            output.path.empty()) {
          output.path = makeDefaultOutputPath(request.defaultOutputDirectory,
                                             kernel.kernelId, output.name);
        }
      }
    }
  }

  llvm::json::Object runManifest;
  runManifest["backend"] = stringifyBackendKind(request.backendKind).str();
  runManifest["artifact_root"] = request.artifactRoot;
  llvm::json::Array tasks;
  for (const PreparedKernel &kernel : kernels) {
    llvm::json::Object task;
    task["task_id"] = kernel.kernelId;
    task["artifact_root"] = kernel.artifactRoot;
    auto depsIt = dependencies.find(kernel.kernelId);
    if (depsIt != dependencies.end()) {
      std::vector<std::string> deps(depsIt->second.begin(),
                                    depsIt->second.end());
      task["dependencies"] = toJsonStringArray(deps);
    }
    task["inputs"] = toJsonTensorBindings(kernel.inputs);
    task["outputs"] = toJsonTensorBindings(kernel.outputs);
    if (!kernel.expectedOutputs.empty())
      task["expected_outputs"] = toJsonTensorBindings(kernel.expectedOutputs);
    if (request.enableProfiling)
      task["profiling"] = *request.enableProfiling;
    if (request.atol)
      task["atol"] = *request.atol;
    if (request.rtol)
      task["rtol"] = *request.rtol;
    if (kernel.hasTiling) {
      llvm::json::Object tiling;
      if (!kernel.tiling.schemaPath.empty())
        tiling["schema"] = kernel.tiling.schemaPath;
      if (!kernel.tiling.params.empty())
        tiling["params"] = kernel.tiling.params;
      if (!kernel.tiling.binaryPath.empty())
        tiling["binary"] = kernel.tiling.binaryPath;
      task["tiling"] = std::move(tiling);
    }
    task["block_dim"] = kernel.blockDim;
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
