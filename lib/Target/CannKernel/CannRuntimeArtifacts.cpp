//===- CannRuntimeArtifacts.cpp - CANN runtime artifact emission ----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/CannKernel/CannRuntimeArtifacts.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <functional>

using namespace mlir;

namespace mlir::afir::cann {
namespace {

struct TilingFieldInfo {
  std::string name;
  std::string type;
  bool isShape = false;
  std::string shapeKey;
};

struct WorkspaceInfo {
  int64_t sizeBytes = 0;
  std::string sizeExpr = "0";
};

static constexpr llvm::StringLiteral kKernelMetadataKernelKey = "kernel";
static constexpr llvm::StringLiteral kKernelMetadataDecisionIdKey =
    "decision_id";
static constexpr llvm::StringLiteral kKernelMetadataSelectedTileShapeKey =
    "selected_tile_shape";
static constexpr llvm::StringLiteral kKernelMetadataTailPoliciesKey =
    "tail_policies";
static constexpr llvm::StringLiteral kKernelMetadataTailPlanKey =
    "tail_plan";

static SmallVector<func::FuncOp> collectGlobalKernels(ModuleOp module) {
  SmallVector<func::FuncOp> kernels;
  for (Operation &child : module.getBody()->getOperations())
    if (auto funcOp = dyn_cast<func::FuncOp>(child))
      if (funcOp->hasAttr(ascendc::attr::global))
        kernels.push_back(funcOp);
  return kernels;
}

static FailureOr<func::FuncOp> getPrimaryGlobalKernel(ModuleOp module,
                                                      StringRef artifactName) {
  SmallVector<func::FuncOp> kernels = collectGlobalKernels(module);
  if (kernels.empty()) {
    module.emitError() << artifactName << " requires at least one global kernel";
    return failure();
  }
  return kernels.front();
}

static FailureOr<emitasc::PyStructType> getTilingType(func::FuncOp funcOp) {
  if (funcOp.getNumArguments() < 2) {
    funcOp.emitError("CANN ABI requires workspace and tiling arguments");
    return failure();
  }
  Type tilingType = funcOp.getArgument(funcOp.getNumArguments() - 1).getType();
  auto pyStruct = dyn_cast<emitasc::PyStructType>(tilingType);
  if (!pyStruct) {
    funcOp.emitError("last CANN ABI argument must be !emitasc.py_struct");
    return failure();
  }
  return pyStruct;
}

static bool isShapeField(StringRef name) {
  return name.starts_with("dim_arg");
}

static std::string makeShapeKey(StringRef name) {
  StringRef rest = name.drop_front(4);
  auto pos = rest.rfind('_');
  if (pos == StringRef::npos)
    return rest.str();
  return rest.substr(0, pos).str() + "_dim" + rest.substr(pos + 1).str();
}

static std::string getHostCppType(Type type) {
  if (type.isInteger(64))
    return "int64_t";
  if (type.isInteger(32))
    return "int32_t";
  if (type.isInteger(16))
    return "int16_t";
  if (type.isInteger(8))
    return "int8_t";
  return "int64_t";
}

static std::string sanitizeCppIdentifier(StringRef value) {
  std::string result;
  result.reserve(value.size() + 1);
  for (char c : value) {
    unsigned char ch = static_cast<unsigned char>(c);
    if (std::isalnum(ch) || c == '_')
      result.push_back(c);
    else
      result.push_back('_');
  }
  if (result.empty() ||
      std::isdigit(static_cast<unsigned char>(result.front())))
    result.insert(result.begin(), '_');
  return result;
}

static std::string getJsonType(Type type) {
  if (type.isInteger(64))
    return "int64";
  if (type.isInteger(32))
    return "int32";
  if (type.isInteger(16))
    return "int16";
  if (type.isInteger(8))
    return "int8";
  return "int64";
}

static FailureOr<SmallVector<TilingFieldInfo>>
collectTilingFields(func::FuncOp funcOp, emitasc::PyStructType tilingType) {
  auto types = tilingType.getTypesAttr().getValue();
  auto names = tilingType.getNamesAttr().getValue();
  if (types.size() != names.size()) {
    funcOp.emitError("PyStructType types/names size mismatch");
    return failure();
  }

  SmallVector<TilingFieldInfo> fields;
  fields.reserve(names.size());
  for (auto [typeAttr, nameAttr] : llvm::zip(types, names)) {
    StringRef name = cast<StringAttr>(nameAttr).getValue();
    Type type = cast<TypeAttr>(typeAttr).getValue();
    TilingFieldInfo field;
    field.name = name.str();
    field.type = getJsonType(type);
    field.isShape = isShapeField(name);
    if (field.isShape)
      field.shapeKey = makeShapeKey(name);
    fields.push_back(std::move(field));
  }
  return fields;
}

static llvm::json::Array buildTilingSchema(ArrayRef<TilingFieldInfo> fields) {
  llvm::json::Array schema;
  for (const TilingFieldInfo &field : fields) {
    llvm::json::Object entry;
    entry["name"] = field.name;
    entry["type"] = field.type;
    entry["fixed"] = field.isShape;
    if (field.isShape)
      entry["shape_key"] = field.shapeKey;
    else
      entry["values"] = llvm::json::Array{};
    schema.push_back(std::move(entry));
  }
  return schema;
}

static llvm::json::Array buildShapeArgOrder(ArrayRef<TilingFieldInfo> fields) {
  llvm::json::Array shapeArgOrder;
  int64_t shapeAbiPosition = 0;
  for (const TilingFieldInfo &field : fields) {
    if (!field.isShape)
      continue;
    llvm::json::Object shapeArg;
    shapeArg["name"] = field.name;
    shapeArg["shapeKey"] = field.shapeKey;
    shapeArg["abiPosition"] = shapeAbiPosition++;
    shapeArgOrder.push_back(std::move(shapeArg));
  }
  return shapeArgOrder;
}

static llvm::json::Object buildShapeDescriptor(ArrayRef<TilingFieldInfo> fields) {
  llvm::json::Array dynamicDims;
  for (const TilingFieldInfo &field : fields) {
    if (!field.isShape)
      continue;
    llvm::json::Object dim;
    dim["shapeKey"] = field.shapeKey;
    dim["tilingField"] = field.name;
    dynamicDims.push_back(std::move(dim));
  }

  llvm::json::Object shape;
  shape["rank"] = static_cast<int64_t>(dynamicDims.size());
  shape["dynamicDims"] = std::move(dynamicDims);
  shape["shapeArgOrder"] = buildShapeArgOrder(fields);
  return shape;
}

static FailureOr<WorkspaceInfo> getWorkspaceInfo(func::FuncOp funcOp) {
  WorkspaceInfo info;
  auto sizeAttr = funcOp->getAttrOfType<IntegerAttr>(
      ::mlir::afir::ascend::kCannWorkspaceSizeBytesAttr);
  if (sizeAttr) {
    int64_t sizeBytes = sizeAttr.getInt();
    if (sizeBytes < 0)
      return funcOp.emitError()
             << ::mlir::afir::ascend::kCannWorkspaceSizeBytesAttr
             << " must be a non-negative integer attribute";
    info.sizeBytes = sizeBytes;
    info.sizeExpr = std::to_string(sizeBytes);
  }

  auto exprAttr = funcOp->getAttrOfType<StringAttr>(
      ::mlir::afir::ascend::kCannWorkspaceSizeExprAttr);
  if (exprAttr) {
    if (exprAttr.getValue().trim().empty())
      return funcOp.emitError()
             << ::mlir::afir::ascend::kCannWorkspaceSizeExprAttr
             << " must not be empty";
    info.sizeExpr = exprAttr.getValue().str();
  }
  return info;
}

static llvm::json::Object buildWorkspaceDescriptor(func::FuncOp funcOp,
                                                   const WorkspaceInfo &info) {
  llvm::json::Object workspace;
  workspace["mode"] = "fixed";
  workspace["argIndex"] =
      static_cast<int64_t>(funcOp.getNumArguments() >= 2
                               ? funcOp.getNumArguments() - 2
                               : 0);
  workspace["sizeExpr"] = info.sizeExpr;
  workspace["sizeBytes"] = info.sizeBytes;
  return workspace;
}

static int64_t getCannWorkspaceArgIndex(func::FuncOp funcOp) {
  return static_cast<int64_t>(funcOp.getNumArguments() >= 2
                                  ? funcOp.getNumArguments() - 2
                                  : 0);
}

static std::string getKernelKindString(func::FuncOp funcOp) {
  auto kindAttr = funcOp->getAttrOfType<StringAttr>(
      ::mlir::afir::ascend::kAscendCKernelKindAttr);
  if (!kindAttr)
    return "unknown";
  StringRef kind = kindAttr.getValue();
  if (kind == ::mlir::afir::ascend::kAscendCKernelKindVec ||
      kind == ::mlir::afir::ascend::kAscendCKernelKindCube ||
      kind == ::mlir::afir::ascend::kAscendCKernelKindMix)
    return kind.str();
  return "unknown";
}

static std::string getHostTilingBindingId(func::FuncOp funcOp) {
  return (funcOp.getName() + ".host_tiling").str();
}

static llvm::json::Object buildHostTilingBinding(func::FuncOp funcOp) {
  StringRef kernelName = funcOp.getName();
  llvm::json::Object symbols;
  symbols["getTilingSize"] =
      (llvm::Twine(kernelName) + "_GetTilingSize").str();
  symbols["getTiling"] = (llvm::Twine(kernelName) + "_GetTiling").str();
  symbols["getBlockDim"] = (llvm::Twine(kernelName) + "_GetBlockDim").str();
  symbols["getWorkspaceSize"] =
      (llvm::Twine(kernelName) + "_GetWorkspaceSize").str();

  llvm::json::Object binding;
  binding["id"] = getHostTilingBindingId(funcOp);
  binding["library"] = "host_tiling.so";
  binding["symbols"] = std::move(symbols);
  return binding;
}

static llvm::json::Array
buildHostTilingBindings(ArrayRef<func::FuncOp> kernels) {
  llvm::json::Array bindings;
  for (func::FuncOp kernel : kernels)
    bindings.push_back(buildHostTilingBinding(kernel));
  return bindings;
}

static Value stripViewLike(Value value) {
  while (true) {
    if (auto castOp = value.getDefiningOp<memref::CastOp>()) {
      value = castOp.getSource();
      continue;
    }
    if (auto subviewOp = value.getDefiningOp<memref::SubViewOp>()) {
      value = subviewOp.getSource();
      continue;
    }
    if (auto collapseOp = value.getDefiningOp<memref::CollapseShapeOp>()) {
      value = collapseOp.getSrc();
      continue;
    }
    if (auto expandOp = value.getDefiningOp<memref::ExpandShapeOp>()) {
      value = expandOp.getSrc();
      continue;
    }
    if (auto reinterpretOp =
            value.getDefiningOp<memref::ReinterpretCastOp>()) {
      value = reinterpretOp.getSource();
      continue;
    }
    return value;
  }
}

static llvm::json::Array buildWritesToInputArgs(func::FuncOp funcOp,
                                                int64_t numInputs) {
  llvm::SmallSetVector<int64_t, 8> writtenInputArgs;
  if (funcOp.getBody().empty())
    return llvm::json::Array{};

  Block &entryBlock = funcOp.getBody().front();

  auto recordWriteTo = [&](Value memref) {
    Value root = stripViewLike(memref);
    auto blockArg = dyn_cast<BlockArgument>(root);
    if (!blockArg || blockArg.getOwner() != &entryBlock)
      return;
    int64_t argIndex = static_cast<int64_t>(blockArg.getArgNumber());
    if (argIndex < numInputs)
      writtenInputArgs.insert(argIndex);
  };

  funcOp.walk([&](memref::StoreOp storeOp) {
    recordWriteTo(storeOp.getMemref());
  });
  funcOp.walk([&](emitasc::CallOpaqueOp callOp) {
    if (!callOp.getCallee().starts_with("afir_gm_store<"))
      return;
    ValueRange operands = callOp.getCalleeOperands();
    if (operands.empty())
      return;
    recordWriteTo(operands.front());
  });

  llvm::json::Array writes;
  for (int64_t argIndex : writtenInputArgs)
    writes.push_back(argIndex);
  return writes;
}

static FailureOr<std::string> getRuntimeDTypeName(Type elementType,
                                                  func::FuncOp funcOp) {
  if (elementType.isF16())
    return std::string("f16");
  if (elementType.isBF16())
    return std::string("bf16");
  if (elementType.isF32())
    return std::string("f32");
  if (elementType.isInteger(8))
    return std::string("int8");
  if (elementType.isInteger(32))
    return std::string("int32");
  if (elementType.isInteger(64))
    return std::string("int64");
  return funcOp.emitError() << "unsupported runtime ABI tensor element type: "
                            << elementType;
}

static llvm::json::Array buildRuntimeShape(MemRefType memrefType) {
  llvm::json::Array shape;
  for (int64_t dim : memrefType.getShape())
    shape.push_back(ShapedType::isDynamic(dim) ? -1 : dim);
  return shape;
}

static FailureOr<llvm::json::Object>
buildRuntimeTensorDescriptor(func::FuncOp funcOp, int64_t argIndex,
                             StringRef prefix, int64_t logicalIndex) {
  auto memrefType = dyn_cast<MemRefType>(funcOp.getArgument(argIndex).getType());
  if (!memrefType)
    return funcOp.emitError() << "runtime ABI tensor argument " << argIndex
                              << " must be a memref";
  FailureOr<std::string> dtype =
      getRuntimeDTypeName(memrefType.getElementType(), funcOp);
  if (failed(dtype))
    return failure();

  llvm::json::Object descriptor;
  descriptor["name"] = (prefix + llvm::Twine(logicalIndex)).str();
  descriptor["shape"] = buildRuntimeShape(memrefType);
  descriptor["dtype"] = *dtype;
  return descriptor;
}

static FailureOr<llvm::json::Array>
buildRuntimeTensorDescriptors(func::FuncOp funcOp, int64_t begin, int64_t end,
                              StringRef prefix) {
  llvm::json::Array descriptors;
  for (int64_t argIndex = begin; argIndex < end; ++argIndex) {
    FailureOr<llvm::json::Object> descriptor =
        buildRuntimeTensorDescriptor(funcOp, argIndex, prefix,
                                     argIndex - begin);
    if (failed(descriptor))
      return failure();
    descriptors.push_back(std::move(*descriptor));
  }
  return descriptors;
}

static FailureOr<llvm::json::Object>
buildAbiDescriptor(func::FuncOp funcOp) {
  auto numInputsAttr = funcOp->getAttrOfType<IntegerAttr>("cann.num_inputs");
  if (!numInputsAttr)
    return funcOp.emitError()
           << "runtime manifest ABI requires cann.num_inputs";

  int64_t numInputs = numInputsAttr.getInt();
  if (numInputs < 0)
    return funcOp.emitError()
           << "cann.num_inputs must be a non-negative integer";

  int64_t workspaceArgIndex = getCannWorkspaceArgIndex(funcOp);
  if (workspaceArgIndex < numInputs)
    return funcOp.emitError()
           << "CANN ABI workspace argument index " << workspaceArgIndex
           << " is before cann.num_inputs " << numInputs;

  llvm::json::Object abi;
  abi["numInputs"] = numInputs;
  abi["numOutputs"] = workspaceArgIndex - numInputs;
  abi["workspaceArgIndex"] = workspaceArgIndex;
  abi["writesToInputArgs"] = buildWritesToInputArgs(funcOp, numInputs);
  FailureOr<llvm::json::Array> inputs =
      buildRuntimeTensorDescriptors(funcOp, 0, numInputs, "arg");
  if (failed(inputs))
    return failure();
  FailureOr<llvm::json::Array> outputs = buildRuntimeTensorDescriptors(
      funcOp, numInputs, workspaceArgIndex, "out");
  if (failed(outputs))
    return failure();
  abi["inputs"] = std::move(*inputs);
  abi["outputs"] = std::move(*outputs);
  return abi;
}

static FailureOr<std::string>
buildHostWorkspaceSizeExpr(func::FuncOp funcOp, StringRef expr,
                           ArrayRef<TilingFieldInfo> fields,
                           bool &usesShapeArgs) {
  llvm::StringMap<unsigned> shapeFieldAbiPositions;
  unsigned shapeIndex = 0;
  for (const TilingFieldInfo &field : fields) {
    if (!field.isShape)
      continue;
    shapeFieldAbiPositions[field.name] = shapeIndex++;
  }

  std::string hostExpr;
  for (size_t i = 0, e = expr.size(); i < e;) {
    unsigned char ch = static_cast<unsigned char>(expr[i]);
    if (std::isalpha(ch) || expr[i] == '_') {
      size_t start = i++;
      while (i < e) {
        unsigned char identCh = static_cast<unsigned char>(expr[i]);
        if (!std::isalnum(identCh) && expr[i] != '_')
          break;
        ++i;
      }
      StringRef ident = expr.slice(start, i);
      auto it = shapeFieldAbiPositions.find(ident);
      if (it == shapeFieldAbiPositions.end())
        return funcOp.emitError()
               << ::mlir::afir::ascend::kCannWorkspaceSizeExprAttr
               << " references unknown tiling shape field \"" << ident
               << "\"";
      usesShapeArgs = true;
      hostExpr += "shape_args[" + std::to_string(it->second) + "]";
      continue;
    }

    if (std::isdigit(ch) || std::isspace(ch) || expr[i] == '+' ||
        expr[i] == '-' || expr[i] == '*' || expr[i] == '/' ||
        expr[i] == '%' || expr[i] == '(' || expr[i] == ')') {
      hostExpr.push_back(expr[i++]);
      continue;
    }

    return funcOp.emitError()
           << ::mlir::afir::ascend::kCannWorkspaceSizeExprAttr
           << " contains unsupported character '" << expr[i] << "'";
  }

  return hostExpr;
}

static llvm::json::Object buildResourceDescriptor(func::FuncOp funcOp) {
  llvm::json::Array memorySpaces;
  for (BlockArgument arg : funcOp.getArguments()) {
    auto memrefType = dyn_cast<MemRefType>(arg.getType());
    if (!memrefType)
      continue;
    llvm::json::Object memorySpace;
    memorySpace["argIndex"] = static_cast<int64_t>(arg.getArgNumber());
    if (auto intAttr =
            dyn_cast_or_null<IntegerAttr>(memrefType.getMemorySpace()))
      memorySpace["memorySpace"] = intAttr.getInt();
    else
      memorySpace["memorySpace"] = 0;
    memorySpaces.push_back(std::move(memorySpace));
  }

  llvm::json::Object resources;
  resources["executionUnit"] =
      funcOp->hasAttr(ascendc::attr::aicore) ? "aicore" : "host";
  resources["kernelKind"] = getKernelKindString(funcOp);
  resources["memorySpaces"] = std::move(memorySpaces);
  resources["mixResourceType"] = "unknown";
  return resources;
}

static bool isSupportedTailPolicy(StringRef value) {
  return llvm::StringSwitch<bool>(value)
      .Case("must_divide", true)
      .Case("masked_tail", true)
      .Case("scalar_epilogue", true)
      .Case("pad_and_mask", true)
      .Case("full_extent", true)
      .Default(false);
}

static bool isSupportedTailBufferingMode(StringRef value) {
  return llvm::StringSwitch<bool>(value)
      .Case("separate_tail_buffer", true)
      .Case("reuse_main_buffer_after_drain", true)
      .Default(false);
}

static bool isSupportedAffectedPrimitiveUse(StringRef value) {
  return llvm::StringSwitch<bool>(value)
      .Case("data_copy", true)
      .Case("vector_compute", true)
      .Case("reduction", true)
      .Case("gather_index", true)
      .Case("cube_m", true)
      .Case("cube_n", true)
      .Case("cube_k", true)
      .Case("write_back", true)
      .Default(false);
}

static FailureOr<DictionaryAttr>
lookupKernelScheduleMetadata(func::FuncOp funcOp) {
  auto metadata =
      funcOp->getAttrOfType<ArrayAttr>(
          ::mlir::afir::ascend::kScheduleKernelMetadataAttr);
  if (!metadata)
    return DictionaryAttr();

  DictionaryAttr fallbackEntry;
  for (auto [index, rawEntry] : llvm::enumerate(metadata)) {
    auto entry = dyn_cast<DictionaryAttr>(rawEntry);
    if (!entry)
      return funcOp.emitError()
             << ::mlir::afir::ascend::kScheduleKernelMetadataAttr
             << " element " << index << " must be a dictionary attribute";

    auto kernel =
        dyn_cast_or_null<StringAttr>(entry.get(kKernelMetadataKernelKey));
    if (!kernel)
      return funcOp.emitError()
             << ::mlir::afir::ascend::kScheduleKernelMetadataAttr
             << " element " << index
             << " entries must include a string kernel field";

    if (kernel.getValue() == funcOp.getName())
      return entry;
    if (!fallbackEntry && metadata.size() == 1)
      fallbackEntry = entry;
  }

  return fallbackEntry;
}

static Attribute getScheduleMetadataAttr(func::FuncOp funcOp,
                                         DictionaryAttr kernelMetadata,
                                         StringRef attrName,
                                         StringRef kernelMetadataKey) {
  if (Attribute attr = funcOp->getAttr(attrName))
    return attr;
  return kernelMetadata ? kernelMetadata.get(kernelMetadataKey) : Attribute();
}

static LogicalResult checkScheduleMetadataCompleteness(func::FuncOp funcOp) {
  FailureOr<DictionaryAttr> kernelMetadata =
      lookupKernelScheduleMetadata(funcOp);
  if (failed(kernelMetadata))
    return failure();

  bool hasSelectedTileShape =
      static_cast<bool>(getScheduleMetadataAttr(
          funcOp, *kernelMetadata,
          ::mlir::afir::ascend::kScheduleSelectedTileShapeAttr,
          kKernelMetadataSelectedTileShapeKey));
  bool hasTailPolicies =
      static_cast<bool>(getScheduleMetadataAttr(
          funcOp, *kernelMetadata,
          ::mlir::afir::ascend::kScheduleTailPoliciesAttr,
          kKernelMetadataTailPoliciesKey));
  bool hasTailPlan = static_cast<bool>(getScheduleMetadataAttr(
      funcOp, *kernelMetadata, ::mlir::afir::ascend::kScheduleTailPlanAttr,
      kKernelMetadataTailPlanKey));
  bool hasAnyMetadata = hasSelectedTileShape || hasTailPolicies || hasTailPlan;
  bool hasAllMetadata = hasSelectedTileShape && hasTailPolicies && hasTailPlan;
  if (!hasAnyMetadata || hasAllMetadata)
    return success();

  return funcOp.emitError()
         << "schedule metadata requires "
         << ::mlir::afir::ascend::kScheduleSelectedTileShapeAttr << ", "
         << ::mlir::afir::ascend::kScheduleTailPoliciesAttr << ", and "
         << ::mlir::afir::ascend::kScheduleTailPlanAttr << " together";
}

static LogicalResult validateScheduleMetadataAttributes(func::FuncOp funcOp) {
  FailureOr<DictionaryAttr> kernelMetadata =
      lookupKernelScheduleMetadata(funcOp);
  if (failed(kernelMetadata))
    return failure();

  if (Attribute selectedTileShape = getScheduleMetadataAttr(
          funcOp, *kernelMetadata,
          ::mlir::afir::ascend::kScheduleSelectedTileShapeAttr,
          kKernelMetadataSelectedTileShapeKey))
    if (!isa<DenseI64ArrayAttr>(selectedTileShape))
      return funcOp.emitError()
             << ::mlir::afir::ascend::kScheduleSelectedTileShapeAttr
             << " must be a dense i64 array attribute";

  if (Attribute tailPolicies = getScheduleMetadataAttr(
          funcOp, *kernelMetadata,
          ::mlir::afir::ascend::kScheduleTailPoliciesAttr,
          kKernelMetadataTailPoliciesKey))
    if (!isa<ArrayAttr>(tailPolicies))
      return funcOp.emitError()
             << ::mlir::afir::ascend::kScheduleTailPoliciesAttr
             << " must be an array attribute";

  return success();
}

static FailureOr<SmallVector<int64_t>>
collectSelectedTileShape(func::FuncOp funcOp) {
  FailureOr<DictionaryAttr> kernelMetadata =
      lookupKernelScheduleMetadata(funcOp);
  if (failed(kernelMetadata))
    return failure();

  auto selectedTileShape = dyn_cast_or_null<DenseI64ArrayAttr>(
      getScheduleMetadataAttr(
          funcOp, *kernelMetadata,
          ::mlir::afir::ascend::kScheduleSelectedTileShapeAttr,
          kKernelMetadataSelectedTileShapeKey));
  if (!selectedTileShape)
    return SmallVector<int64_t>{};

  SmallVector<int64_t> values;
  values.append(selectedTileShape.asArrayRef().begin(),
                selectedTileShape.asArrayRef().end());
  return values;
}

static FailureOr<llvm::json::Object>
buildScheduleTilingParams(func::FuncOp funcOp) {
  llvm::json::Object tilingParams;
  FailureOr<DictionaryAttr> kernelMetadata =
      lookupKernelScheduleMetadata(funcOp);
  if (failed(kernelMetadata))
    return failure();

  if (failed(checkScheduleMetadataCompleteness(funcOp)))
    return failure();
  if (failed(validateScheduleMetadataAttributes(funcOp)))
    return failure();

  if (auto selectedTileShape = dyn_cast_or_null<DenseI64ArrayAttr>(
          getScheduleMetadataAttr(
              funcOp, *kernelMetadata,
              ::mlir::afir::ascend::kScheduleSelectedTileShapeAttr,
              kKernelMetadataSelectedTileShapeKey))) {
    llvm::json::Array selectedTileShapeJson;
    for (int64_t tileSize : selectedTileShape.asArrayRef())
      selectedTileShapeJson.push_back(tileSize);
    tilingParams["selected_tile_shape"] = std::move(selectedTileShapeJson);
  }

  if (auto tailPolicies = dyn_cast_or_null<ArrayAttr>(
          getScheduleMetadataAttr(
              funcOp, *kernelMetadata,
              ::mlir::afir::ascend::kScheduleTailPoliciesAttr,
              kKernelMetadataTailPoliciesKey))) {
    llvm::json::Array tailPoliciesJson;
    for (auto [index, tailPolicyAttr] : llvm::enumerate(tailPolicies)) {
      auto tailPolicy = dyn_cast<StringAttr>(tailPolicyAttr);
      if (!tailPolicy)
        return funcOp.emitError()
               << ::mlir::afir::ascend::kScheduleTailPoliciesAttr
               << " element " << index << " must be a string attribute";
      if (!isSupportedTailPolicy(tailPolicy.getValue()))
        return funcOp.emitError()
               << ::mlir::afir::ascend::kScheduleTailPoliciesAttr
               << " element " << index << " has unsupported value '"
               << tailPolicy.getValue()
               << "'; expected one of must_divide, masked_tail, "
                  "scalar_epilogue, pad_and_mask, full_extent";
      tailPoliciesJson.push_back(tailPolicy.getValue().str());
    }
    tilingParams["tail_policies"] = std::move(tailPoliciesJson);
  }

  if (Attribute rawTailPlanAttr = getScheduleMetadataAttr(
          funcOp, *kernelMetadata,
          ::mlir::afir::ascend::kScheduleTailPlanAttr,
          kKernelMetadataTailPlanKey)) {
    auto tailPlanAttr = dyn_cast<ArrayAttr>(rawTailPlanAttr);
    if (!tailPlanAttr)
      return funcOp.emitError()
             << ::mlir::afir::ascend::kScheduleTailPlanAttr
             << " must be an array attribute";

    llvm::json::Array tailPlanJson;
    for (auto [index, tailPlanEntryAttr] : llvm::enumerate(tailPlanAttr)) {
      auto tailPlanEntry = dyn_cast<DictionaryAttr>(tailPlanEntryAttr);
      if (!tailPlanEntry)
        return funcOp.emitError()
               << ::mlir::afir::ascend::kScheduleTailPlanAttr
               << " element " << index
               << " must be a dictionary attribute";

      auto axis = dyn_cast_or_null<IntegerAttr>(tailPlanEntry.get("axis"));
      if (!axis || !axis.getType().isInteger(64))
        return funcOp.emitError()
               << ::mlir::afir::ascend::kScheduleTailPlanAttr
               << " element " << index
               << " field 'axis' must be an i64 integer attribute";
      auto selected =
          dyn_cast_or_null<StringAttr>(tailPlanEntry.get("selected"));
      if (!selected)
        return funcOp.emitError()
               << ::mlir::afir::ascend::kScheduleTailPlanAttr
               << " element " << index
               << " field 'selected' must be a string attribute";
      if (!isSupportedTailPolicy(selected.getValue()))
        return funcOp.emitError()
               << ::mlir::afir::ascend::kScheduleTailPlanAttr
               << " element " << index
               << " field 'selected' has unsupported value '"
               << selected.getValue()
               << "'; expected one of must_divide, masked_tail, "
                  "scalar_epilogue, pad_and_mask, full_extent";
      auto affected =
          dyn_cast_or_null<ArrayAttr>(tailPlanEntry.get("affected"));
      if (!affected)
        return funcOp.emitError()
               << ::mlir::afir::ascend::kScheduleTailPlanAttr
               << " element " << index
               << " field 'affected' must be an array attribute";
      auto align = dyn_cast_or_null<IntegerAttr>(tailPlanEntry.get("align"));
      if (!align || !align.getType().isInteger(64))
        return funcOp.emitError()
               << ::mlir::afir::ascend::kScheduleTailPlanAttr
               << " element " << index
               << " field 'align' must be an i64 integer attribute";
      auto buffering =
          dyn_cast_or_null<StringAttr>(tailPlanEntry.get("buffering"));
      if (!buffering)
        return funcOp.emitError()
               << ::mlir::afir::ascend::kScheduleTailPlanAttr
               << " element " << index
               << " field 'buffering' must be a string attribute";
      if (!isSupportedTailBufferingMode(buffering.getValue()))
        return funcOp.emitError()
               << ::mlir::afir::ascend::kScheduleTailPlanAttr
               << " element " << index
               << " field 'buffering' has unsupported value '"
               << buffering.getValue()
               << "'; expected one of separate_tail_buffer, "
                  "reuse_main_buffer_after_drain";

      llvm::json::Array affectedPrimitiveUsesJson;
      for (auto [affectedIndex, affectedAttr] : llvm::enumerate(affected)) {
        auto affectedUse = dyn_cast<StringAttr>(affectedAttr);
        if (!affectedUse)
          return funcOp.emitError()
                 << ::mlir::afir::ascend::kScheduleTailPlanAttr
                 << " element " << index << " field 'affected' element "
                 << affectedIndex << " must be a string attribute";
        if (!isSupportedAffectedPrimitiveUse(affectedUse.getValue()))
          return funcOp.emitError()
                 << ::mlir::afir::ascend::kScheduleTailPlanAttr
                 << " element " << index << " field 'affected' element "
                 << affectedIndex << " has unsupported value '"
                 << affectedUse.getValue()
                 << "'; expected one of data_copy, vector_compute, reduction, "
                    "gather_index, cube_m, cube_n, cube_k, write_back";
        affectedPrimitiveUsesJson.push_back(affectedUse.getValue().str());
      }

      llvm::json::Object tailPlanObject;
      tailPlanObject["axis"] = axis.getInt();
      tailPlanObject["selectedPolicy"] = selected.getValue().str();
      tailPlanObject["alignmentGranularity"] = align.getInt();
      tailPlanObject["tailBufferingMode"] = buffering.getValue().str();
      tailPlanObject["affectedPrimitiveUses"] =
          std::move(affectedPrimitiveUsesJson);
      tailPlanJson.push_back(std::move(tailPlanObject));
    }
    tilingParams["tail_plan"] = std::move(tailPlanJson);
  }

  return tilingParams;
}

static FailureOr<llvm::json::Array> buildScheduleEntries(func::FuncOp funcOp) {
  FailureOr<llvm::json::Object> tilingParams =
      buildScheduleTilingParams(funcOp);
  if (failed(tilingParams))
    return failure();

  std::string decisionId = "static_0";
  FailureOr<DictionaryAttr> kernelMetadata =
      lookupKernelScheduleMetadata(funcOp);
  if (failed(kernelMetadata))
    return failure();
  if (auto metadataDecisionId =
          dyn_cast_or_null<StringAttr>(
              (*kernelMetadata)
                  ? (*kernelMetadata).get(kKernelMetadataDecisionIdKey)
                  : Attribute()))
    decisionId = metadataDecisionId.getValue().str();

  llvm::json::Object scheduleEntry;
  scheduleEntry["decisionId"] = decisionId;
  scheduleEntry["guard"] = "true";
  scheduleEntry["hostTilingId"] = getHostTilingBindingId(funcOp);
  scheduleEntry["tilingParams"] = std::move(*tilingParams);
  llvm::json::Array scheduleEntries;
  scheduleEntries.push_back(std::move(scheduleEntry));
  return scheduleEntries;
}

static FailureOr<llvm::json::Object>
buildKernelManifestEntry(func::FuncOp funcOp, int64_t entryIndex,
                         StringRef soc) {
  FailureOr<emitasc::PyStructType> tilingTypeOr = getTilingType(funcOp);
  if (failed(tilingTypeOr))
    return failure();
  FailureOr<SmallVector<TilingFieldInfo>> fieldsOr =
      collectTilingFields(funcOp, *tilingTypeOr);
  if (failed(fieldsOr))
    return failure();
  FailureOr<llvm::json::Object> tilingParams =
      buildScheduleTilingParams(funcOp);
  if (failed(tilingParams))
    return failure();
  FailureOr<llvm::json::Array> scheduleEntries =
      buildScheduleEntries(funcOp);
  if (failed(scheduleEntries))
    return failure();
  FailureOr<WorkspaceInfo> workspaceInfo = getWorkspaceInfo(funcOp);
  if (failed(workspaceInfo))
    return failure();

  llvm::json::Object kernelEntry;
  kernelEntry["kernel_id"] = funcOp.getName().str();
  kernelEntry["entry_index"] = entryIndex;
  kernelEntry["shapeBucketKey"] = "static";
  kernelEntry["guardSet"] = llvm::json::Array{};
  kernelEntry["tilingSchema"] = buildTilingSchema(*fieldsOr);
  kernelEntry["scheduleEntries"] = std::move(*scheduleEntries);
  kernelEntry["tilingParams"] = std::move(*tilingParams);
  kernelEntry["abiSignature"] = (funcOp.getName() + ":cann_static").str();
  kernelEntry["cacheKey"] = (funcOp.getName() + ":static:" + soc).str();
  kernelEntry["kernelKind"] = getKernelKindString(funcOp);
  kernelEntry["workspaceSizeExpr"] = workspaceInfo->sizeExpr;
  kernelEntry["workspaceSizeBytes"] = workspaceInfo->sizeBytes;
  kernelEntry["shapeArgOrder"] = buildShapeArgOrder(*fieldsOr);
  kernelEntry["shape"] = buildShapeDescriptor(*fieldsOr);
  kernelEntry["workspace"] = buildWorkspaceDescriptor(funcOp, *workspaceInfo);
  FailureOr<llvm::json::Object> abi = buildAbiDescriptor(funcOp);
  if (failed(abi))
    return failure();
  kernelEntry["abi"] = std::move(*abi);
  kernelEntry["resources"] = buildResourceDescriptor(funcOp);
  return kernelEntry;
}

static LogicalResult detectKernelGraphCycle(ArrayRef<func::FuncOp> kernels,
                                            ArrayRef<std::pair<std::string,
                                                               std::string>>
                                                edges) {
  llvm::StringMap<unsigned> kernelIndex;
  for (auto [index, kernelRef] : llvm::enumerate(kernels)) {
    func::FuncOp kernel = kernelRef;
    kernelIndex[kernel.getName()] = index;
  }

  SmallVector<SmallVector<unsigned>> adjacency(kernels.size());
  for (const auto &edge : edges) {
    unsigned from = kernelIndex.lookup(edge.first);
    unsigned to = kernelIndex.lookup(edge.second);
    adjacency[from].push_back(to);
  }

  enum class VisitState { Unvisited, Visiting, Visited };
  SmallVector<VisitState> states(kernels.size(), VisitState::Unvisited);
  std::function<bool(unsigned)> hasCycle = [&](unsigned node) {
    states[node] = VisitState::Visiting;
    for (unsigned next : adjacency[node]) {
      if (states[next] == VisitState::Visiting)
        return true;
      if (states[next] == VisitState::Unvisited && hasCycle(next))
        return true;
    }
    states[node] = VisitState::Visited;
    return false;
  };

  for (unsigned i = 0, e = kernels.size(); i < e; ++i)
    if (states[i] == VisitState::Unvisited && hasCycle(i))
      return failure();
  return success();
}

static LogicalResult
collectKernelGraphNameAliases(ArrayRef<func::FuncOp> kernels,
                              llvm::StringMap<std::string> &aliases) {
  for (func::FuncOp kernel : kernels) {
    aliases[kernel.getName()] = kernel.getName().str();

    auto metadata =
        kernel->getAttrOfType<ArrayAttr>(
            ::mlir::afir::ascend::kScheduleKernelMetadataAttr);
    if (!metadata)
      continue;

    for (auto [index, rawEntry] : llvm::enumerate(metadata)) {
      auto entry = dyn_cast<DictionaryAttr>(rawEntry);
      if (!entry)
        return kernel.emitError()
               << ::mlir::afir::ascend::kScheduleKernelMetadataAttr
               << " element " << index << " must be a dictionary attribute";

      auto internalKernel =
          dyn_cast_or_null<StringAttr>(entry.get(kKernelMetadataKernelKey));
      if (!internalKernel)
        return kernel.emitError()
               << ::mlir::afir::ascend::kScheduleKernelMetadataAttr
               << " element " << index
               << " entries must include a string kernel field";

      auto existing = aliases.find(internalKernel.getValue());
      if (existing != aliases.end() &&
          StringRef(existing->second) != kernel.getName())
        return kernel.emitError()
               << ::mlir::afir::ascend::kScheduleKernelMetadataAttr
               << " element " << index << " maps internal kernel '"
               << internalKernel.getValue() << "' to both '"
               << existing->second << "' and '" << kernel.getName() << "'";
      aliases[internalKernel.getValue()] = kernel.getName().str();
    }
  }
  return success();
}

static FailureOr<llvm::json::Array>
buildKernelGraphEdges(ModuleOp module, ArrayRef<func::FuncOp> kernels) {
  llvm::StringSet<> kernelNames;
  for (func::FuncOp kernel : kernels)
    kernelNames.insert(kernel.getName());
  llvm::StringMap<std::string> kernelAliases;
  if (failed(collectKernelGraphNameAliases(kernels, kernelAliases)))
    return failure();

  auto edgesAttr = module->getAttrOfType<ArrayAttr>(
      ::mlir::afir::ascend::kKernelGraphEdgesAttr);
  llvm::json::Array edgesJson;
  if (!edgesAttr)
    return edgesJson;

  SmallVector<std::pair<std::string, std::string>> edges;
  for (auto [index, edgeAttr] : llvm::enumerate(edgesAttr)) {
    auto edge = dyn_cast<DictionaryAttr>(edgeAttr);
    if (!edge)
      return module.emitError()
             << ::mlir::afir::ascend::kKernelGraphEdgesAttr << " element "
             << index << " must be a dictionary attribute";

    auto from = dyn_cast_or_null<StringAttr>(edge.get("from"));
    auto to = dyn_cast_or_null<StringAttr>(edge.get("to"));
    if (!from || !to)
      return module.emitError()
             << ::mlir::afir::ascend::kKernelGraphEdgesAttr << " element "
             << index << " requires string 'from' and 'to' fields";

    auto resolveKernelName =
        [&](StringRef rawName) -> std::optional<std::string> {
      auto alias = kernelAliases.find(rawName);
      if (alias == kernelAliases.end())
        return std::nullopt;
      return alias->second;
    };

    bool fromKnown = kernelNames.contains(from.getValue());
    bool toKnown = kernelNames.contains(to.getValue());
    std::optional<std::string> resolvedFrom =
        resolveKernelName(from.getValue());
    std::optional<std::string> resolvedTo = resolveKernelName(to.getValue());
    bool staleMergedSyntheticEdge =
        !resolvedFrom && !resolvedTo && kernels.size() == 1 &&
        from.getValue().starts_with("kernel_") &&
        to.getValue().starts_with("kernel_");
    if (staleMergedSyntheticEdge)
      continue;

    if (!resolvedFrom)
      return module.emitError()
             << ::mlir::afir::ascend::kKernelGraphEdgesAttr << " element "
             << index << " references unknown source kernel '"
             << from.getValue() << "'";
    if (!resolvedTo)
      return module.emitError()
             << ::mlir::afir::ascend::kKernelGraphEdgesAttr << " element "
             << index << " references unknown target kernel '" << to.getValue()
             << "'";

    if (*resolvedFrom == *resolvedTo && (!fromKnown || !toKnown))
      continue;

    auto carriedBuffers =
        dyn_cast_or_null<ArrayAttr>(edge.get("carried_buffers"));
    if (!carriedBuffers || carriedBuffers.empty())
      return module.emitError()
             << ::mlir::afir::ascend::kKernelGraphEdgesAttr << " element "
             << index
             << " requires non-empty array field 'carried_buffers'";

    llvm::json::Array carriedBuffersJson;
    for (auto [bufferIndex, bufferAttr] : llvm::enumerate(carriedBuffers)) {
      auto buffer = dyn_cast<StringAttr>(bufferAttr);
      if (!buffer)
        return module.emitError()
               << ::mlir::afir::ascend::kKernelGraphEdgesAttr << " element "
               << index << " field 'carried_buffers' element " << bufferIndex
               << " must be a string";
      carriedBuffersJson.push_back(buffer.getValue().str());
    }

    llvm::json::Object edgeJson;
    edgeJson["from"] = *resolvedFrom;
    edgeJson["to"] = *resolvedTo;
    edgeJson["carriedBuffers"] = std::move(carriedBuffersJson);
    edgesJson.push_back(std::move(edgeJson));
    edges.push_back({*resolvedFrom, *resolvedTo});
  }

  if (failed(detectKernelGraphCycle(kernels, edges)))
    return module.emitError()
           << ::mlir::afir::ascend::kKernelGraphEdgesAttr
           << " must describe an acyclic kernel graph";
  return edgesJson;
}

static FailureOr<llvm::json::Object>
buildKernelGraph(ModuleOp module, ArrayRef<func::FuncOp> kernels) {
  llvm::json::Array kernelNodes;
  for (auto [index, kernelRef] : llvm::enumerate(kernels)) {
    func::FuncOp kernel = kernelRef;
    llvm::json::Object kernelNode;
    kernelNode["name"] = kernel.getName().str();
    kernelNode["entry_index"] = static_cast<int64_t>(index);
    kernelNodes.push_back(std::move(kernelNode));
  }

  FailureOr<llvm::json::Array> kernelEdges =
      buildKernelGraphEdges(module, kernels);
  if (failed(kernelEdges))
    return failure();

  llvm::json::Object kernelGraph;
  kernelGraph["nodes"] = std::move(kernelNodes);
  kernelGraph["edges"] = std::move(*kernelEdges);
  return kernelGraph;
}

static LogicalResult checkFileError(Operation *diagOp,
                                    llvm::raw_fd_ostream &file,
                                    StringRef outPath, StringRef phase) {
  if (!file.has_error())
    return success();
  std::error_code ec = file.error();
  file.clear_error();
  return diagOp->emitError()
         << "failed to " << phase << " runtime artifact '" << outPath
         << "': " << ec.message();
}

static void closeIgnoringFileError(llvm::raw_fd_ostream &file) {
  file.close();
  if (file.has_error())
    file.clear_error();
}

static LogicalResult writeJsonFile(Operation *diagOp, StringRef outPath,
                                   llvm::json::Object root) {
  std::error_code ec;
  llvm::raw_fd_ostream file(outPath, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return diagOp->emitError()
           << "failed to write runtime artifact '" << outPath
           << "': " << ec.message();

  llvm::json::OStream json(file, /*IndentSize=*/2);
  json.value(llvm::json::Value(std::move(root)));
  file << "\n";
  if (failed(checkFileError(diagOp, file, outPath, "write"))) {
    closeIgnoringFileError(file);
    return failure();
  }
  file.flush();
  if (failed(checkFileError(diagOp, file, outPath, "flush"))) {
    closeIgnoringFileError(file);
    return failure();
  }
  file.close();
  return checkFileError(diagOp, file, outPath, "finalize");
}

static LogicalResult writeTextFile(Operation *diagOp, StringRef outPath,
                                   std::function<void(raw_ostream &)> emit) {
  std::error_code ec;
  llvm::raw_fd_ostream file(outPath, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return diagOp->emitError()
           << "failed to write runtime artifact '" << outPath
           << "': " << ec.message();
  emit(file);
  if (failed(checkFileError(diagOp, file, outPath, "write"))) {
    closeIgnoringFileError(file);
    return failure();
  }
  file.flush();
  if (failed(checkFileError(diagOp, file, outPath, "flush"))) {
    closeIgnoringFileError(file);
    return failure();
  }
  file.close();
  return checkFileError(diagOp, file, outPath, "finalize");
}

} // namespace

LogicalResult emitTilingSpaceJson(ModuleOp module, StringRef outPath,
                                  const CannRuntimeArtifactOptions &options) {
  FailureOr<func::FuncOp> funcOr =
      getPrimaryGlobalKernel(module, "tiling space");
  if (failed(funcOr))
    return failure();
  FailureOr<emitasc::PyStructType> tilingTypeOr = getTilingType(*funcOr);
  if (failed(tilingTypeOr))
    return failure();
  FailureOr<SmallVector<TilingFieldInfo>> fieldsOr =
      collectTilingFields(*funcOr, *tilingTypeOr);
  if (failed(fieldsOr))
    return failure();
  FailureOr<WorkspaceInfo> workspaceInfo = getWorkspaceInfo(*funcOr);
  if (failed(workspaceInfo))
    return failure();

  llvm::json::Object root;
  root["schema_version"] = "2.0";
  root["kernel"] = funcOr->getName().str();
  root["kernel_file"] = options.kernelFile.str();
  root["soc"] = options.soc.str();
  root["block_dim_expr"] = "20";
  root["workspace_size_expr"] = workspaceInfo->sizeExpr;
  root["tiling_params"] = buildTilingSchema(*fieldsOr);
  return writeJsonFile(module.getOperation(), outPath, std::move(root));
}

LogicalResult
emitRuntimeManifestJson(ModuleOp module, StringRef outPath,
                        const CannRuntimeArtifactOptions &options) {
  SmallVector<func::FuncOp> kernels = collectGlobalKernels(module);
  if (kernels.empty()) {
    module.emitError() << "runtime manifest requires at least one global kernel";
    return failure();
  }

  func::FuncOp primaryKernel = kernels.front();
  FailureOr<emitasc::PyStructType> tilingTypeOr =
      getTilingType(primaryKernel);
  if (failed(tilingTypeOr))
    return failure();
  FailureOr<SmallVector<TilingFieldInfo>> fieldsOr =
      collectTilingFields(primaryKernel, *tilingTypeOr);
  if (failed(fieldsOr))
    return failure();

  FailureOr<llvm::json::Object> tilingParams =
      buildScheduleTilingParams(primaryKernel);
  if (failed(tilingParams))
    return failure();
  FailureOr<llvm::json::Array> scheduleEntries =
      buildScheduleEntries(primaryKernel);
  if (failed(scheduleEntries))
    return failure();
  FailureOr<WorkspaceInfo> primaryWorkspaceInfo =
      getWorkspaceInfo(primaryKernel);
  if (failed(primaryWorkspaceInfo))
    return failure();

  llvm::json::Array kernelEntries;
  for (auto [index, kernel] : llvm::enumerate(kernels)) {
    FailureOr<llvm::json::Object> kernelEntry = buildKernelManifestEntry(
        kernel, static_cast<int64_t>(index), options.soc);
    if (failed(kernelEntry))
      return failure();
    kernelEntries.push_back(std::move(*kernelEntry));
  }

  FailureOr<llvm::json::Object> kernelGraph =
      buildKernelGraph(module, kernels);
  if (failed(kernelGraph))
    return failure();

  llvm::json::Object root;
  root["kernelName"] = primaryKernel.getName().str();
  root["shapeBucketKey"] = "static";
  root["guardSet"] = llvm::json::Array{};
  root["tilingSchema"] = buildTilingSchema(*fieldsOr);
  root["scheduleEntries"] = std::move(*scheduleEntries);
  root["abiSignature"] = (primaryKernel.getName() + ":cann_static").str();
  root["cacheKey"] =
      (primaryKernel.getName() + ":static:" + options.soc).str();
  root["kernelKind"] = getKernelKindString(primaryKernel);
  root["workspaceSizeExpr"] = primaryWorkspaceInfo->sizeExpr;
  root["workspaceSizeBytes"] = primaryWorkspaceInfo->sizeBytes;
  root["shapeArgOrder"] = buildShapeArgOrder(*fieldsOr);
  root["shape"] = buildShapeDescriptor(*fieldsOr);
  root["workspace"] =
      buildWorkspaceDescriptor(primaryKernel, *primaryWorkspaceInfo);
  root["resources"] = buildResourceDescriptor(primaryKernel);
  root["hostTilingBindings"] = buildHostTilingBindings(kernels);
  root["kernelGraph"] = std::move(*kernelGraph);
  root["kernel_entries"] = std::move(kernelEntries);
  return writeJsonFile(module.getOperation(), outPath, std::move(root));
}

LogicalResult emitHostTilingCpp(ModuleOp module, StringRef outPath,
                                const CannRuntimeArtifactOptions &options) {
  struct HostTilingKernelInfo {
    func::FuncOp funcOp;
    emitasc::PyStructType tilingType;
    WorkspaceInfo workspaceInfo;
    SmallVector<TilingFieldInfo> fields;
    SmallVector<unsigned> shapeFieldPositions;
    SmallVector<int64_t> selectedTileShape;
    std::string hostWorkspaceExpr;
    bool workspaceExprUsesShapeArgs = false;
    std::string structName;
  };

  SmallVector<func::FuncOp> kernels = collectGlobalKernels(module);
  if (kernels.empty()) {
    module.emitError() << "host tiling requires at least one global kernel";
    return failure();
  }

  SmallVector<HostTilingKernelInfo, 4> infos;
  llvm::StringSet<> usedStructNames;
  bool hasMultipleKernels = kernels.size() > 1;
  for (func::FuncOp kernel : kernels) {
    FailureOr<emitasc::PyStructType> tilingTypeOr = getTilingType(kernel);
    if (failed(tilingTypeOr))
      return failure();
    FailureOr<WorkspaceInfo> workspaceInfo = getWorkspaceInfo(kernel);
    if (failed(workspaceInfo))
      return failure();
    FailureOr<SmallVector<TilingFieldInfo>> fieldsOr =
        collectTilingFields(kernel, *tilingTypeOr);
    if (failed(fieldsOr))
      return failure();
    FailureOr<SmallVector<int64_t>> selectedTileShape =
        collectSelectedTileShape(kernel);
    if (failed(selectedTileShape))
      return failure();

    auto types = tilingTypeOr->getTypesAttr().getValue();
    auto names = tilingTypeOr->getNamesAttr().getValue();
    if (types.size() != names.size())
      return kernel.emitError("PyStructType types/names size mismatch");

    HostTilingKernelInfo info;
    info.funcOp = kernel;
    info.tilingType = *tilingTypeOr;
    info.workspaceInfo = *workspaceInfo;
    info.fields = std::move(*fieldsOr);
    info.selectedTileShape = std::move(*selectedTileShape);
    for (auto [index, nameAttr] : llvm::enumerate(names)) {
      StringRef name = cast<StringAttr>(nameAttr).getValue();
      if (isShapeField(name))
        info.shapeFieldPositions.push_back(index);
    }
    FailureOr<std::string> hostWorkspaceExpr = buildHostWorkspaceSizeExpr(
        kernel, info.workspaceInfo.sizeExpr, info.fields,
        info.workspaceExprUsesShapeArgs);
    if (failed(hostWorkspaceExpr))
      return failure();
    info.hostWorkspaceExpr = std::move(*hostWorkspaceExpr);

    std::string baseStructName =
        sanitizeCppIdentifier(info.tilingType.getNameAttr().getValue());
    info.structName = baseStructName;
    if (hasMultipleKernels &&
        (baseStructName == "TilingData" ||
         usedStructNames.contains(info.structName))) {
      info.structName =
          baseStructName + "_" + sanitizeCppIdentifier(kernel.getName());
    }
    usedStructNames.insert(info.structName);
    infos.push_back(std::move(info));
  }

  return writeTextFile(module.getOperation(), outPath, [&](raw_ostream &os) {
    os << "#include <cstdint>\n";
    os << "#include <cstring>\n\n";

    for (const HostTilingKernelInfo &info : infos) {
      auto types = info.tilingType.getTypesAttr().getValue();
      auto names = info.tilingType.getNamesAttr().getValue();
      os << "struct " << info.structName << " {\n";
      for (auto [typeAttr, nameAttr] : llvm::zip(types, names)) {
        Type type = cast<TypeAttr>(typeAttr).getValue();
        StringRef name = cast<StringAttr>(nameAttr).getValue();
        os << "  " << getHostCppType(type) << " " << name << ";\n";
      }
      os << "};\n\n";
    }

    os << "extern \"C\" {\n\n";
    for (const HostTilingKernelInfo &info : infos) {
      func::FuncOp funcOp = info.funcOp;
      StringRef kernelName = funcOp.getName();
      auto names = info.tilingType.getNamesAttr().getValue();
      os << "int32_t " << kernelName << "_GetTilingSize(void) {\n";
      os << "  return static_cast<int32_t>(sizeof(" << info.structName
         << "));\n";
      os << "}\n\n";

      os << "int32_t " << kernelName
         << "_GetTiling(const int64_t* shape_args, int32_t shape_count, "
            "void* tiling_out) {\n";
      os << "  if (shape_count != " << info.shapeFieldPositions.size()
         << " || tiling_out == nullptr";
      if (!info.shapeFieldPositions.empty())
        os << " || shape_args == nullptr";
      os << ")\n";
      os << "    return 1;\n";
      os << "  " << info.structName << " data{};\n";
      unsigned shapeIndex = 0;
      unsigned tileIndex = 0;
      for (auto [index, nameAttr] : llvm::enumerate(names)) {
        StringRef name = cast<StringAttr>(nameAttr).getValue();
        os << "  data." << name << " = ";
        if (isShapeField(name)) {
          os << "shape_args[" << shapeIndex++ << "]";
        } else if (tileIndex < info.selectedTileShape.size()) {
          os << info.selectedTileShape[tileIndex++];
        } else {
          os << "0";
          ++tileIndex;
        }
        os << ";\n";
      }
      os << "  std::memcpy(tiling_out, &data, sizeof(" << info.structName
         << "));\n";
      os << "  return 0;\n";
      os << "}\n\n";

      os << "int64_t " << kernelName
         << "_GetBlockDim(const int64_t* shape_args, int32_t shape_count) {\n";
      os << "  (void)shape_args;\n";
      os << "  return shape_count == " << info.shapeFieldPositions.size()
         << " ? 20 : -1;\n";
      os << "}\n\n";

      os << "int64_t " << kernelName
         << "_GetWorkspaceSize(const int64_t* shape_args, int32_t shape_count) {\n";
      if (!info.workspaceExprUsesShapeArgs)
        os << "  (void)shape_args;\n";
      os << "  return shape_count == " << info.shapeFieldPositions.size();
      if (info.workspaceExprUsesShapeArgs)
        os << " && shape_args != nullptr";
      os << " ? " << info.hostWorkspaceExpr << " : -1;\n";
      os << "}\n\n";
    }
    os << "} // extern \"C\"\n";
  });
}

} // namespace mlir::afir::cann
