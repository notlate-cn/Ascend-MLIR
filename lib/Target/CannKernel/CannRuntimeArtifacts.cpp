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

#include <algorithm>
#include <cctype>
#include <functional>
#include <optional>

using namespace mlir;

namespace mlir::ascend::cann {
namespace {

struct TilingFieldInfo {
  std::string name;
  std::string type;
  bool isShape = false;
  std::string shapeKey;
};

struct TileParamSpaceInfo {
  std::string name;
  std::string binding;
  int64_t defaultValue = ShapedType::kDynamic;
  int64_t upperBound = ShapedType::kDynamic;
  int64_t extent = ShapedType::kDynamic;
};

struct WorkspaceInfo {
  int64_t sizeBytes = 0;
  std::string sizeExpr = "0";
};

static constexpr llvm::StringLiteral kKernelMetadataKernelKey = "kernel";
static constexpr llvm::StringLiteral kKernelMetadataDecisionIdKey =
    "decision_id";
static constexpr llvm::StringLiteral kKernelMetadataTileBindingKey =
    "tile_binding";
static constexpr llvm::StringLiteral kKernelMetadataTileParamsKey =
    "tile_params";
static constexpr llvm::StringLiteral kKernelMetadataTailPoliciesKey =
    "tail_policies";
static constexpr llvm::StringLiteral kKernelMetadataTailPlanKey =
    "tail_plan";
static constexpr llvm::StringLiteral kKernelMetadataGuardKey = "guard";
static constexpr llvm::StringLiteral kKernelMetadataPriorityKey = "priority";
static constexpr llvm::StringLiteral kKernelMetadataFallbackKey = "fallback";
static constexpr llvm::StringLiteral kKernelMetadataShapeBucketKey =
    "shape_bucket_key";
static constexpr llvm::StringLiteral kKernelMetadataHostTilingIdKey =
    "host_tiling_id";
static constexpr llvm::StringLiteral kKernelMetadataWorkspaceSizeExprKey =
    "workspace_size_expr";
static constexpr llvm::StringLiteral kKernelMetadataWorkspaceSizeBytesKey =
    "workspace_size_bytes";
static constexpr llvm::StringLiteral kKernelMetadataBlockDimKey = "block_dim";
static constexpr llvm::StringLiteral kKernelMetadataStructuredLoweringKey =
    "structured_lowering";

static SmallVector<func::FuncOp> collectGlobalKernels(ModuleOp module) {
  SmallVector<func::FuncOp> kernels;
  for (Operation &child : module.getBody()->getOperations())
    if (auto funcOp = dyn_cast<func::FuncOp>(child))
      if (funcOp->hasAttr(ascendc::attr::global))
        kernels.push_back(funcOp);
  return kernels;
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

static FailureOr<llvm::json::Array>
buildStringArrayJson(func::FuncOp funcOp, ArrayAttr arrayAttr,
                     StringRef fieldName) {
  llvm::json::Array values;
  for (auto [index, rawValue] : llvm::enumerate(arrayAttr)) {
    auto value = dyn_cast<StringAttr>(rawValue);
    if (!value)
      return funcOp.emitError()
             << "structured_lowering field '" << fieldName << "' element "
             << index << " must be a string attribute";
    values.push_back(value.getValue().str());
  }
  return values;
}

static FailureOr<llvm::json::Value>
buildStructuredLoweringLoopAxisJson(func::FuncOp funcOp, Attribute axisAttr,
                                    size_t index) {
  if (auto axis = dyn_cast<StringAttr>(axisAttr))
    return llvm::json::Value(axis.getValue().str());

  auto axisDict = dyn_cast<DictionaryAttr>(axisAttr);
  if (!axisDict)
    return funcOp.emitError()
           << ::mlir::ascend::kScheduleStructuredLoweringAttr
           << " field 'loop_axes' element " << index
           << " must be a string or dictionary attribute";

  llvm::json::Object axisJson;
  if (auto axis = dyn_cast_or_null<IntegerAttr>(axisDict.get("axis"))) {
    if (!axis.getType().isInteger(64))
      return funcOp.emitError()
             << ::mlir::ascend::kScheduleStructuredLoweringAttr
             << " field 'loop_axes' element " << index
             << " field 'axis' must be an i64 integer attribute";
    axisJson["axis"] = axis.getInt();
  }
  for (StringRef key : {"axis_kind", "tile_param", "binding"}) {
    if (auto value = dyn_cast_or_null<StringAttr>(axisDict.get(key)))
      axisJson[key] = value.getValue().str();
  }
  for (StringRef key : {"roles", "primitive_uses"}) {
    if (auto values = dyn_cast_or_null<ArrayAttr>(axisDict.get(key))) {
      FailureOr<llvm::json::Array> jsonValues =
          buildStringArrayJson(funcOp, values, key);
      if (failed(jsonValues))
        return failure();
      axisJson[key] = std::move(*jsonValues);
    }
  }
  return llvm::json::Value(std::move(axisJson));
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

static void appendUniqueTileValue(SmallVectorImpl<int64_t> &values,
                                  int64_t value, int64_t upperBound) {
  if (value <= 0)
    return;
  if (!ShapedType::isDynamic(upperBound))
    value = std::min(value, upperBound);
  if (!llvm::is_contained(values, value))
    values.push_back(value);
}

static SmallVector<int64_t>
buildRuntimeTileSearchValues(const TileParamSpaceInfo &tileParam) {
  SmallVector<int64_t> values;
  int64_t upperBound = tileParam.upperBound;
  if (ShapedType::isDynamic(upperBound) || upperBound <= 0)
    upperBound = tileParam.defaultValue;
  if (!ShapedType::isDynamic(tileParam.extent) && tileParam.extent > 0)
    upperBound = ShapedType::isDynamic(upperBound) || upperBound <= 0
                     ? tileParam.extent
                     : std::min(upperBound, tileParam.extent);
  if (ShapedType::isDynamic(upperBound) || upperBound <= 0)
    upperBound = 1;

  int64_t defaultValue = tileParam.defaultValue;
  if (ShapedType::isDynamic(defaultValue) || defaultValue <= 0)
    defaultValue = std::min<int64_t>(32, upperBound);

  appendUniqueTileValue(values, 1, upperBound);
  appendUniqueTileValue(values, defaultValue / 4, upperBound);
  appendUniqueTileValue(values, defaultValue / 2, upperBound);
  appendUniqueTileValue(values, defaultValue, upperBound);
  appendUniqueTileValue(values, defaultValue * 2, upperBound);
  appendUniqueTileValue(values, defaultValue * 4, upperBound);
  if (!ShapedType::isDynamic(tileParam.extent) && tileParam.extent > 0) {
    appendUniqueTileValue(values, tileParam.extent / 4, upperBound);
    appendUniqueTileValue(values, tileParam.extent / 2, upperBound);
    appendUniqueTileValue(values, tileParam.extent, upperBound);
  }

  llvm::sort(values);
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

static llvm::json::Array buildTilingSchema(
    ArrayRef<TilingFieldInfo> fields,
    const llvm::StringMap<TileParamSpaceInfo> *tileParams = nullptr) {
  llvm::json::Array schema;
  for (const TilingFieldInfo &field : fields) {
    llvm::json::Object entry;
    entry["name"] = field.name;
    entry["type"] = field.type;
    const TileParamSpaceInfo *tileParam = nullptr;
    if (tileParams) {
      auto tileParamIt = tileParams->find(field.name);
      if (tileParamIt != tileParams->end())
        tileParam = &tileParamIt->second;
    }
    bool hasTileParam = tileParam != nullptr;
    bool fixedTileParam = hasTileParam && tileParam->binding != "runtime";
    entry["fixed"] = field.isShape || fixedTileParam;
    if (field.isShape) {
      entry["shape_key"] = field.shapeKey;
    } else if (hasTileParam) {
      llvm::json::Array values;
      if (fixedTileParam) {
        int64_t value = tileParam->defaultValue;
        if (!ShapedType::isDynamic(value))
          values.push_back(value);
        entry["value"] = ShapedType::isDynamic(value) ? 0 : value;
      } else {
        for (int64_t value : buildRuntimeTileSearchValues(*tileParam))
          values.push_back(value);
      }
      entry["values"] = std::move(values);
    } else {
      entry["values"] = llvm::json::Array{};
    }
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
      ::mlir::ascend::kCannWorkspaceSizeBytesAttr);
  if (sizeAttr) {
    int64_t sizeBytes = sizeAttr.getInt();
    if (sizeBytes < 0)
      return funcOp.emitError()
             << ::mlir::ascend::kCannWorkspaceSizeBytesAttr
             << " must be a non-negative integer attribute";
    info.sizeBytes = sizeBytes;
    info.sizeExpr = std::to_string(sizeBytes);
  }

  auto exprAttr = funcOp->getAttrOfType<StringAttr>(
      ::mlir::ascend::kCannWorkspaceSizeExprAttr);
  if (exprAttr) {
    if (exprAttr.getValue().trim().empty())
      return funcOp.emitError()
             << ::mlir::ascend::kCannWorkspaceSizeExprAttr
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
      ::mlir::ascend::kAscendCKernelKindAttr);
  if (!kindAttr)
    return "unknown";
  StringRef kind = kindAttr.getValue();
  if (kind == ::mlir::ascend::kAscendCKernelKindVec ||
      kind == ::mlir::ascend::kAscendCKernelKindCube ||
      kind == ::mlir::ascend::kAscendCKernelKindMix)
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
    if (!callOp.getCallee().starts_with("ascend_gm_store<"))
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
           << "artifact manifest ABI requires cann.num_inputs";

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
               << ::mlir::ascend::kCannWorkspaceSizeExprAttr
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
           << ::mlir::ascend::kCannWorkspaceSizeExprAttr
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

static bool isSupportedTileBinding(StringRef value) {
  return llvm::StringSwitch<bool>(value)
      .Case("runtime", true)
      .Case("extent", true)
      .Case("static_fallback", true)
      .Default(false);
}

static bool isSupportedTileAxisKind(StringRef value) {
  return llvm::StringSwitch<bool>(value)
      .Case("parallel", true)
      .Case("reduction", true)
      .Case("unknown", true)
      .Default(false);
}

static bool isSupportedAxisExecutionRole(StringRef value) {
  return llvm::StringSwitch<bool>(value)
      .Case("bind_core", true)
      .Case("kernel_loop", true)
      .Case("vectorize", true)
      .Case("full_reduction", true)
      .Case("chunked_reduction", true)
      .Case("broadcast_projection", true)
      .Case("layout_carry", true)
      .Default(false);
}

static FailureOr<SmallVector<DictionaryAttr>>
collectKernelScheduleMetadata(func::FuncOp funcOp);

static std::optional<std::string>
selectMergedSinkKernelName(func::FuncOp funcOp,
                           ArrayRef<std::string> metadataKernelNames) {
  if (metadataKernelNames.empty())
    return std::nullopt;

  ModuleOp module = funcOp->getParentOfType<ModuleOp>();
  if (!module)
    return std::nullopt;

  auto edgesAttr = module->getAttrOfType<ArrayAttr>(
      ::mlir::ascend::kKernelGraphEdgesAttr);
  if (!edgesAttr)
    return std::nullopt;

  llvm::StringSet<> metadataKernels;
  for (const std::string &name : metadataKernelNames)
    metadataKernels.insert(name);

  llvm::StringSet<> fromKernels;
  llvm::StringSet<> toKernels;
  for (Attribute rawEdge : edgesAttr) {
    auto edge = dyn_cast<DictionaryAttr>(rawEdge);
    if (!edge)
      continue;
    auto from = dyn_cast_or_null<StringAttr>(edge.get("from"));
    auto to = dyn_cast_or_null<StringAttr>(edge.get("to"));
    if (!from || !to || from.getValue() == to.getValue())
      continue;
    if (!metadataKernels.contains(from.getValue()) ||
        !metadataKernels.contains(to.getValue()))
      continue;
    fromKernels.insert(from.getValue());
    toKernels.insert(to.getValue());
  }

  std::optional<std::string> sinkKernel;
  for (const std::string &name : metadataKernelNames) {
    if (!toKernels.contains(name) || fromKernels.contains(name))
      continue;
    if (sinkKernel && *sinkKernel != name)
      return std::nullopt;
    sinkKernel = name;
  }
  return sinkKernel;
}

static FailureOr<DictionaryAttr>
lookupKernelScheduleMetadata(func::FuncOp funcOp) {
  auto entriesOr = collectKernelScheduleMetadata(funcOp);
  if (failed(entriesOr))
    return failure();
  if (!entriesOr->empty())
    return entriesOr->front();
  return DictionaryAttr();
}

static FailureOr<SmallVector<DictionaryAttr>>
collectKernelScheduleMetadata(func::FuncOp funcOp) {
  auto metadata =
      funcOp->getAttrOfType<ArrayAttr>(
          ::mlir::ascend::kScheduleKernelMetadataAttr);
  if (!metadata)
    return SmallVector<DictionaryAttr>{};

  SmallVector<DictionaryAttr> matchingEntries;
  SmallVector<DictionaryAttr> allEntries;
  SmallVector<std::string> metadataKernelNames;
  DictionaryAttr singleFallbackEntry;
  for (auto [index, rawEntry] : llvm::enumerate(metadata)) {
    auto entry = dyn_cast<DictionaryAttr>(rawEntry);
    if (!entry)
      return funcOp.emitError()
             << ::mlir::ascend::kScheduleKernelMetadataAttr
             << " element " << index << " must be a dictionary attribute";

    auto kernel =
        dyn_cast_or_null<StringAttr>(entry.get(kKernelMetadataKernelKey));
    if (!kernel)
      return funcOp.emitError()
             << ::mlir::ascend::kScheduleKernelMetadataAttr
             << " element " << index
             << " entries must include a string kernel field";

    allEntries.push_back(entry);
    metadataKernelNames.push_back(kernel.getValue().str());
    if (kernel.getValue() == funcOp.getName())
      matchingEntries.push_back(entry);
    if (!singleFallbackEntry && metadata.size() == 1)
      singleFallbackEntry = entry;
  }

  if (!matchingEntries.empty())
    return matchingEntries;
  if (singleFallbackEntry)
    return SmallVector<DictionaryAttr>{singleFallbackEntry};
  if (std::optional<std::string> sinkKernel =
          selectMergedSinkKernelName(funcOp, metadataKernelNames)) {
    SmallVector<DictionaryAttr> sinkEntries;
    for (auto [entry, kernelName] :
         llvm::zip_equal(allEntries, metadataKernelNames)) {
      if (kernelName == *sinkKernel)
        sinkEntries.push_back(entry);
    }
    if (!sinkEntries.empty())
      return sinkEntries;
  }
  return SmallVector<DictionaryAttr>{};
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

  bool hasTileBinding = static_cast<bool>(getScheduleMetadataAttr(
      funcOp, *kernelMetadata, ::mlir::ascend::kScheduleTileBindingAttr,
      kKernelMetadataTileBindingKey));
  bool hasTileParams = static_cast<bool>(getScheduleMetadataAttr(
      funcOp, *kernelMetadata, ::mlir::ascend::kScheduleTileParamsAttr,
      kKernelMetadataTileParamsKey));
  bool hasTailPolicies =
      static_cast<bool>(getScheduleMetadataAttr(
          funcOp, *kernelMetadata,
          ::mlir::ascend::kScheduleTailPoliciesAttr,
          kKernelMetadataTailPoliciesKey));
  bool hasTailPlan = static_cast<bool>(getScheduleMetadataAttr(
      funcOp, *kernelMetadata, ::mlir::ascend::kScheduleTailPlanAttr,
      kKernelMetadataTailPlanKey));
  bool hasAnyMetadata =
      hasTileBinding || hasTileParams || hasTailPolicies || hasTailPlan;
  bool hasAllMetadata = hasTailPolicies && hasTailPlan;
  if (hasAnyMetadata && !hasAllMetadata)
    return funcOp.emitError()
           << "schedule metadata requires "
           << ::mlir::ascend::kScheduleTailPoliciesAttr << " and "
           << ::mlir::ascend::kScheduleTailPlanAttr << " together";
  if (hasAnyMetadata && hasTileBinding != hasTileParams)
    return funcOp.emitError()
           << "schedule metadata requires "
           << ::mlir::ascend::kScheduleTileBindingAttr << " and "
           << ::mlir::ascend::kScheduleTileParamsAttr << " together";

  return success();
}

static LogicalResult validateScheduleMetadataAttributes(func::FuncOp funcOp) {
  FailureOr<DictionaryAttr> kernelMetadata =
      lookupKernelScheduleMetadata(funcOp);
  if (failed(kernelMetadata))
    return failure();

  if (Attribute tileBinding = getScheduleMetadataAttr(
          funcOp, *kernelMetadata,
          ::mlir::ascend::kScheduleTileBindingAttr,
          kKernelMetadataTileBindingKey))
    if (!isa<StringAttr>(tileBinding))
      return funcOp.emitError()
             << ::mlir::ascend::kScheduleTileBindingAttr
             << " must be a string attribute";

  if (Attribute tileParams = getScheduleMetadataAttr(
          funcOp, *kernelMetadata,
          ::mlir::ascend::kScheduleTileParamsAttr,
          kKernelMetadataTileParamsKey))
    if (!isa<ArrayAttr>(tileParams))
      return funcOp.emitError()
             << ::mlir::ascend::kScheduleTileParamsAttr
             << " must be an array attribute";

  if (Attribute tailPolicies = getScheduleMetadataAttr(
          funcOp, *kernelMetadata,
          ::mlir::ascend::kScheduleTailPoliciesAttr,
          kKernelMetadataTailPoliciesKey))
    if (!isa<ArrayAttr>(tailPolicies))
      return funcOp.emitError()
             << ::mlir::ascend::kScheduleTailPoliciesAttr
             << " must be an array attribute";

  return success();
}

static FailureOr<llvm::StringMap<TileParamSpaceInfo>>
collectTileParamSpaceInfos(func::FuncOp funcOp) {
  FailureOr<DictionaryAttr> kernelMetadata =
      lookupKernelScheduleMetadata(funcOp);
  if (failed(kernelMetadata))
    return failure();

  llvm::StringMap<TileParamSpaceInfo> infos;
  auto tileParams = dyn_cast_or_null<ArrayAttr>(getScheduleMetadataAttr(
      funcOp, *kernelMetadata, ::mlir::ascend::kScheduleTileParamsAttr,
      kKernelMetadataTileParamsKey));
  if (!tileParams)
    return infos;

  for (auto [index, rawEntry] : llvm::enumerate(tileParams)) {
    auto entry = dyn_cast<DictionaryAttr>(rawEntry);
    if (!entry)
      return funcOp.emitError()
             << ::mlir::ascend::kScheduleTileParamsAttr << " element "
             << index << " must be a dictionary attribute";
    auto name = dyn_cast_or_null<StringAttr>(entry.get("name"));
    auto binding = dyn_cast_or_null<StringAttr>(entry.get("binding"));
    auto defaultValue = dyn_cast_or_null<IntegerAttr>(entry.get("default"));
    auto upperBound = dyn_cast_or_null<IntegerAttr>(entry.get("upper_bound"));
    auto extent = dyn_cast_or_null<IntegerAttr>(entry.get("extent"));
    if (!name || !binding || !defaultValue ||
        !defaultValue.getType().isInteger(64) || !upperBound ||
        !upperBound.getType().isInteger(64) || !extent ||
        !extent.getType().isInteger(64))
      return funcOp.emitError()
             << ::mlir::ascend::kScheduleTileParamsAttr << " element "
             << index
             << " must include string 'name'/'binding' and i64 "
                "'default'/'upper_bound'/'extent' fields";
    TileParamSpaceInfo info;
    info.name = name.getValue().str();
    info.binding = binding.getValue().str();
    info.defaultValue = defaultValue.getInt();
    info.upperBound = upperBound.getInt();
    info.extent = extent.getInt();
    infos[info.name] = std::move(info);
  }

  return infos;
}

static FailureOr<llvm::json::Object>
buildScheduleTilingParams(func::FuncOp funcOp,
                          DictionaryAttr kernelMetadata) {
  llvm::json::Object tilingParams;

  if (failed(checkScheduleMetadataCompleteness(funcOp)))
    return failure();
  if (failed(validateScheduleMetadataAttributes(funcOp)))
    return failure();

  if (auto tileBinding = dyn_cast_or_null<StringAttr>(
          getScheduleMetadataAttr(
              funcOp, kernelMetadata,
              ::mlir::ascend::kScheduleTileBindingAttr,
              kKernelMetadataTileBindingKey))) {
    tilingParams["tile_binding"] = tileBinding.getValue().str();
  }

  if (Attribute rawTileParamsAttr = getScheduleMetadataAttr(
          funcOp, kernelMetadata,
          ::mlir::ascend::kScheduleTileParamsAttr,
          kKernelMetadataTileParamsKey)) {
    auto tileParamsAttr = dyn_cast<ArrayAttr>(rawTileParamsAttr);
    if (!tileParamsAttr)
      return funcOp.emitError()
             << ::mlir::ascend::kScheduleTileParamsAttr
             << " must be an array attribute";

    llvm::json::Array tileParamsJson;
    for (auto [index, tileParamEntryAttr] : llvm::enumerate(tileParamsAttr)) {
      auto tileParamEntry = dyn_cast<DictionaryAttr>(tileParamEntryAttr);
      if (!tileParamEntry)
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTileParamsAttr
               << " element " << index
               << " must be a dictionary attribute";

      auto name = dyn_cast_or_null<StringAttr>(tileParamEntry.get("name"));
      if (!name)
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTileParamsAttr << " element "
               << index << " field 'name' must be a string attribute";
      auto axis = dyn_cast_or_null<IntegerAttr>(tileParamEntry.get("axis"));
      auto defaultValue =
          dyn_cast_or_null<IntegerAttr>(tileParamEntry.get("default"));
      auto upperBound =
          dyn_cast_or_null<IntegerAttr>(tileParamEntry.get("upper_bound"));
      auto extent =
          dyn_cast_or_null<IntegerAttr>(tileParamEntry.get("extent"));
      if (!axis || !axis.getType().isInteger(64) || !defaultValue ||
          !defaultValue.getType().isInteger(64) || !upperBound ||
          !upperBound.getType().isInteger(64) || !extent ||
          !extent.getType().isInteger(64))
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTileParamsAttr << " element "
               << index
               << " fields 'axis', 'default', 'upper_bound', and 'extent' "
                  "must be i64 integer attributes";
      auto axisKind =
          dyn_cast_or_null<StringAttr>(tileParamEntry.get("axis_kind"));
      if (!axisKind || !isSupportedTileAxisKind(axisKind.getValue()))
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTileParamsAttr << " element "
               << index
               << " field 'axis_kind' must be one of parallel, reduction, "
                  "unknown";
      auto binding =
          dyn_cast_or_null<StringAttr>(tileParamEntry.get("binding"));
      if (!binding || !isSupportedTileBinding(binding.getValue()))
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTileParamsAttr << " element "
               << index
               << " field 'binding' must be one of runtime, extent, "
                  "static_fallback";

      auto roles = dyn_cast_or_null<ArrayAttr>(tileParamEntry.get("roles"));
      if (!roles)
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTileParamsAttr << " element "
               << index << " field 'roles' must be an array attribute";
      llvm::json::Array rolesJson;
      for (auto [roleIndex, roleAttr] : llvm::enumerate(roles)) {
        auto role = dyn_cast<StringAttr>(roleAttr);
        if (!role || !isSupportedAxisExecutionRole(role.getValue()))
          return funcOp.emitError()
                 << ::mlir::ascend::kScheduleTileParamsAttr << " element "
                 << index << " field 'roles' element " << roleIndex
                 << " has unsupported value";
        rolesJson.push_back(role.getValue().str());
      }

      auto primitiveUses =
          dyn_cast_or_null<ArrayAttr>(tileParamEntry.get("primitive_uses"));
      if (!primitiveUses)
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTileParamsAttr << " element "
               << index
               << " field 'primitive_uses' must be an array attribute";
      llvm::json::Array primitiveUsesJson;
      for (auto [useIndex, useAttr] : llvm::enumerate(primitiveUses)) {
        auto use = dyn_cast<StringAttr>(useAttr);
        if (!use || !isSupportedAffectedPrimitiveUse(use.getValue()))
          return funcOp.emitError()
                 << ::mlir::ascend::kScheduleTileParamsAttr << " element "
                 << index << " field 'primitive_uses' element " << useIndex
                 << " has unsupported value";
        primitiveUsesJson.push_back(use.getValue().str());
      }

      llvm::json::Object tileParamObject;
      tileParamObject["name"] = name.getValue().str();
      tileParamObject["axis"] = axis.getInt();
      tileParamObject["axisKind"] = axisKind.getValue().str();
      tileParamObject["binding"] = binding.getValue().str();
      tileParamObject["default"] = defaultValue.getInt();
      tileParamObject["upperBound"] = upperBound.getInt();
      tileParamObject["extent"] = extent.getInt();
      tileParamObject["roles"] = std::move(rolesJson);
      tileParamObject["primitiveUses"] = std::move(primitiveUsesJson);
      tileParamsJson.push_back(std::move(tileParamObject));
    }
    tilingParams["tile_params"] = std::move(tileParamsJson);
  }

  if (auto tailPolicies = dyn_cast_or_null<ArrayAttr>(
          getScheduleMetadataAttr(
              funcOp, kernelMetadata,
              ::mlir::ascend::kScheduleTailPoliciesAttr,
              kKernelMetadataTailPoliciesKey))) {
    llvm::json::Array tailPoliciesJson;
    for (auto [index, tailPolicyAttr] : llvm::enumerate(tailPolicies)) {
      auto tailPolicy = dyn_cast<StringAttr>(tailPolicyAttr);
      if (!tailPolicy)
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTailPoliciesAttr
               << " element " << index << " must be a string attribute";
      if (!isSupportedTailPolicy(tailPolicy.getValue()))
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTailPoliciesAttr
               << " element " << index << " has unsupported value '"
               << tailPolicy.getValue()
               << "'; expected one of must_divide, masked_tail, "
                  "scalar_epilogue, pad_and_mask, full_extent";
      tailPoliciesJson.push_back(tailPolicy.getValue().str());
    }
    tilingParams["tail_policies"] = std::move(tailPoliciesJson);
  }

  if (Attribute rawTailPlanAttr = getScheduleMetadataAttr(
          funcOp, kernelMetadata,
          ::mlir::ascend::kScheduleTailPlanAttr,
          kKernelMetadataTailPlanKey)) {
    auto tailPlanAttr = dyn_cast<ArrayAttr>(rawTailPlanAttr);
    if (!tailPlanAttr)
      return funcOp.emitError()
             << ::mlir::ascend::kScheduleTailPlanAttr
             << " must be an array attribute";

    llvm::json::Array tailPlanJson;
    for (auto [index, tailPlanEntryAttr] : llvm::enumerate(tailPlanAttr)) {
      auto tailPlanEntry = dyn_cast<DictionaryAttr>(tailPlanEntryAttr);
      if (!tailPlanEntry)
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTailPlanAttr
               << " element " << index
               << " must be a dictionary attribute";

      auto axis = dyn_cast_or_null<IntegerAttr>(tailPlanEntry.get("axis"));
      if (!axis || !axis.getType().isInteger(64))
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTailPlanAttr
               << " element " << index
               << " field 'axis' must be an i64 integer attribute";
      auto selected =
          dyn_cast_or_null<StringAttr>(tailPlanEntry.get("selected"));
      if (!selected)
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTailPlanAttr
               << " element " << index
               << " field 'selected' must be a string attribute";
      if (!isSupportedTailPolicy(selected.getValue()))
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTailPlanAttr
               << " element " << index
               << " field 'selected' has unsupported value '"
               << selected.getValue()
               << "'; expected one of must_divide, masked_tail, "
                  "scalar_epilogue, pad_and_mask, full_extent";
      auto affected =
          dyn_cast_or_null<ArrayAttr>(tailPlanEntry.get("affected"));
      if (!affected)
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTailPlanAttr
               << " element " << index
               << " field 'affected' must be an array attribute";
      auto align = dyn_cast_or_null<IntegerAttr>(tailPlanEntry.get("align"));
      if (!align || !align.getType().isInteger(64))
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTailPlanAttr
               << " element " << index
               << " field 'align' must be an i64 integer attribute";
      auto buffering =
          dyn_cast_or_null<StringAttr>(tailPlanEntry.get("buffering"));
      if (!buffering)
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTailPlanAttr
               << " element " << index
               << " field 'buffering' must be a string attribute";
      if (!isSupportedTailBufferingMode(buffering.getValue()))
        return funcOp.emitError()
               << ::mlir::ascend::kScheduleTailPlanAttr
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
                 << ::mlir::ascend::kScheduleTailPlanAttr
                 << " element " << index << " field 'affected' element "
                 << affectedIndex << " must be a string attribute";
        if (!isSupportedAffectedPrimitiveUse(affectedUse.getValue()))
          return funcOp.emitError()
                 << ::mlir::ascend::kScheduleTailPlanAttr
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

  if (Attribute rawStructuredLoweringAttr = getScheduleMetadataAttr(
          funcOp, kernelMetadata,
          ::mlir::ascend::kScheduleStructuredLoweringAttr,
          kKernelMetadataStructuredLoweringKey)) {
    auto structuredLowering =
        dyn_cast<DictionaryAttr>(rawStructuredLoweringAttr);
    if (!structuredLowering)
      return funcOp.emitError()
             << ::mlir::ascend::kScheduleStructuredLoweringAttr
             << " must be a dictionary attribute";

    llvm::json::Object structuredLoweringJson;
    for (StringRef key :
         {"contract", "representation", "cache_read_marker",
          "cache_write_marker", "pipeline_marker", "double_buffer_marker"}) {
      if (auto value =
              dyn_cast_or_null<StringAttr>(structuredLowering.get(key)))
        structuredLoweringJson[key] = value.getValue().str();
    }
    for (StringRef key : {"guard_marker_count", "tail_marker_count"}) {
      if (auto value =
              dyn_cast_or_null<IntegerAttr>(structuredLowering.get(key))) {
        if (!value.getType().isInteger(64))
          return funcOp.emitError()
                 << ::mlir::ascend::kScheduleStructuredLoweringAttr
                 << " field '" << key << "' must be an i64 integer attribute";
        structuredLoweringJson[key] = value.getInt();
      }
    }
    if (auto loopAxes =
            dyn_cast_or_null<ArrayAttr>(structuredLowering.get("loop_axes"))) {
      llvm::json::Array loopAxesJson;
      for (auto [index, axisAttr] : llvm::enumerate(loopAxes)) {
        FailureOr<llvm::json::Value> axisJson =
            buildStructuredLoweringLoopAxisJson(funcOp, axisAttr, index);
        if (failed(axisJson))
          return failure();
        loopAxesJson.push_back(std::move(*axisJson));
      }
      structuredLoweringJson["loop_axes"] = std::move(loopAxesJson);
    }
    tilingParams["structured_lowering"] = std::move(structuredLoweringJson);
  }

  return tilingParams;
}

static FailureOr<llvm::json::Object>
buildScheduleTilingParams(func::FuncOp funcOp) {
  FailureOr<DictionaryAttr> kernelMetadata =
      lookupKernelScheduleMetadata(funcOp);
  if (failed(kernelMetadata))
    return failure();
  return buildScheduleTilingParams(funcOp, *kernelMetadata);
}

static std::optional<std::string> getStringMetadata(DictionaryAttr entry,
                                                    StringRef key) {
  if (!entry)
    return std::nullopt;
  if (auto attr = dyn_cast_or_null<StringAttr>(entry.get(key)))
    return attr.getValue().str();
  return std::nullopt;
}

static std::optional<int64_t> getI64Metadata(DictionaryAttr entry,
                                             StringRef key) {
  if (!entry)
    return std::nullopt;
  auto attr = dyn_cast_or_null<IntegerAttr>(entry.get(key));
  if (!attr || !attr.getType().isInteger(64))
    return std::nullopt;
  return attr.getInt();
}

static std::optional<bool> getBoolMetadata(DictionaryAttr entry,
                                           StringRef key) {
  if (!entry)
    return std::nullopt;
  if (auto attr = dyn_cast_or_null<BoolAttr>(entry.get(key)))
    return attr.getValue();
  return std::nullopt;
}

static llvm::json::Array
buildGuardSet(ArrayRef<DictionaryAttr> metadataEntries) {
  SmallVector<std::string, 4> guards;
  for (DictionaryAttr entry : metadataEntries) {
    std::optional<std::string> guard =
        getStringMetadata(entry, kKernelMetadataGuardKey);
    if (guard && !guard->empty() && *guard != "true")
      if (!llvm::is_contained(guards, *guard))
        guards.push_back(*guard);
  }

  llvm::json::Array out;
  for (const std::string &guard : guards)
    out.push_back(guard);
  return out;
}

struct HostScheduleVariantInfo {
  unsigned index = 0;
  unsigned order = 0;
  int64_t priority = 0;
  bool fallback = false;
  int64_t blockDim = 20;
  std::string guardExpr = "true";
  std::string workspaceExpr = "0";
  llvm::StringMap<int64_t> tileParamValues;
};

static FailureOr<std::string>
buildHostGuardExpr(func::FuncOp funcOp, StringRef expr,
                   ArrayRef<TilingFieldInfo> fields) {
  expr = expr.trim();
  if (expr.empty() || expr == "true")
    return std::string("true");
  if (expr == "false")
    return std::string("false");

  llvm::StringMap<unsigned> shapeFieldAbiPositions;
  unsigned shapeIndex = 0;
  for (const TilingFieldInfo &field : fields) {
    if (!field.isShape)
      continue;
    shapeFieldAbiPositions[field.name] = shapeIndex;
    shapeFieldAbiPositions[field.shapeKey] = shapeIndex;
    ++shapeIndex;
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
      if (ident == "true" || ident == "false") {
        hostExpr += ident.str();
        continue;
      }
      auto it = shapeFieldAbiPositions.find(ident);
      if (it == shapeFieldAbiPositions.end())
        return funcOp.emitError()
               << "host tiling guard references unknown shape value \""
               << ident << "\"";
      hostExpr += "shape_args[" + std::to_string(it->second) + "]";
      continue;
    }

    if (std::isdigit(ch) || std::isspace(ch) || expr[i] == '+' ||
        expr[i] == '-' || expr[i] == '*' || expr[i] == '/' ||
        expr[i] == '%' || expr[i] == '(' || expr[i] == ')' ||
        expr[i] == '<' || expr[i] == '>' || expr[i] == '=' ||
        expr[i] == '!' || expr[i] == '&' || expr[i] == '|') {
      hostExpr.push_back(expr[i++]);
      continue;
    }

    return funcOp.emitError()
           << "host tiling guard contains unsupported character '" << expr[i]
           << "'";
  }

  return hostExpr;
}

static LogicalResult
populateHostTileParamValues(func::FuncOp funcOp, Attribute rawTileParams,
                            llvm::StringMap<int64_t> &tileParamValues) {
  if (!rawTileParams)
    return success();
  auto tileParams = dyn_cast<ArrayAttr>(rawTileParams);
  if (!tileParams)
    return funcOp.emitError()
           << "host tiling tile_params metadata must be an array attribute";

  for (auto [index, rawEntry] : llvm::enumerate(tileParams)) {
    auto entry = dyn_cast<DictionaryAttr>(rawEntry);
    if (!entry)
      return funcOp.emitError()
             << "host tiling tile_params element " << index
             << " must be a dictionary attribute";
    auto name = dyn_cast_or_null<StringAttr>(entry.get("name"));
    auto defaultValue = dyn_cast_or_null<IntegerAttr>(entry.get("default"));
    if (!name || !defaultValue || !defaultValue.getType().isInteger(64))
      return funcOp.emitError()
             << "host tiling tile_params element " << index
             << " must include string 'name' and i64 'default' fields";
    tileParamValues[name.getValue()] = defaultValue.getInt();
  }
  return success();
}

static FailureOr<SmallVector<HostScheduleVariantInfo>>
collectHostScheduleVariants(func::FuncOp funcOp,
                            ArrayRef<TilingFieldInfo> fields,
                            const llvm::StringMap<TileParamSpaceInfo>
                                &tileParamInfos,
                            StringRef defaultWorkspaceExpr) {
  FailureOr<SmallVector<DictionaryAttr>> metadataEntries =
      collectKernelScheduleMetadata(funcOp);
  if (failed(metadataEntries))
    return failure();

  SmallVector<HostScheduleVariantInfo> variants;
  auto appendVariant = [&](DictionaryAttr metadata,
                           unsigned order) -> LogicalResult {
    HostScheduleVariantInfo variant;
    variant.index = variants.size();
    variant.order = order;
    variant.priority =
        getI64Metadata(metadata, kKernelMetadataPriorityKey).value_or(order);
    variant.fallback =
        getBoolMetadata(metadata, kKernelMetadataFallbackKey).value_or(false);
    variant.blockDim =
        getI64Metadata(metadata, kKernelMetadataBlockDimKey).value_or(20);

    FailureOr<std::string> guardExpr = buildHostGuardExpr(
        funcOp, getStringMetadata(metadata, kKernelMetadataGuardKey)
                    .value_or("true"),
        fields);
    if (failed(guardExpr))
      return failure();
    variant.guardExpr = std::move(*guardExpr);

    if (std::optional<std::string> workspaceExpr =
            getStringMetadata(metadata, kKernelMetadataWorkspaceSizeExprKey)) {
      bool usesShapeArgs = false;
      FailureOr<std::string> hostExpr = buildHostWorkspaceSizeExpr(
          funcOp, *workspaceExpr, fields, usesShapeArgs);
      if (failed(hostExpr))
        return failure();
      variant.workspaceExpr = std::move(*hostExpr);
    } else if (std::optional<int64_t> workspaceBytes =
                   getI64Metadata(metadata,
                                  kKernelMetadataWorkspaceSizeBytesKey)) {
      variant.workspaceExpr = std::to_string(*workspaceBytes);
    } else {
      variant.workspaceExpr = defaultWorkspaceExpr.str();
    }

    Attribute rawTileParams = getScheduleMetadataAttr(
        funcOp, metadata, ::mlir::ascend::kScheduleTileParamsAttr,
        kKernelMetadataTileParamsKey);
    if (failed(populateHostTileParamValues(funcOp, rawTileParams,
                                           variant.tileParamValues)))
      return failure();
    for (const auto &it : tileParamInfos)
      if (!variant.tileParamValues.contains(it.getKey()))
        variant.tileParamValues[it.getKey()] = it.getValue().defaultValue;

    variants.push_back(std::move(variant));
    return success();
  };

  if (metadataEntries->empty()) {
    if (failed(appendVariant(DictionaryAttr(), 0)))
      return failure();
  } else {
    for (auto [order, metadata] : llvm::enumerate(*metadataEntries))
      if (failed(appendVariant(metadata, order)))
        return failure();
  }
  return variants;
}

static SmallVector<const HostScheduleVariantInfo *>
getHostScheduleSelectionOrder(ArrayRef<HostScheduleVariantInfo> variants,
                              bool fallbackGroup) {
  SmallVector<const HostScheduleVariantInfo *> ordered;
  for (const HostScheduleVariantInfo &variant : variants)
    if (variant.fallback == fallbackGroup)
      ordered.push_back(&variant);
  llvm::stable_sort(ordered, [](const HostScheduleVariantInfo *lhs,
                                const HostScheduleVariantInfo *rhs) {
    if (lhs->priority != rhs->priority)
      return lhs->priority < rhs->priority;
    return lhs->order < rhs->order;
  });
  return ordered;
}

static std::string combineShapeBucketKeys(ArrayRef<std::string> keys) {
  if (keys.empty())
    return "static";
  if (keys.size() == 1)
    return keys.front();

  SmallVector<StringRef, 4> suffixes;
  StringRef commonPrefix;
  bool hasCommonPrefix = true;
  for (const std::string &key : keys) {
    StringRef keyRef(key);
    size_t split = keyRef.rfind('.');
    if (split == StringRef::npos) {
      hasCommonPrefix = false;
      break;
    }
    StringRef prefix = keyRef.take_front(split);
    if (commonPrefix.empty())
      commonPrefix = prefix;
    else if (commonPrefix != prefix) {
      hasCommonPrefix = false;
      break;
    }
    suffixes.push_back(keyRef.drop_front(split + 1));
  }

  std::string result;
  llvm::raw_string_ostream os(result);
  if (hasCommonPrefix && !commonPrefix.empty()) {
    os << commonPrefix << ".";
    llvm::interleave(suffixes, os, [&](StringRef suffix) { os << suffix; },
                     "_or_");
    return os.str();
  }

  llvm::interleave(keys, os, [&](StringRef key) { os << key; }, "_or_");
  return os.str();
}

static std::string
buildShapeBucketKey(ArrayRef<DictionaryAttr> metadataEntries) {
  SmallVector<std::string, 4> keys;
  for (DictionaryAttr entry : metadataEntries) {
    std::optional<std::string> key =
        getStringMetadata(entry, kKernelMetadataShapeBucketKey);
    if (key && !key->empty())
      if (!llvm::is_contained(keys, *key))
        keys.push_back(*key);
  }
  return combineShapeBucketKeys(keys);
}

static FailureOr<llvm::json::Array> buildScheduleEntries(func::FuncOp funcOp) {
  llvm::json::Array scheduleEntries;
  FailureOr<SmallVector<DictionaryAttr>> metadataEntries =
      collectKernelScheduleMetadata(funcOp);
  if (failed(metadataEntries))
    return failure();

  if (metadataEntries->empty()) {
    FailureOr<llvm::json::Object> tilingParams =
        buildScheduleTilingParams(funcOp, DictionaryAttr());
    if (failed(tilingParams))
      return failure();

    llvm::json::Object scheduleEntry;
    scheduleEntry["decisionId"] = "static_0";
    scheduleEntry["kernelName"] = funcOp.getName().str();
    scheduleEntry["guard"] = "true";
    scheduleEntry["priority"] = 0;
    scheduleEntry["fallback"] = false;
    scheduleEntry["shapeBucketKey"] = "static";
    scheduleEntry["hostTilingId"] = getHostTilingBindingId(funcOp);
    scheduleEntry["tilingParams"] = std::move(*tilingParams);
    scheduleEntries.push_back(std::move(scheduleEntry));
    return scheduleEntries;
  }

  for (auto [index, metadata] : llvm::enumerate(*metadataEntries)) {
    FailureOr<llvm::json::Object> tilingParams =
        buildScheduleTilingParams(funcOp, metadata);
    if (failed(tilingParams))
      return failure();

    std::string decisionId =
        getStringMetadata(metadata, kKernelMetadataDecisionIdKey)
            .value_or((llvm::Twine("static_") + llvm::Twine(index)).str());
    llvm::json::Object scheduleEntry;
    scheduleEntry["decisionId"] = decisionId;
    scheduleEntry["kernelName"] = funcOp.getName().str();
    scheduleEntry["guard"] =
        getStringMetadata(metadata, kKernelMetadataGuardKey).value_or("true");
    scheduleEntry["priority"] =
        getI64Metadata(metadata, kKernelMetadataPriorityKey)
            .value_or(static_cast<int64_t>(index));
    scheduleEntry["fallback"] =
        getBoolMetadata(metadata, kKernelMetadataFallbackKey).value_or(false);
    scheduleEntry["shapeBucketKey"] =
        getStringMetadata(metadata, kKernelMetadataShapeBucketKey)
            .value_or("static");
    scheduleEntry["hostTilingId"] =
        getStringMetadata(metadata, kKernelMetadataHostTilingIdKey)
            .value_or(getHostTilingBindingId(funcOp));
    if (std::optional<std::string> workspaceExpr =
            getStringMetadata(metadata, kKernelMetadataWorkspaceSizeExprKey))
      scheduleEntry["workspaceSizeExpr"] = *workspaceExpr;
    if (std::optional<int64_t> workspaceBytes =
            getI64Metadata(metadata, kKernelMetadataWorkspaceSizeBytesKey))
      scheduleEntry["workspaceSizeBytes"] = *workspaceBytes;
    if (std::optional<int64_t> blockDim =
            getI64Metadata(metadata, kKernelMetadataBlockDimKey))
      scheduleEntry["blockDim"] = *blockDim;
    scheduleEntry["tilingParams"] = std::move(*tilingParams);
    scheduleEntries.push_back(std::move(scheduleEntry));
  }
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
  FailureOr<llvm::StringMap<TileParamSpaceInfo>> tileParamInfos =
      collectTileParamSpaceInfos(funcOp);
  if (failed(tileParamInfos))
    return failure();
  FailureOr<llvm::json::Object> tilingParams =
      buildScheduleTilingParams(funcOp);
  if (failed(tilingParams))
    return failure();
  FailureOr<llvm::json::Array> scheduleEntries =
      buildScheduleEntries(funcOp);
  if (failed(scheduleEntries))
    return failure();
  FailureOr<SmallVector<DictionaryAttr>> metadataEntries =
      collectKernelScheduleMetadata(funcOp);
  if (failed(metadataEntries))
    return failure();
  FailureOr<WorkspaceInfo> workspaceInfo = getWorkspaceInfo(funcOp);
  if (failed(workspaceInfo))
    return failure();

  llvm::json::Object kernelEntry;
  kernelEntry["kernel_id"] = funcOp.getName().str();
  kernelEntry["entry_index"] = entryIndex;
  kernelEntry["shapeBucketKey"] = buildShapeBucketKey(*metadataEntries);
  kernelEntry["guardSet"] = buildGuardSet(*metadataEntries);
  kernelEntry["tilingSchema"] = buildTilingSchema(*fieldsOr, &*tileParamInfos);
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
            ::mlir::ascend::kScheduleKernelMetadataAttr);
    if (!metadata)
      continue;

    for (auto [index, rawEntry] : llvm::enumerate(metadata)) {
      auto entry = dyn_cast<DictionaryAttr>(rawEntry);
      if (!entry)
        return kernel.emitError()
               << ::mlir::ascend::kScheduleKernelMetadataAttr
               << " element " << index << " must be a dictionary attribute";

      auto internalKernel =
          dyn_cast_or_null<StringAttr>(entry.get(kKernelMetadataKernelKey));
      if (!internalKernel)
        return kernel.emitError()
               << ::mlir::ascend::kScheduleKernelMetadataAttr
               << " element " << index
               << " entries must include a string kernel field";

      auto existing = aliases.find(internalKernel.getValue());
      if (existing != aliases.end() &&
          StringRef(existing->second) != kernel.getName())
        return kernel.emitError()
               << ::mlir::ascend::kScheduleKernelMetadataAttr
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
      ::mlir::ascend::kKernelGraphEdgesAttr);
  llvm::json::Array edgesJson;
  if (!edgesAttr)
    return edgesJson;

  SmallVector<std::pair<std::string, std::string>> edges;
  for (auto [index, edgeAttr] : llvm::enumerate(edgesAttr)) {
    auto edge = dyn_cast<DictionaryAttr>(edgeAttr);
    if (!edge)
      return module.emitError()
             << ::mlir::ascend::kKernelGraphEdgesAttr << " element "
             << index << " must be a dictionary attribute";

    auto from = dyn_cast_or_null<StringAttr>(edge.get("from"));
    auto to = dyn_cast_or_null<StringAttr>(edge.get("to"));
    if (!from || !to)
      return module.emitError()
             << ::mlir::ascend::kKernelGraphEdgesAttr << " element "
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
             << ::mlir::ascend::kKernelGraphEdgesAttr << " element "
             << index << " references unknown source kernel '"
             << from.getValue() << "'";
    if (!resolvedTo)
      return module.emitError()
             << ::mlir::ascend::kKernelGraphEdgesAttr << " element "
             << index << " references unknown target kernel '" << to.getValue()
             << "'";

    if (*resolvedFrom == *resolvedTo && (!fromKnown || !toKnown))
      continue;

    auto carriedBuffers =
        dyn_cast_or_null<ArrayAttr>(edge.get("carried_buffers"));
    if (!carriedBuffers || carriedBuffers.empty())
      return module.emitError()
             << ::mlir::ascend::kKernelGraphEdgesAttr << " element "
             << index
             << " requires non-empty array field 'carried_buffers'";

    llvm::json::Array carriedBuffersJson;
    for (auto [bufferIndex, bufferAttr] : llvm::enumerate(carriedBuffers)) {
      auto buffer = dyn_cast<StringAttr>(bufferAttr);
      if (!buffer)
        return module.emitError()
               << ::mlir::ascend::kKernelGraphEdgesAttr << " element "
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
           << ::mlir::ascend::kKernelGraphEdgesAttr
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
  SmallVector<func::FuncOp> kernels = collectGlobalKernels(module);
  if (kernels.empty()) {
    module.emitError() << "tiling space requires at least one global kernel";
    return failure();
  }

  auto buildKernelObject = [&](func::FuncOp kernel)
      -> FailureOr<llvm::json::Object> {
    FailureOr<emitasc::PyStructType> tilingTypeOr = getTilingType(kernel);
    if (failed(tilingTypeOr))
      return failure();
    FailureOr<SmallVector<TilingFieldInfo>> fieldsOr =
        collectTilingFields(kernel, *tilingTypeOr);
    if (failed(fieldsOr))
      return failure();
    FailureOr<llvm::StringMap<TileParamSpaceInfo>> tileParamInfos =
        collectTileParamSpaceInfos(kernel);
    if (failed(tileParamInfos))
      return failure();
    FailureOr<WorkspaceInfo> workspaceInfo = getWorkspaceInfo(kernel);
    if (failed(workspaceInfo))
      return failure();
    FailureOr<llvm::json::Array> scheduleEntries =
        buildScheduleEntries(kernel);
    if (failed(scheduleEntries))
      return failure();
    FailureOr<SmallVector<DictionaryAttr>> metadataEntries =
        collectKernelScheduleMetadata(kernel);
    if (failed(metadataEntries))
      return failure();

    llvm::json::Object kernelObject;
    kernelObject["kernel"] = kernel.getName().str();
    kernelObject["kernel_file"] = options.kernelFile.str();
    kernelObject["soc"] = options.soc.str();
    kernelObject["block_dim_expr"] = "20";
    kernelObject["workspace_size_expr"] = workspaceInfo->sizeExpr;
    kernelObject["shapeBucketKey"] = buildShapeBucketKey(*metadataEntries);
    kernelObject["guardSet"] = buildGuardSet(*metadataEntries);
    kernelObject["tiling_params"] =
        buildTilingSchema(*fieldsOr, &*tileParamInfos);
    kernelObject["scheduleEntries"] = std::move(*scheduleEntries);
    return kernelObject;
  };

  FailureOr<llvm::json::Object> primaryKernelObject =
      buildKernelObject(kernels.front());
  if (failed(primaryKernelObject))
    return failure();

  llvm::json::Object root = std::move(*primaryKernelObject);
  root["schema_version"] = "2.0";
  llvm::json::Array kernelObjects;
  for (func::FuncOp kernel : kernels) {
    FailureOr<llvm::json::Object> kernelObject = buildKernelObject(kernel);
    if (failed(kernelObject))
      return failure();
    kernelObjects.push_back(std::move(*kernelObject));
  }
  root["kernels"] = std::move(kernelObjects);
  return writeJsonFile(module.getOperation(), outPath, std::move(root));
}

LogicalResult
emitArtifactManifestJson(ModuleOp module, StringRef outPath,
                         const CannRuntimeArtifactOptions &options) {
  SmallVector<func::FuncOp> kernels = collectGlobalKernels(module);
  if (kernels.empty()) {
    module.emitError() << "artifact manifest requires at least one global kernel";
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
  FailureOr<llvm::StringMap<TileParamSpaceInfo>> primaryTileParamInfos =
      collectTileParamSpaceInfos(primaryKernel);
  if (failed(primaryTileParamInfos))
    return failure();

  FailureOr<llvm::json::Object> tilingParams =
      buildScheduleTilingParams(primaryKernel);
  if (failed(tilingParams))
    return failure();
  FailureOr<llvm::json::Array> scheduleEntries =
      buildScheduleEntries(primaryKernel);
  if (failed(scheduleEntries))
    return failure();
  FailureOr<SmallVector<DictionaryAttr>> primaryMetadataEntries =
      collectKernelScheduleMetadata(primaryKernel);
  if (failed(primaryMetadataEntries))
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
  root["shapeBucketKey"] = buildShapeBucketKey(*primaryMetadataEntries);
  root["guardSet"] = buildGuardSet(*primaryMetadataEntries);
  root["tilingSchema"] =
      buildTilingSchema(*fieldsOr, &*primaryTileParamInfos);
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
    llvm::StringMap<TileParamSpaceInfo> tileParamInfos;
    SmallVector<HostScheduleVariantInfo> scheduleVariants;
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
    FailureOr<llvm::StringMap<TileParamSpaceInfo>> tileParamInfos =
        collectTileParamSpaceInfos(kernel);
    if (failed(tileParamInfos))
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
    info.tileParamInfos = std::move(*tileParamInfos);
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
    FailureOr<SmallVector<HostScheduleVariantInfo>> scheduleVariants =
        collectHostScheduleVariants(kernel, info.fields, info.tileParamInfos,
                                    info.hostWorkspaceExpr);
    if (failed(scheduleVariants))
      return failure();
    info.scheduleVariants = std::move(*scheduleVariants);

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

    for (const HostTilingKernelInfo &info : infos) {
      func::FuncOp funcOp = info.funcOp;
      StringRef kernelName = funcOp.getName();
      os << "static int32_t " << kernelName
         << "_SelectScheduleEntry(const int64_t* shape_args) {\n";
      os << "  (void)shape_args;\n";
      for (const HostScheduleVariantInfo *variant :
           getHostScheduleSelectionOrder(info.scheduleVariants,
                                         /*fallbackGroup=*/false)) {
        os << "  if (" << variant->guardExpr << ")\n";
        os << "    return " << variant->index << ";\n";
      }
      for (const HostScheduleVariantInfo *variant :
           getHostScheduleSelectionOrder(info.scheduleVariants,
                                         /*fallbackGroup=*/true)) {
        os << "  if (" << variant->guardExpr << ")\n";
        os << "    return " << variant->index << ";\n";
      }
      os << "  return -1;\n";
      os << "}\n\n";
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
      os << "  int32_t schedule_index = " << kernelName
         << "_SelectScheduleEntry(shape_args);\n";
      os << "  if (schedule_index < 0)\n";
      os << "    return 2;\n";
      unsigned shapeIndex = 0;
      for (auto [index, nameAttr] : llvm::enumerate(names)) {
        StringRef name = cast<StringAttr>(nameAttr).getValue();
        if (isShapeField(name)) {
          os << "  data." << name << " = ";
          os << "shape_args[" << shapeIndex++ << "]";
          os << ";\n";
          continue;
        }
        os << "  switch (schedule_index) {\n";
        for (const HostScheduleVariantInfo &variant :
             info.scheduleVariants) {
          int64_t value = 0;
          auto tileParamIt = variant.tileParamValues.find(name);
          if (tileParamIt != variant.tileParamValues.end())
            value = tileParamIt->second;
          if (ShapedType::isDynamic(value) || value <= 0)
            value = 1;
          os << "  case " << variant.index << ":\n";
          os << "    data." << name << " = " << value << ";\n";
          os << "    break;\n";
        }
        os << "  default:\n";
        os << "    return 3;\n";
        os << "  }\n";
      }
      os << "  std::memcpy(tiling_out, &data, sizeof(" << info.structName
         << "));\n";
      os << "  return 0;\n";
      os << "}\n\n";

      os << "int64_t " << kernelName
         << "_GetBlockDim(const int64_t* shape_args, int32_t shape_count) {\n";
      os << "  if (shape_count != " << info.shapeFieldPositions.size();
      if (!info.shapeFieldPositions.empty())
        os << " || shape_args == nullptr";
      os << ")\n";
      os << "    return -1;\n";
      os << "  switch (" << kernelName
         << "_SelectScheduleEntry(shape_args)) {\n";
      for (const HostScheduleVariantInfo &variant : info.scheduleVariants) {
        os << "  case " << variant.index << ":\n";
        os << "    return " << variant.blockDim << ";\n";
      }
      os << "  default:\n";
      os << "    return -1;\n";
      os << "  }\n";
      os << "}\n\n";

      os << "int64_t " << kernelName
         << "_GetWorkspaceSize(const int64_t* shape_args, int32_t shape_count) {\n";
      os << "  if (shape_count != " << info.shapeFieldPositions.size();
      if (!info.shapeFieldPositions.empty())
        os << " || shape_args == nullptr";
      os << ")\n";
      os << "    return -1;\n";
      os << "  switch (" << kernelName
         << "_SelectScheduleEntry(shape_args)) {\n";
      for (const HostScheduleVariantInfo &variant : info.scheduleVariants) {
        os << "  case " << variant.index << ":\n";
        os << "    return " << variant.workspaceExpr << ";\n";
      }
      os << "  default:\n";
      os << "    return -1;\n";
      os << "  }\n";
      os << "}\n\n";
    }
    os << "} // extern \"C\"\n";
  });
}

} // namespace mlir::ascend::cann
