//===- CannRuntimeArtifacts.cpp - CANN runtime artifact emission ----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/CannKernel/CannRuntimeArtifacts.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

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

static SmallVector<func::FuncOp> collectGlobalKernels(ModuleOp module) {
  SmallVector<func::FuncOp> kernels;
  for (Operation &child : module.getBody()->getOperations())
    if (auto funcOp = dyn_cast<func::FuncOp>(child))
      if (funcOp->hasAttr(ascendc::attr::global))
        kernels.push_back(funcOp);
  return kernels;
}

static FailureOr<func::FuncOp> getSingleGlobalKernel(ModuleOp module,
                                                     StringRef artifactName) {
  SmallVector<func::FuncOp> kernels = collectGlobalKernels(module);
  if (kernels.size() != 1) {
    module.emitError() << artifactName
                       << " MVP supports exactly one global kernel";
    return failure();
  }
  return kernels.front();
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

static LogicalResult checkScheduleMetadataCompleteness(func::FuncOp funcOp) {
  bool hasSelectedTileShape =
      funcOp->hasAttr(::mlir::afir::ascend::kScheduleSelectedTileShapeAttr);
  bool hasTailPolicies =
      funcOp->hasAttr(::mlir::afir::ascend::kScheduleTailPoliciesAttr);
  bool hasTailPlan =
      funcOp->hasAttr(::mlir::afir::ascend::kScheduleTailPlanAttr);
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
  if (Attribute selectedTileShape =
          funcOp->getAttr(::mlir::afir::ascend::kScheduleSelectedTileShapeAttr))
    if (!isa<DenseI64ArrayAttr>(selectedTileShape))
      return funcOp.emitError()
             << ::mlir::afir::ascend::kScheduleSelectedTileShapeAttr
             << " must be a dense i64 array attribute";

  if (Attribute tailPolicies =
          funcOp->getAttr(::mlir::afir::ascend::kScheduleTailPoliciesAttr))
    if (!isa<ArrayAttr>(tailPolicies))
      return funcOp.emitError()
             << ::mlir::afir::ascend::kScheduleTailPoliciesAttr
             << " must be an array attribute";

  return success();
}

static FailureOr<llvm::json::Object>
buildScheduleTilingParams(func::FuncOp funcOp) {
  llvm::json::Object tilingParams;

  if (failed(checkScheduleMetadataCompleteness(funcOp)))
    return failure();
  if (failed(validateScheduleMetadataAttributes(funcOp)))
    return failure();

  if (auto selectedTileShape =
          funcOp->getAttrOfType<DenseI64ArrayAttr>(
              ::mlir::afir::ascend::kScheduleSelectedTileShapeAttr)) {
    llvm::json::Array selectedTileShapeJson;
    for (int64_t tileSize : selectedTileShape.asArrayRef())
      selectedTileShapeJson.push_back(tileSize);
    tilingParams["selected_tile_shape"] = std::move(selectedTileShapeJson);
  }

  if (auto tailPolicies =
          funcOp->getAttrOfType<ArrayAttr>(
              ::mlir::afir::ascend::kScheduleTailPoliciesAttr)) {
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

  if (Attribute rawTailPlanAttr =
          funcOp->getAttr(::mlir::afir::ascend::kScheduleTailPlanAttr)) {
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

  llvm::json::Object root;
  root["schema_version"] = "2.0";
  root["kernel"] = funcOr->getName().str();
  root["kernel_file"] = options.kernelFile.str();
  root["soc"] = options.soc.str();
  root["block_dim_expr"] = "20";
  root["workspace_size_expr"] = "0";
  root["tiling_params"] = buildTilingSchema(*fieldsOr);
  return writeJsonFile(module.getOperation(), outPath, std::move(root));
}

LogicalResult
emitRuntimeManifestJson(ModuleOp module, StringRef outPath,
                        const CannRuntimeArtifactOptions &options) {
  FailureOr<func::FuncOp> funcOr =
      getSingleGlobalKernel(module, "runtime manifest");
  if (failed(funcOr))
    return failure();
  FailureOr<emitasc::PyStructType> tilingTypeOr = getTilingType(*funcOr);
  if (failed(tilingTypeOr))
    return failure();
  FailureOr<SmallVector<TilingFieldInfo>> fieldsOr =
      collectTilingFields(*funcOr, *tilingTypeOr);
  if (failed(fieldsOr))
    return failure();

  llvm::json::Array shapeArgOrder;
  int64_t shapeAbiPosition = 0;
  for (auto [index, field] : llvm::enumerate(*fieldsOr)) {
    if (!field.isShape)
      continue;
    llvm::json::Object shapeArg;
    shapeArg["name"] = field.name;
    shapeArg["shapeKey"] = field.shapeKey;
    shapeArg["abiPosition"] = shapeAbiPosition++;
    shapeArgOrder.push_back(std::move(shapeArg));
  }

  FailureOr<llvm::json::Object> tilingParams =
      buildScheduleTilingParams(*funcOr);
  if (failed(tilingParams))
    return failure();
  FailureOr<llvm::json::Object> kernelEntryTilingParams =
      buildScheduleTilingParams(*funcOr);
  if (failed(kernelEntryTilingParams))
    return failure();

  llvm::json::Object scheduleEntry;
  scheduleEntry["decisionId"] = "static_0";
  scheduleEntry["guard"] = "true";
  scheduleEntry["tilingParams"] = std::move(*tilingParams);
  llvm::json::Array scheduleEntries;
  scheduleEntries.push_back(std::move(scheduleEntry));

  llvm::json::Object kernelNode;
  kernelNode["name"] = funcOr->getName().str();
  llvm::json::Array kernelNodes;
  kernelNodes.push_back(std::move(kernelNode));
  llvm::json::Object kernelGraph;
  kernelGraph["nodes"] = std::move(kernelNodes);
  kernelGraph["edges"] = llvm::json::Array{};

  llvm::json::Object kernelEntry;
  kernelEntry["kernel_id"] = funcOr->getName().str();
  kernelEntry["entry_index"] = 0;
  kernelEntry["tilingParams"] = std::move(*kernelEntryTilingParams);
  llvm::json::Array kernelEntries;
  kernelEntries.push_back(std::move(kernelEntry));

  llvm::json::Object root;
  root["kernelName"] = funcOr->getName().str();
  root["shapeBucketKey"] = "static";
  root["guardSet"] = llvm::json::Array{};
  root["tilingSchema"] = buildTilingSchema(*fieldsOr);
  root["scheduleEntries"] = std::move(scheduleEntries);
  root["abiSignature"] = (funcOr->getName() + ":cann_static").str();
  root["cacheKey"] =
      (funcOr->getName() + ":static:" + options.soc).str();
  root["workspaceSizeExpr"] = "0";
  root["workspaceSizeBytes"] = 0;
  root["shapeArgOrder"] = std::move(shapeArgOrder);
  root["kernelGraph"] = std::move(kernelGraph);
  root["kernel_entries"] = std::move(kernelEntries);
  return writeJsonFile(module.getOperation(), outPath, std::move(root));
}

LogicalResult emitHostTilingCpp(ModuleOp module, StringRef outPath,
                                const CannRuntimeArtifactOptions &options) {
  FailureOr<func::FuncOp> funcOr = getSingleGlobalKernel(module, "host tiling");
  if (failed(funcOr))
    return failure();
  FailureOr<emitasc::PyStructType> tilingTypeOr = getTilingType(*funcOr);
  if (failed(tilingTypeOr))
    return failure();

  auto types = tilingTypeOr->getTypesAttr().getValue();
  auto names = tilingTypeOr->getNamesAttr().getValue();
  if (types.size() != names.size())
    return funcOr->emitError("PyStructType types/names size mismatch");

  SmallVector<unsigned> shapeFieldPositions;
  for (auto [index, nameAttr] : llvm::enumerate(names)) {
    StringRef name = cast<StringAttr>(nameAttr).getValue();
    if (isShapeField(name))
      shapeFieldPositions.push_back(index);
  }

  return writeTextFile(module.getOperation(), outPath, [&](raw_ostream &os) {
    StringRef kernelName = funcOr->getName();
    os << "#include <cstdint>\n";
    os << "#include <cstring>\n\n";
    os << "struct TilingData {\n";
    for (auto [typeAttr, nameAttr] : llvm::zip(types, names)) {
      Type type = cast<TypeAttr>(typeAttr).getValue();
      StringRef name = cast<StringAttr>(nameAttr).getValue();
      os << "  " << getHostCppType(type) << " " << name << ";\n";
    }
    os << "};\n\n";
    os << "extern \"C\" {\n\n";
    os << "int32_t " << kernelName << "_GetTilingSize(void) {\n";
    os << "  return static_cast<int32_t>(sizeof(TilingData));\n";
    os << "}\n\n";

    os << "int32_t " << kernelName
       << "_GetTiling(const int64_t* shape_args, int32_t shape_count, "
          "void* tiling_out) {\n";
    os << "  if (shape_count != " << shapeFieldPositions.size()
       << " || tiling_out == nullptr";
    if (!shapeFieldPositions.empty())
      os << " || shape_args == nullptr";
    os << ")\n";
    os << "    return 1;\n";
    os << "  TilingData data{};\n";
    unsigned shapeIndex = 0;
    for (auto [index, nameAttr] : llvm::enumerate(names)) {
      StringRef name = cast<StringAttr>(nameAttr).getValue();
      os << "  data." << name << " = ";
      if (isShapeField(name))
        os << "shape_args[" << shapeIndex++ << "]";
      else
        os << "0";
      os << ";\n";
    }
    os << "  std::memcpy(tiling_out, &data, sizeof(TilingData));\n";
    os << "  return 0;\n";
    os << "}\n\n";

    os << "int64_t " << kernelName
       << "_GetBlockDim(const int64_t* shape_args, int32_t shape_count) {\n";
    os << "  (void)shape_args;\n";
    os << "  return shape_count == " << shapeFieldPositions.size()
       << " ? 20 : -1;\n";
    os << "}\n\n";

    os << "int64_t " << kernelName
       << "_GetWorkspaceSize(const int64_t* shape_args, int32_t shape_count) {\n";
    os << "  (void)shape_args;\n";
    os << "  return shape_count == " << shapeFieldPositions.size()
       << " ? 0 : -1;\n";
    os << "}\n\n";
    os << "} // extern \"C\"\n";
  });
}

} // namespace mlir::afir::cann
