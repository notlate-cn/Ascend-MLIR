//===- NetworkJsonEmitter.cpp - Emit network.json -------------------------===//
//
// Walks a coordinator func body, classifies callees, and writes network.json.
//
//===----------------------------------------------------------------------===//

#include "Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.h"
#include "Conversion/VectorPlan/TilePlan.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::vector_plan {

//===----------------------------------------------------------------------===//
// Dtype name helper
//===----------------------------------------------------------------------===//

static std::string dtypeName(mlir::Type t) {
  if (t.isF16())    return "f16";
  if (t.isBF16())   return "bf16";
  if (t.isF32())    return "f32";
  if (t.isInteger(8))  return "int8";
  if (t.isInteger(32)) return "int32";
  if (t.isInteger(64)) return "int64";
  return "unknown";
}

//===----------------------------------------------------------------------===//
// Shape/dtype extraction from a RankedTensorType
//===----------------------------------------------------------------------===//

static llvm::json::Object tensorDescriptor(mlir::RankedTensorType ty) {
  llvm::json::Array shape;
  for (int64_t d : ty.getShape())
    shape.push_back(mlir::ShapedType::isDynamic(d) ? int64_t{-1} : d);
  llvm::json::Object desc;
  desc["shape"] = std::move(shape);
  desc["dtype"] = dtypeName(ty.getElementType());
  return desc;
}

//===----------------------------------------------------------------------===//
// emitNetworkJson
//===----------------------------------------------------------------------===//

llvm::Error emitNetworkJson(mlir::ModuleOp module, mlir::func::FuncOp coord,
                            llvm::raw_ostream &os) {
  mlir::SymbolTable symTable(module);

  // Map from SSA Value → its source descriptor (partial json::Object).
  // For network inputs: {"from":"input","name":"argN"}
  // For kernel results: {"from":"kernel","kernel":"<id>","result":N}
  llvm::DenseMap<mlir::Value, llvm::json::Object> valueSource;

  // Seed with coordinator arguments.
  llvm::json::Array inputsArr;
  for (auto [idx, arg] : llvm::enumerate(coord.getArguments())) {
    std::string argName = ("arg" + llvm::Twine(idx)).str();
    llvm::json::Object src;
    src["from"] = "input";
    src["name"] = argName;
    valueSource[arg] = src;

    // Only emit tensor-typed args as network inputs (skip non-tensor).
    auto ty = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
    if (!ty)
      continue;
    auto desc = tensorDescriptor(ty);
    desc["name"] = argName;
    inputsArr.push_back(std::move(desc));
  }

  llvm::json::Array kernelsArr;
  llvm::json::Array outputsArr;

  for (mlir::Operation &op : coord.getBody().front()) {
    if (auto castOp = mlir::dyn_cast<mlir::tensor::CastOp>(&op)) {
      // Alias: propagate source descriptor to the cast result.
      auto it = valueSource.find(castOp.getSource());
      if (it != valueSource.end())
        valueSource[castOp.getResult()] = it->second;
      continue;
    }
    if (auto expandOp = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(&op)) {
      // Same-buffer view alias (used by SplitRCoreGroup to chunk an R-axis
      // input into [N, K] before calling the partial kernel).  Propagate the
      // source descriptor BUT override its shape/dtype with the expanded
      // view's so downstream kernel arg descriptors and the network_runner
      // shape-key resolver see the rank that the kernel actually consumes.
      auto it = valueSource.find(expandOp.getSrc());
      if (it != valueSource.end()) {
        llvm::json::Object src = it->second;
        if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(
                expandOp.getResult().getType())) {
          auto desc = tensorDescriptor(rt);
          src["shape"] = std::move(desc["shape"]);
          src["dtype"] = std::move(desc["dtype"]);
        }
        valueSource[expandOp.getResult()] = std::move(src);
      }
      continue;
    }
    if (auto collapseOp = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(&op)) {
      auto it = valueSource.find(collapseOp.getSrc());
      if (it != valueSource.end()) {
        llvm::json::Object src = it->second;
        if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(
                collapseOp.getResult().getType())) {
          auto desc = tensorDescriptor(rt);
          src["shape"] = std::move(desc["shape"]);
          src["dtype"] = std::move(desc["dtype"]);
        }
        valueSource[collapseOp.getResult()] = std::move(src);
      }
      continue;
    }

    if (auto retOp = mlir::dyn_cast<mlir::func::ReturnOp>(&op)) {
      for (auto [idx, operand] : llvm::enumerate(retOp.getOperands())) {
        llvm::json::Object outDesc;
        outDesc["name"] = ("out" + llvm::Twine(idx)).str();
        auto it = valueSource.find(operand);
        if (it == valueSource.end())
          return llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "emitNetworkJson: return operand %u of %s has no provenance "
              "descriptor",
              static_cast<unsigned>(idx),
              coord.getName().str().c_str());
        // Copy source fields into outDesc.
        for (auto &[k, v] : it->second)
          outDesc[k] = v;
        outputsArr.push_back(std::move(outDesc));
      }
      continue;
    }

    if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(&op)) {
      llvm::StringRef calleeName = callOp.getCallee();

      // Look up callee to check for aclnn.op attr.
      auto callee = symTable.lookup<mlir::func::FuncOp>(calleeName);
      bool isAclnn = callee && callee->hasAttr("aclnn.op");
      std::string kind = isAclnn ? "aclnn" : "ascendc";

      // Build args array.
      llvm::json::Array argsArr;
      for (auto [argIdx, operand] : llvm::enumerate(callOp.getOperands())) {
        auto it = valueSource.find(operand);
        if (it == valueSource.end())
          return llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "emitNetworkJson: call arg %u of %s has no provenance descriptor",
              static_cast<unsigned>(argIdx),
              calleeName.str().c_str());
        llvm::json::Object argDesc;
        for (auto &[k, v] : it->second)
          argDesc[k] = v;
        argsArr.push_back(std::move(argDesc));
      }

      // Build results array and register results in valueSource.
      llvm::json::Array resultsArr;
      for (auto [rIdx, result] : llvm::enumerate(callOp.getResults())) {
        std::string resName =
            (calleeName + "_r" + llvm::Twine(rIdx)).str();
        llvm::json::Object resDesc;
        resDesc["name"] = resName;
        auto ty = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
        if (ty) {
          llvm::json::Array shape;
          for (int64_t d : ty.getShape())
            shape.push_back(mlir::ShapedType::isDynamic(d) ? int64_t{-1} : d);
          resDesc["shape"] = std::move(shape);
          resDesc["dtype"] = dtypeName(ty.getElementType());
        }
        resultsArr.push_back(std::move(resDesc));

        // Register source descriptor for downstream consumers.
        llvm::json::Object src;
        src["from"] = "kernel";
        src["kernel"] = calleeName.str();
        src["result"] = static_cast<int64_t>(rIdx);
        valueSource[result] = std::move(src);
      }

      // Build kernel entry.
      llvm::json::Object kernelEntry;
      kernelEntry["id"]      = calleeName.str();
      kernelEntry["kind"]    = kind;
      if (!isAclnn)
        kernelEntry["file"] = (calleeName + ".mlir").str();
      if (isAclnn) {
        if (auto opAttr = callee->getAttrOfType<mlir::StringAttr>("aclnn.op"))
          kernelEntry["op"] = opAttr.getValue().str();
        if (auto layoutAttr =
                callee->getAttrOfType<mlir::StringAttr>("aclnn.layout"))
          kernelEntry["layout"] = layoutAttr.getValue().str();
      }
      // v2 schema: also emit a full per-MLIR-arg table for the runner.
      if (callee) {
        if (auto schema = mlir::vector_plan::lookupTilingInfoSchema(
                module, calleeName)) {
          llvm::json::Array schArgs;
          for (auto &a : schema->args) {
            llvm::json::Object e;
            e["mlir_index"] = static_cast<int64_t>(a.mlirIndex);
            switch (a.role) {
              case mlir::vector_plan::SchemaArgRole::Input:
                e["role"] = "input";
                e["network_index"] = static_cast<int64_t>(a.networkIndex);
                break;
              case mlir::vector_plan::SchemaArgRole::Output: {
                e["role"] = "output";
                e["result_index"] = static_cast<int64_t>(a.resultIndex);
                llvm::json::Array se;
                for (auto &s : a.shapeExpr) se.push_back(s);
                e["shape_expr"] = std::move(se);
                break;
              }
              case mlir::vector_plan::SchemaArgRole::TileParam:
                e["role"] = "tile_param";
                e["name"] = a.tileParamName;
                break;
              case mlir::vector_plan::SchemaArgRole::Workspace:
                e["role"] = "workspace";
                break;
              case mlir::vector_plan::SchemaArgRole::TilingDataStruct:
                e["role"] = "tiling_data_struct";
                break;
            }
            schArgs.push_back(std::move(e));
          }
          kernelEntry["schema_args"] = std::move(schArgs);
        }
      }
      kernelEntry["args"]    = std::move(argsArr);
      kernelEntry["results"] = std::move(resultsArr);
      kernelsArr.push_back(std::move(kernelEntry));
      continue;
    }

    // Anything else is unsupported.
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "emitNetworkJson: unsupported op in coordinator body: %s",
        op.getName().getStringRef().str().c_str());
  }

  // Assemble root object.
  llvm::json::Object root;
  root["function"] = coord.getName().str();
  root["inputs"]   = std::move(inputsArr);
  root["kernels"]  = std::move(kernelsArr);
  root["outputs"]  = std::move(outputsArr);

  llvm::json::OStream jos(os, /*IndentSize=*/2);
  jos.value(llvm::json::Value(std::move(root)));
  os << "\n";

  return llvm::Error::success();
}

} // namespace mlir::vector_plan
