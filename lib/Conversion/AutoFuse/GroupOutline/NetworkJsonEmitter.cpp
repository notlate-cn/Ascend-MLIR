//===- NetworkJsonEmitter.cpp - Emit network.json -------------------------===//
//
// Walks a coordinator func body, classifies callees, and writes network.json.
//
//===----------------------------------------------------------------------===//

#include "Conversion/AutoFuse/GroupOutline/NetworkJsonEmitter.h"
#include "Conversion/AutoFuse/GroupOutline/OpRoleClassifier.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstdio>

namespace mlir::auto_fuse {

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

    if (auto emptyOp = mlir::dyn_cast<mlir::tensor::EmptyOp>(&op)) {
      // A fresh DPS init/output buffer; the runner allocates it.
      llvm::json::Object src;
      src["from"] = "alloc";
      if (auto rt =
              mlir::dyn_cast<mlir::RankedTensorType>(emptyOp.getType())) {
        auto desc = tensorDescriptor(rt);
        src["shape"] = std::move(desc["shape"]);
        src["dtype"] = std::move(desc["dtype"]);
      }
      valueSource[emptyOp.getResult()] = std::move(src);
      continue;
    }

    if (auto sliceOp = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(&op)) {
      // A static sub-view of another value; the runner copies the slice out.
      llvm::json::Object src;
      src["from"] = "slice";
      if (auto it = valueSource.find(sliceOp.getSource());
          it != valueSource.end()) {
        llvm::json::Object srcCopy = it->second;
        src["source"] = std::move(srcCopy);
      }
      auto toArr = [](llvm::ArrayRef<int64_t> xs) {
        llvm::json::Array a;
        for (int64_t x : xs)
          a.push_back(mlir::ShapedType::isDynamic(x) ? int64_t{-1} : x);
        return a;
      };
      src["offsets"] = toArr(sliceOp.getStaticOffsets());
      src["sizes"]   = toArr(sliceOp.getStaticSizes());
      src["strides"] = toArr(sliceOp.getStaticStrides());
      if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(
              sliceOp.getResult().getType())) {
        auto desc = tensorDescriptor(rt);
        src["shape"] = std::move(desc["shape"]);
        src["dtype"] = std::move(desc["dtype"]);
      }
      valueSource[sliceOp.getResult()] = std::move(src);
      continue;
    }

    if (auto padOp = mlir::dyn_cast<mlir::tensor::PadOp>(&op)) {
      // Explicit pad ahead of a conv (torch.export emits one per padded conv).
      // Materialized host-side by AclnnBackend's CoordEmitter as a zero-filled
      // buffer with the source copied into the unpadded interior; here we just
      // register provenance so downstream kernel arg descriptors resolve.
      llvm::json::Object src;
      src["from"] = "pad";
      if (auto it = valueSource.find(padOp.getSource());
          it != valueSource.end()) {
        llvm::json::Object srcCopy = it->second;
        src["source"] = std::move(srcCopy);
      }
      auto toArr = [](llvm::ArrayRef<int64_t> xs) {
        llvm::json::Array a;
        for (int64_t x : xs)
          a.push_back(mlir::ShapedType::isDynamic(x) ? int64_t{-1} : x);
        return a;
      };
      src["low"]  = toArr(padOp.getStaticLow());
      src["high"] = toArr(padOp.getStaticHigh());
      // Best-effort static pad value (the yielded scalar in the pad region).
      // Almost always 0.0 for conv-padding.
      if (auto yieldOp = mlir::dyn_cast<mlir::tensor::YieldOp>(
              padOp.getBody()->getTerminator())) {
        if (auto cst = yieldOp.getValue()
                           .getDefiningOp<mlir::arith::ConstantOp>()) {
          if (auto fa = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
            // JSON spec lacks ±inf/NaN literals; emit a string fallback so
            // downstream parsers (Python json) don't choke.  Conv-padding
            // uses 0.0, maxpool-padding uses -inf.
            double dv = fa.getValueAsDouble();
            if (std::isfinite(dv))
              src["pad_value"] = dv;
            else if (std::isnan(dv))
              src["pad_value"] = "nan";
            else
              src["pad_value"] = (dv < 0) ? "-inf" : "inf";
          } else if (auto ia =
                         mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue())) {
            src["pad_value"] = static_cast<int64_t>(ia.getInt());
          }
        }
      }
      if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(
              padOp.getResult().getType())) {
        auto desc = tensorDescriptor(rt);
        src["shape"] = std::move(desc["shape"]);
        src["dtype"] = std::move(desc["dtype"]);
      }
      valueSource[padOp.getResult()] = std::move(src);
      continue;
    }

    if (auto cstOp = mlir::dyn_cast<mlir::arith::ConstantOp>(&op)) {
      // A constant used as a kernel arg (linalg.fill value, attention scale,
      // a weight tensor, ...).  Register it as a "const" source so downstream
      // call args resolve; the runner materializes it (scalar value inline,
      // tensor weight via its resource key).
      llvm::json::Object src;
      src["from"] = "const";
      mlir::Attribute val = cstOp.getValue();
      if (auto fa = mlir::dyn_cast<mlir::FloatAttr>(val)) {
        src["value"] = fa.getValueAsDouble();
        src["dtype"] = dtypeName(fa.getType());
      } else if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(val)) {
        src["value"] = static_cast<int64_t>(ia.getInt());
        src["dtype"] = dtypeName(ia.getType());
      } else if (auto rt =
                     mlir::dyn_cast<mlir::RankedTensorType>(cstOp.getType())) {
        auto desc = tensorDescriptor(rt);
        src["shape"] = std::move(desc["shape"]);
        src["dtype"] = std::move(desc["dtype"]);
        if (auto dr = mlir::dyn_cast<mlir::DenseResourceElementsAttr>(val))
          src["resource"] = dr.getRawHandle().getKey().str();
      }
      valueSource[cstOp.getResult()] = std::move(src);
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

    // A compile-time scalar (e.g. the `%c1` index feeding a tensor.dim, or a
    // shape literal): no runtime provenance needed.  Skip it — only ops that
    // produce a *tensor* or a dynamic-dim value that flows into a kernel call
    // need a descriptor.
    if (mlir::isa<mlir::arith::ConstantOp>(&op))
      continue;

    // A dynamic-dimension query on a coordinator value: record an "input_dim"
    // provenance so a kernel size-arg that consumes it resolves to the actual
    // runtime extent (the runner reads the source input's npy shape).  Without
    // this the outliner non-deterministically leaves tensor.dim in the
    // coordinator (vs sinking it into the kernel) and emission flakily failed
    // with "unsupported op in coordinator body: tensor.dim".
    if (auto dimOp = mlir::dyn_cast<mlir::tensor::DimOp>(&op)) {
      auto it = valueSource.find(dimOp.getSource());
      std::optional<int64_t> cdim = dimOp.getConstantIndex();
      if (it != valueSource.end() && cdim) {
        llvm::json::Object src;
        src["from"] = "input_dim";
        if (auto *nameV = it->second.get("name"))
          src["name"] = *nameV;
        src["dim"] = *cdim;
        valueSource[dimOp.getResult()] = std::move(src);
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

//===----------------------------------------------------------------------===//
// emitNetworkProvenanceJson (debug-only sidecar)
//===----------------------------------------------------------------------===//

static std::string padId(unsigned n) {
  char buf[8];
  std::snprintf(buf, sizeof(buf), "op_%03u", n);
  return std::string(buf);
}

static std::string locToString(mlir::Location loc) {
  std::string s;
  llvm::raw_string_ostream os(s);
  loc.print(os);
  return s;
}

// Trace through trivial DPS / view / glue ops back to the producing source op.
static mlir::Operation *traceToSourceOp(mlir::Value v) {
  mlir::Operation *def = v.getDefiningOp();
  while (def && llvm::isa<mlir::tensor::CastOp, mlir::tensor::ExtractSliceOp,
                          mlir::tensor::ExpandShapeOp,
                          mlir::tensor::CollapseShapeOp>(def)) {
    def = def->getOperand(0).getDefiningOp();
  }
  return def;
}

llvm::Error emitNetworkProvenanceJson(mlir::ModuleOp module,
                                      mlir::func::FuncOp coord,
                                      llvm::raw_ostream &os) {
  mlir::SymbolTable symTable(module);

  llvm::json::Array kernelsArr;
  unsigned opCounter = 0;

  for (mlir::Operation &op : coord.getBody().front()) {
    auto callOp = mlir::dyn_cast<mlir::func::CallOp>(&op);
    if (!callOp)
      continue;

    llvm::StringRef calleeName = callOp.getCallee();
    auto callee = symTable.lookup<mlir::func::FuncOp>(calleeName);
    bool isAclnn = callee && callee->hasAttr("aclnn.op");

    llvm::json::Object entry;
    entry["kernel_id"] = calleeName.str();
    entry["kind"] = isAclnn ? "aclnn" : "ascendc";
    entry["runtime_task"] = calleeName.str();

    llvm::json::Array sourceOps;
    llvm::json::Array boundary;
    std::string summary;

    if (isAclnn) {
      // Synthetic single source op from aclnn.op attr.
      std::string opName = "unknown";
      if (auto a = callee->getAttrOfType<mlir::StringAttr>("aclnn.op"))
        opName = a.getValue().str();
      std::string id = padId(opCounter++);
      llvm::json::Object so;
      so["id"] = id;
      so["name"] = "aclnn." + opName;
      so["op_role"] = opName;
      so["loc"] = locToString(callOp.getLoc());
      so["result_ssa"] = "%0";
      sourceOps.push_back(std::move(so));
      summary = opName;
      for (unsigned i = 0, e = callOp.getNumResults(); i < e; ++i) {
        llvm::json::Object b;
        b["result_index"] = static_cast<int64_t>(i);
        b["source_op_id"] = id;
        boundary.push_back(std::move(b));
      }
    } else if (!callee || callee.isExternal() || callee.getBody().empty()) {
      // Declaration / external / no body — emit a placeholder.
      std::string id = padId(opCounter++);
      llvm::json::Object so;
      so["id"] = id;
      so["name"] = callee ? callee.getName().str() : std::string("<extern>");
      so["op_role"] = "unknown";
      so["loc"] = locToString(callOp.getLoc());
      sourceOps.push_back(std::move(so));
      summary = "unknown";
    } else {
      // AscendC: walk the callee body for source ops (linalg.LinalgOp).
      mlir::AsmState asmState(callee);
      llvm::DenseMap<mlir::Operation *, std::string> idOf;
      for (mlir::Operation &kop : callee.getBody().front()) {
        if (!mlir::isa<mlir::linalg::LinalgOp>(&kop))
          continue;
        std::string id = padId(opCounter++);
        idOf[&kop] = id;
        std::string role = classifyOpRole(&kop);
        llvm::json::Object so;
        so["id"] = id;
        so["name"] = kop.getName().getStringRef().str();
        so["op_role"] = role;
        so["loc"] = locToString(kop.getLoc());
        // result_ssa: capture the printed SSA name of the first result.
        std::string ssaName;
        if (kop.getNumResults() > 0) {
          llvm::raw_string_ostream ss(ssaName);
          kop.getResult(0).printAsOperand(ss, asmState);
        }
        so["result_ssa"] = ssaName;
        sourceOps.push_back(std::move(so));
        if (!summary.empty()) summary += "+";
        summary += role;
      }
      // Build boundary: each callee return operand traces back to a source op.
      auto retOp = mlir::dyn_cast<mlir::func::ReturnOp>(
          callee.getBody().front().getTerminator());
      if (retOp) {
        for (auto [i, v] : llvm::enumerate(retOp.getOperands())) {
          mlir::Operation *src = traceToSourceOp(v);
          auto it = src ? idOf.find(src) : idOf.end();
          llvm::json::Object b;
          b["result_index"] = static_cast<int64_t>(i);
          b["source_op_id"] = (it != idOf.end()) ? it->second : std::string("unknown");
          boundary.push_back(std::move(b));
        }
      }
    }

    // Convention from HostLaunchHelper's dump-intermediates layout.
    llvm::json::Array hints;
    for (unsigned i = 0, e = callOp.getNumResults(); i < e; ++i)
      hints.push_back(
          ("intermediates_default/" + calleeName + "_out_" + llvm::Twine(i) +
           ".npy").str());

    entry["fused_ops_summary"] = summary;
    entry["source_ops"] = std::move(sourceOps);
    entry["boundary_source_ops"] = std::move(boundary);
    entry["output_checkpoint_hint"] = std::move(hints);
    entry["torch_hint"] = nullptr;

    kernelsArr.push_back(std::move(entry));
  }

  llvm::json::Object root;
  root["schema_version"] = 1;
  root["tool"] = "NetworkJsonEmitter";
  // Descriptive label only; source op order is stable per-emit but the
  // referenced "stage" is the post-outline coordinator (which preserves
  // post-recognize-aclnn linalg op identity inside private kernel funcs).
  root["stage_input"] = "post-recognize-aclnn-outlined";
  root["function"] = coord.getName().str();
  root["kernels"] = std::move(kernelsArr);

  llvm::json::OStream jos(os, /*IndentSize=*/2);
  jos.value(llvm::json::Value(std::move(root)));
  os << "\n";

  return llvm::Error::success();
}

} // namespace mlir::auto_fuse
