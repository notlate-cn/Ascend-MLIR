#include "TilePlanGen.h"
#include "TileFuseUtils.h"
#include "TilePlanGenInternal.h"
#include "Analysis/SymbolicShape/DimSymbolTable.h"
#include "Analysis/SymbolicShape/SymExpr.h"
#include "Conversion/AutoFuse/TilePlan.h"
#include "Target/CannKernel/SocSpec.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <limits>

#define DEBUG_TYPE "tile-plan-gen"

using namespace mlir;
using namespace mlir::auto_fuse;

namespace mlir::afir {

// ===========================================================================
// This file is the "schedule" stage of auto-fuse-tile-fuse.  It mirrors
// AutoFuse's optimize/autoschedule (see
// docs/superpowers/plans/2026-05-11-port-af-scheduler-to-auto-fuse.zh.md):
//
//   genVectorTilePlan                     ≈ AutoSchedule::DoAutoSchedule
//     ├─ (info.grouping)                  ≈ TilingGroup::GenTilingGroup  (X/Y/R/N)
//     ├─ enumerateTilingCases  → [draft]  ≈ GenTilingCase + PruneTilingCase
//     ├─ for each draft: costEstimate     ≈ score_func
//     ├─ pickBest = argmin                ≈ argmin score
//     └─ buildPlan(winner)                ≈ Scheduler::BlockSplit + TileSplit
//                                            (+ computeVectorizedDims, §3.5 row-loop
//                                             degradation)
//
// Status: P1 (refactor) + P3a (per-operand `vectorizedDims` + the §3.5 row-loop
// degradation) + P5a (enumerate/build/cost/pick skeleton) + P5b (real
// reduce-axis enumeration: `enumerateTilingCases` walks the `ubR` candidates
// — R kept whole, or an oversized / `--enable-reduction-split` R axis ub-split,
// plus an ∞-scored RCore variant — `costEstimate` rejects the infeasible ones,
// `buildPlan` consumes `draft.ubTilingAxisR`; the §3.4 change that a non-ub
// parallel axis is fully loaded instead of getting its own `XBLOCK_SUB_n`).
// Still pending: ubY/ubX enumeration over the rest of the y/x groups (needs
// LoopNestBuilder support for a non-block ub axis), the RCore/FullLoad reduce
// templates' codegen (their drafts are enumerated but `costEstimate` keeps them
// ∞ until the codegen lands), a real cost model (§3.6 blockDim / vectorized
// bytes — comes with P6's UB-peak accounting), and the transpose templates
// score (§5).  See the plan doc.
// ===========================================================================




// Populate `schema.fields` with Tunable entries for every tile-param block
// argument in `plan.tileable`, in the order they appear (matches the existing
// TilingData struct layout).
static void buildSchemaTunableFields(const TilePlan &plan, func::FuncOp func,
                                     TilingInfoSchema &schema) {
  for (auto &group : plan.tileable) {
    for (const auto &tp : group) {
      auto ba = dyn_cast<BlockArgument>(tp.ssa);
      if (!ba) continue;
      int64_t defaultVal = 0;
      if (auto attr = func.getArgAttrOfType<IntegerAttr>(
              ba.getArgNumber(), "auto_fuse.default_tile_size"))
        defaultVal = attr.getInt();
      int64_t axisSize = -1;
      if (plan.group && tp.axisIdx >= 0 &&
          tp.axisIdx < (int)plan.group->collapsedAxes.size()) {
        int64_t s = plan.group->collapsedAxes[tp.axisIdx].staticSize;
        if (s != ShapedType::kDynamic) axisSize = s;
      }
      assert(ba.getArgNumber() <= (unsigned)INT32_MAX && "arg_index overflow");
      SchemaField f;
      f.name         = tp.name;
      f.kind         = SchemaFieldKind::Tunable;
      f.axisSize     = axisSize;
      f.defaultValue = defaultVal;
      f.argIndex     = (int32_t)ba.getArgNumber();
      schema.fields.push_back(std::move(f));
    }
  }
}

// Populate `schema.args` from the kernel func signature: input / tile-param /
// workspace from block arguments, plus synthetic Output entries from the
// `func.return` operands (pre-bufferize the outputs are not yet args).
static void buildSchemaArgs(func::FuncOp func, TilingInfoSchema &schema) {
  Block &entry = func.getBody().front();
  unsigned numNetworkInputs = 0;
  for (BlockArgument ba : entry.getArguments()) {
    SchemaArg a;
    a.mlirIndex = (int32_t)ba.getArgNumber();
    Type ty = ba.getType();
    if (auto mt = dyn_cast<MemRefType>(ty)) {
      // Workspace heuristic: post-bufferize the workspace memref is always
      // memref<...xi8> (added by canonicalize-cann-signature).  Kernels with
      // genuine i8 user data (quantized inputs) would currently mis-classify
      // here; revisit when such a kernel lands by gating on an explicit arg
      // attribute stamped by the workspace-allocation pass.
      if (mt.getElementType().isInteger(8)) {
        a.role = SchemaArgRole::Workspace;
      } else if (mt.getLayout().isIdentity()) {
        a.role = SchemaArgRole::Input;
        if (auto attr = func.getArgAttrOfType<IntegerAttr>(
                ba.getArgNumber(), "auto_fuse.call_arg_index"))
          a.callArgIndex = (int32_t)attr.getInt();
        else
          a.callArgIndex = (int32_t)numNetworkInputs;
        ++numNetworkInputs;
      } else {
        // TODO multi-result: resultIndex hard-coded to 0; revisit when
        // kernels with >1 DPS output exist.
        a.role = SchemaArgRole::Output;
        a.resultIndex = 0;
      }
    } else if (isa<RankedTensorType>(ty)) {
      // Pre-bufferize: every tensor arg is an input (DPS init tensors are
      // tensor.empty results, not func args).
      a.role = SchemaArgRole::Input;
      if (auto attr = func.getArgAttrOfType<IntegerAttr>(
              ba.getArgNumber(), "auto_fuse.call_arg_index"))
        a.callArgIndex = (int32_t)attr.getInt();
      else
        a.callArgIndex = (int32_t)numNetworkInputs;
      ++numNetworkInputs;
    } else if (isa<IndexType>(ty)) {
      a.role = SchemaArgRole::TileParam;
      for (auto &f : schema.fields)
        if (f.kind == SchemaFieldKind::Tunable &&
            f.argIndex == a.mlirIndex)
          a.tileParamName = f.name;
    } else {
      continue; // unknown arg type — skip
    }
    schema.args.push_back(std::move(a));
  }

  // Synthesize Output SchemaArg entries from func return values.  Pre-
  // bufferize the kernel returns tensors; one-shot-bufferize will later
  // append a corresponding output memref arg per result at position
  // numArgs + i.  We pre-assign that anticipated mlirIndex so post-
  // bufferize consumers can index into the (now-larger) arg list.
  // TODO multi-result: works correctly today (one Output per return).
  if (entry.empty()) return;
  auto retOp = dyn_cast<func::ReturnOp>(entry.getTerminator());
  if (!retOp) return;
  // Skip if we already classified a real Output arg above
  // (post-bufferize path).  Invariant across iterations — the loop only
  // appends new Outputs, so checking once before the loop suffices.
  bool alreadyHaveOutput = llvm::any_of(schema.args, [&](auto &x) {
    return x.role == SchemaArgRole::Output;
  });
  if (alreadyHaveOutput) return;
  unsigned baseIdx = entry.getNumArguments();
  for (auto [i, v] : llvm::enumerate(retOp.getOperands())) {
    if (!isa<RankedTensorType, MemRefType>(v.getType())) continue;
    SchemaArg a;
    a.mlirIndex   = (int32_t)(baseIdx + i);
    a.role        = SchemaArgRole::Output;
    a.resultIndex = (int32_t)i;
    schema.args.push_back(std::move(a));
  }
}

// For each Output SchemaArg, populate `shapeExpr[]` per output dim using the
// symbolic-shape table for real args, or by tracing the synthetic-output
// linalg.generic → tensor.empty → tensor.dim chain back to an Input arg.
static void computeOutputShapeExprs(
    func::FuncOp func, TilingInfoSchema &schema,
    const std::optional<symshape::DimSymbolTable> &symTable) {
  Block &entry = func.getBody().front();

  auto tryEmitShapeExpr = [&](SchemaArg &a, ShapedType st,
                              StringAttr symAttr) {
    auto symList = symAttr ? symshape::parseSymExprList(symAttr.getValue())
                           : std::nullopt;
    for (int64_t d = 0, e = st.getRank(); d < e; ++d) {
      std::string expr;
      if (symList && (size_t)d < symList->size() && symTable) {
        const auto &se = (*symList)[d];
        if (se.getKind() == symshape::SymExpr::Kind::Sym) {
          auto src = symTable->sourceOf(se.getSym());
          for (auto &ia : schema.args)
            if (ia.role == SchemaArgRole::Input &&
                (unsigned)ia.mlirIndex == src.first) {
              expr = "arg" + std::to_string(ia.callArgIndex) +
                     "_dim" + std::to_string(src.second);
              break;
            }
        }
      }
      if (expr.empty() && d < (int64_t)st.getShape().size() &&
          !ShapedType::isDynamic(st.getShape()[d]))
        expr = std::to_string(st.getShape()[d]);
      a.shapeExpr.push_back(expr);
    }
  };

  // Helper for synthetic-output dynamic dims: trace
  //   retOp.getOperand(ri) → linalg.generic → tensor.empty(%d0, %d1, ...)
  //   → tensor.dim %argN, %cIdx  where %argN is an Input schema arg.
  // Render "arg<callArgIndex>_dim<dimIdx>" on success.  The plan spec §6
  // explicitly defers anything more involved (reduce/matmul) to later.
  auto resolveSyntheticDynDim =
      [&](Value retVal, int64_t outDim) -> std::string {
    auto genericOp = retVal.getDefiningOp<linalg::GenericOp>();
    if (!genericOp || genericOp.getOutputs().empty()) {
      LLVM_DEBUG(llvm::dbgs() << "[tiling-info] synthetic output dim "
                              << outDim << ": no defining linalg.generic\n");
      return {};
    }
    auto emptyOp = genericOp.getOutputs()[0].getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp) {
      LLVM_DEBUG(llvm::dbgs() << "[tiling-info] synthetic output dim "
                              << outDim << ": init not tensor.empty\n");
      return {};
    }
    auto mixed = emptyOp.getMixedSizes();
    if ((size_t)outDim >= mixed.size()) return {};
    auto ofr = mixed[outDim];
    auto val = dyn_cast<Value>(ofr);
    if (!val) return {};
    auto dimOp = val.getDefiningOp<tensor::DimOp>();
    if (!dimOp) {
      LLVM_DEBUG(llvm::dbgs() << "[tiling-info] synthetic output dim "
                              << outDim << ": dyn size not tensor.dim\n");
      return {};
    }
    auto ba = dyn_cast<BlockArgument>(dimOp.getSource());
    if (!ba || ba.getOwner() != &entry) return {};
    auto cst = dimOp.getIndex().getDefiningOp<arith::ConstantOp>();
    if (!cst) return {};
    auto intAttr = dyn_cast<IntegerAttr>(cst.getValue());
    if (!intAttr) return {};
    int32_t dimIdx = (int32_t)intAttr.getValue().getSExtValue();
    for (auto &ia : schema.args) {
      if (ia.role == SchemaArgRole::Input &&
          (unsigned)ia.mlirIndex == ba.getArgNumber()) {
        return "arg" + std::to_string(ia.callArgIndex) +
               "_dim" + std::to_string(dimIdx);
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "[tiling-info] synthetic output dim "
                            << outDim << ": no matching Input arg\n");
    return {};
  };

  for (auto &a : schema.args) {
    if (a.role != SchemaArgRole::Output) continue;
    ShapedType st;
    StringAttr symAttr;
    Value retVal;
    bool synthetic = false;
    if ((unsigned)a.mlirIndex < entry.getNumArguments()) {
      // Real arg (post-bufferize path).
      st = cast<ShapedType>(entry.getArgument(a.mlirIndex).getType());
      symAttr = func.getArgAttrOfType<StringAttr>(
          a.mlirIndex, "afir.symbolic_shape");
    } else if (auto retOp =
                   dyn_cast<func::ReturnOp>(entry.getTerminator())) {
      // Synthetic output: use the corresponding return value type.
      unsigned ri = (unsigned)a.resultIndex;
      if (ri < retOp.getNumOperands()) {
        retVal = retOp.getOperand(ri);
        st = dyn_cast<ShapedType>(retVal.getType());
        synthetic = true;
      }
    }
    if (!st) continue;
    if (!synthetic) {
      tryEmitShapeExpr(a, st, symAttr);
      continue;
    }
    // Synthetic-output branch: symAttr is unavailable.  Use static dims
    // directly; for dynamic dims, trace the linalg.generic → tensor.empty
    // → tensor.dim chain back to an Input schema arg.
    for (int64_t d = 0, e = st.getRank(); d < e; ++d) {
      std::string expr;
      if (!ShapedType::isDynamic(st.getShape()[d])) {
        expr = std::to_string(st.getShape()[d]);
      } else {
        expr = resolveSyntheticDynDim(retVal, d);
      }
      a.shapeExpr.push_back(expr);
    }
  }
}

// Walk memref.dim / tensor.dim ops in the kernel body, dedup via (argN, dimIdx),
// and append ShapeDerived SchemaField entries for each unique source dim.
static void collectShapeDerivedFields(func::FuncOp func,
                                      TilingInfoSchema &schema) {
  Block &entry = func.getBody().front();
  llvm::DenseSet<std::pair<int32_t, int32_t>> seenDims;
  auto recordDim = [&](BlockArgument ba, int32_t dimI) {
    if (!ba || ba.getOwner() != &entry) return;
    int32_t argN = (int32_t)ba.getArgNumber();
    // Skip if already a tunable on the same MLIR arg (won't normally
    // collide -- tunables are index-typed, dim sources are shaped -- but
    // dedup is cheap insurance).
    for (auto &f : schema.fields)
      if (f.kind == SchemaFieldKind::Tunable &&
          f.argIndex == argN)
        return;
    if (!seenDims.insert({argN, dimI}).second) return;
    SchemaField f;
    f.name      = "dim_arg" + std::to_string(argN) + "_" + std::to_string(dimI);
    f.kind      = SchemaFieldKind::ShapeDerived;
    f.sourceArg = argN;
    f.sourceDim = dimI;
    schema.fields.push_back(std::move(f));
  };
  func.walk([&](Operation *op) {
    if (auto dimOp = dyn_cast<memref::DimOp>(op)) {
      auto ba = dyn_cast<BlockArgument>(dimOp.getSource());
      if (auto cst = dimOp.getIndex().getDefiningOp<arith::ConstantOp>())
        if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
          recordDim(ba, (int32_t)intAttr.getValue().getSExtValue());
    } else if (auto dimOp = dyn_cast<tensor::DimOp>(op)) {
      auto ba = dyn_cast<BlockArgument>(dimOp.getSource());
      if (auto cst = dimOp.getIndex().getDefiningOp<arith::ConstantOp>())
        if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
          recordDim(ba, (int32_t)intAttr.getValue().getSExtValue());
    }
  });
}

// Group Input args' shape dims by their shared shape-symbol root, so the
// runtime can validate that user-passed input shapes are mutually consistent
// (e.g. for `a + b` with a,b both `?x?x?`, a.dim_i must equal b.dim_i).
// Reads `afir.dim_symbols` (func attr) + per-arg `afir.symbolic_shape`.
// Skips constant entries (broadcast `1`) and unrecognized SymExpr forms.
static void collectShapeEqualities(func::FuncOp func,
                                   TilingInfoSchema &schema) {
  auto dimSymsAttr = func->getAttrOfType<ArrayAttr>("afir.dim_symbols");
  if (!dimSymsAttr) return;
  auto symTable = symshape::DimSymbolTable::fromAttr(dimSymsAttr);
  if (!symTable) return;

  // Group (callArgIndex, dim) by the (rootArg, rootDim) the table resolves
  // their SymId to.  Composite key via DenseMap<pair<unsigned,unsigned>>.
  llvm::DenseMap<std::pair<unsigned, unsigned>,
                 llvm::SmallVector<std::pair<int32_t, int32_t>, 4>>
      groups;
  for (auto &a : schema.args) {
    if (a.role != SchemaArgRole::Input) continue;
    if (a.callArgIndex < 0) continue;
    auto symAttr = func.getArgAttrOfType<StringAttr>(
        (unsigned)a.mlirIndex, "afir.symbolic_shape");
    if (!symAttr) continue;
    auto list = symshape::parseSymExprList(symAttr.getValue());
    if (!list) continue;
    for (int32_t d = 0; d < (int32_t)list->size(); ++d) {
      const auto &e = (*list)[d];
      if (e.getKind() != symshape::SymExpr::Kind::Sym) continue; // skip consts
      if (e.getSym() >= symTable->numRoots()) continue;
      auto src = symTable->sourceOf(e.getSym());
      groups[{src.first, src.second}].push_back({a.callArgIndex, d});
    }
  }

  // Stable output order: sort group keys.
  llvm::SmallVector<std::pair<unsigned, unsigned>, 8> keys;
  for (auto &kv : groups) keys.push_back(kv.first);
  llvm::sort(keys);
  for (auto &k : keys) {
    auto &group = groups[k];
    if (group.size() < 2) continue; // singleton: nothing to validate
    schema.shapeEqualities.push_back(std::move(group));
  }
}

// Serialize TilePlan constraints into the existing {kind, lhs, rhs} dict
// array; returns a null ArrayAttr when there are no constraints.
static ArrayAttr serializeConstraints(MLIRContext *ctx,
                                      ArrayRef<TileConstraint> constraints) {
  if (constraints.empty()) return ArrayAttr();
  SmallVector<Attribute> cs;
  for (auto &c : constraints) {
    NamedAttrList ca;
    ca.append("kind",
              StringAttr::get(ctx, c.kind == TileConstraint::Divides
                                       ? "divides" : "le_bytes"));
    ca.append("lhs", StringAttr::get(ctx, c.lhs));
    ca.append("rhs", StringAttr::get(ctx, c.rhs));
    cs.push_back(ca.getDictionary(ctx));
  }
  return ArrayAttr::get(ctx, cs);
}

// Append the schema dict to the module-level `auto_fuse.tiling_infos`
// ArrayAttr, preserving any existing entries.
static void serializeAndAttach(ModuleOp moduleOp, const TilingInfoSchema &schema,
                               ArrayAttr constraintsAttr) {
  MLIRContext *ctx = moduleOp.getContext();
  DictionaryAttr entryDict =
      serializeTilingInfoSchema(ctx, schema, constraintsAttr);
  StringRef attrName = "auto_fuse.tiling_infos";
  SmallVector<Attribute> infos;
  if (auto existing = moduleOp->getAttrOfType<ArrayAttr>(attrName))
    llvm::append_range(infos, existing.getValue());
  infos.push_back(entryDict);
  moduleOp->setAttr(attrName, ArrayAttr::get(ctx, infos));
}

void emitTilingInfos(func::FuncOp func, const TilePlan &plan) {
  MLIRContext *ctx = func.getContext();
  auto moduleOp = func->getParentOfType<ModuleOp>();
  if (!moduleOp) return;

  // block_dim_expr: a SymExpr string `ceil(<block axis extent>/XBLOCK)` with the
  // block axis extent rendered in `argN_dimD` shape-key terms (the autotuner's
  // var names).  Built from afir.axis_extents / afir.dim_symbols (set by
  // afir-symbolize-shapes + Collapse) -- absent when those aren't available
  // (e.g. multi-op funcs).  Also stamped on the func so afir-translate can lift
  // it into tiling_space.json without re-deriving.
  std::string blockDimExpr;
  {
    auto dimSymsAttr = func->getAttrOfType<ArrayAttr>("afir.dim_symbols");
    auto axisExtAttr = func->getAttrOfType<ArrayAttr>("afir.axis_extents");
    StringRef xblockName;
    for (auto &grp : plan.tileable)
      for (const auto &tp : grp)
        if (tp.level == TileLevel::Outer)
          xblockName = tp.name;
    std::optional<symshape::DimSymbolTable> symTable;
    if (dimSymsAttr)
      symTable = symshape::DimSymbolTable::fromAttr(dimSymsAttr);
    if (axisExtAttr && symTable && !xblockName.empty() &&
        !plan.blockFusedAxes.empty()) {
      symshape::SymExpr ext;
      bool ok = true;
      for (int ax : plan.blockFusedAxes) {
        if (ax < 0 || ax >= (int)axisExtAttr.size()) { ok = false; break; }
        auto s = dyn_cast<StringAttr>(axisExtAttr[ax]);
        auto e = s ? symshape::parseSymExpr(s.getValue()) : std::nullopt;
        if (!e || !e->isValid()) { ok = false; break; }
        ext = ext.isValid() ? symshape::SymExpr::mul(ext, *e) : *e;
      }
      if (ok && ext.isValid()) {
        auto nameFor = [&](symshape::SymId id) -> std::string {
          auto src = symTable->sourceOf(id);
          return "arg" + std::to_string(src.first) + "_dim" +
                 std::to_string(src.second);
        };
        std::string extExpr = ext.emitC(nameFor);
        blockDimExpr = "ceil(" + extExpr + "/" + xblockName.str() + ")";
        // Stamp the bare extent expression too — afir-translate lifts it into
        // tiling_space.json so the runner can evaluate the runtime upper bound
        // (sub product of arg*_dim*) and prune candidates with XBLOCK > extent.
        func->setAttr("afir.axis_extent_expr",
                      StringAttr::get(ctx, extExpr));
      }
    }
  }
  // Static fallback: when symbolic attrs are absent (e.g. an outlined kernel
  // whose static shape never went through afir-symbolize-shapes) but every
  // block-fused axis has a known static extent, emit a literal-int expr. The
  // autotuner's grammar accepts plain integers, so eval just folds.
  if (blockDimExpr.empty() && plan.group && !plan.blockFusedAxes.empty()) {
    StringRef xblockName;
    for (auto &grp : plan.tileable)
      for (const auto &tp : grp)
        if (tp.level == TileLevel::Outer)
          xblockName = tp.name;
    if (!xblockName.empty()) {
      int64_t extent = 1;
      bool allStatic = true;
      for (int ax : plan.blockFusedAxes) {
        if (ax < 0 || ax >= (int)plan.group->collapsedAxes.size()) {
          allStatic = false; break;
        }
        int64_t s = plan.group->collapsedAxes[ax].staticSize;
        if (s == ShapedType::kDynamic) { allStatic = false; break; }
        extent *= s;
      }
      if (allStatic && extent > 0) {
        std::string extExpr = std::to_string(extent);
        blockDimExpr = "ceil(" + extExpr + "/" + xblockName.str() + ")";
        func->setAttr("afir.axis_extent_expr",
                      StringAttr::get(ctx, extExpr));
      }
    }
  }
  if (!blockDimExpr.empty())
    func->setAttr("afir.block_dim_expr", StringAttr::get(ctx, blockDimExpr));

  // Stamp the picked reduce template so downstream passes / lit tests can
  // observe which path was taken without re-running the cost model.  Skipped
  // for non-reduce kernels (ReduceTemplate::None).
  StringRef rtName;
  switch (plan.reduceTemplate) {
  case TilePlan::ReduceTemplate::Common:   rtName = "Common";   break;
  case TilePlan::ReduceTemplate::FullLoad: rtName = "FullLoad"; break;
  case TilePlan::ReduceTemplate::RCore:    rtName = "RCore";    break;
  case TilePlan::ReduceTemplate::None:     break;
  }
  if (!rtName.empty())
    func->setAttr("afir.reduce_template", StringAttr::get(ctx, rtName));

  // ─── Build schema_version=2 TilingInfoSchema ────────────────────────────
  // Authored here (pre-bufferize) so that arg_index references for tunable
  // tile-params land on the index-typed BlockArguments TilePlanGen just
  // injected.  Inputs are tensor-typed at this stage; outputs are not yet
  // appended as args (one-shot-bufferize will add them post-pipeline).  We
  // synthesize Output SchemaArg entries from the func return types using
  // the anticipated post-bufferize positions (numArgs + resultIdx).
  TilingInfoSchema schema;
  schema.kernelId     = func.getName().str();
  schema.blockDimExpr = blockDimExpr;
  if (auto a = func->getAttrOfType<StringAttr>("afir.axis_extent_expr"))
    schema.axisExtentExpr = a.getValue().str();

  buildSchemaTunableFields(plan, func, schema);
  buildSchemaArgs(func, schema);

  // Symbolic-shape table for output shape_expr derivation.  Absent for
  // kernels that never went through afir-symbolize-shapes; computeOutput-
  // ShapeExprs then falls back to literal static dims / synthetic tracing.
  std::optional<symshape::DimSymbolTable> symTable;
  if (auto a = func->getAttrOfType<ArrayAttr>("afir.dim_symbols"))
    symTable = symshape::DimSymbolTable::fromAttr(a);
  computeOutputShapeExprs(func, schema, symTable);

  collectShapeDerivedFields(func, schema);
  collectShapeEqualities(func, schema);

  ArrayAttr constraintsAttr = serializeConstraints(ctx, plan.constraints);
  serializeAndAttach(moduleOp, schema, constraintsAttr);
}

// ===========================================================================
// Public entry points used by AutoFuseTileFusePass for P1b multi-variant
// codegen (one func per feasible TilePlanDraft).
// ===========================================================================
SmallVector<auto_fuse::TilePlanDraft>
enumerateFeasibleDrafts(const auto_fuse::CollapsedGroupInfo &info,
                        bool enableReductionSplit,
                        bool relaxNonBlockUbY,
                        llvm::StringRef socName) {
  // CV-fusion Phase 2: Cube kernels enumerate exactly one cube draft.  Real
  // cube TilingCase enumeration (AF's GenMatmulTilingCase) — picking among
  // multiple tile-shape candidates — is deferred to Phase 2 v2.  For now,
  // hardcoded preset defaults (128/128 outer × 32/32 inner × 16 K).
  if (info.kind == GroupInfo::Kind::Cube) {
    auto_fuse::TilePlanDraft d;
    bool hasTrailingVec = false;
    for (linalg::LinalgOp op : info.topoMembers)
      if (!isa<linalg::MatmulOp, linalg::MatmulTransposeAOp,
                linalg::MatmulTransposeBOp, linalg::BatchMatmulOp>(
              op.getOperation())) {
        hasTrailingVec = true;
        break;
      }
    d.cubeKind = hasTrailingVec ? auto_fuse::CubeKind::MatmulVecFuse
                                 : auto_fuse::CubeKind::MatmulOnly;
    return {d};
  }
  const auto_fuse::AxisGrouping &g = info.grouping;
  unsigned elemBytes = operandElemBytes(info);
  DenseSet<int> vecDims = computeVectorizedDims(info);
  SocConstants soc = getSocConstants(socName);
  auto drafts = enumerateTilingCases(g, info, enableReductionSplit, elemBytes);
  SmallVector<auto_fuse::TilePlanDraft> feasible;
  for (const auto &d : drafts) {
    double s = costEstimate(g, info, vecDims, d, elemBytes, soc,
                             relaxNonBlockUbY);
    if (s < kInfeasible) feasible.push_back(d);
  }
  return feasible;
}

auto_fuse::TilePlan
buildPlanForDraft(func::FuncOp func,
                  const auto_fuse::CollapsedGroupInfo &info,
                  const auto_fuse::TilePlanDraft &draft,
                  OpBuilder &builder, Location loc,
                  llvm::StringRef socName) {
  TilePlan plan = buildPlan(func, info, info.grouping, draft, builder, loc);
  // Vector-axis constraints (Divides/LeBytes) don't apply to cube plans —
  // cube tile-data legality is enforced by the existing mix-compiler /
  // matmul tiling library (Phase 6 wires this).  Skip to keep the schema
  // clean.
  if (plan.cubeKind == auto_fuse::CubeKind::None)
    populateConstraints(plan, info, func, operandElemBytes(info),
                        getSocConstants(socName));
  return plan;
}

} // namespace mlir::afir
