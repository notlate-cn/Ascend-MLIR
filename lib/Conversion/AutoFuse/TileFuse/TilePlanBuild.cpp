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

static Value insertFuncArg(func::FuncOp func, OpBuilder &builder,
                            Location loc, int64_t defaultVal,
                            StringRef /*paramName*/) {
  unsigned idx = func.getNumArguments();
  SmallVector<Type> argTypes(func.getFunctionType().getInputs());
  argTypes.push_back(builder.getIndexType());
  func.setType(FunctionType::get(builder.getContext(), argTypes,
                                  func.getFunctionType().getResults()));
  Value newArg =
      func.getBody().front().addArgument(builder.getIndexType(), loc);
  func.setArgAttrs(idx,
                   {NamedAttribute(
                       builder.getStringAttr("auto_fuse.default_tile_size"),
                       builder.getI64IntegerAttr(defaultVal))});
  return newArg;
}



// CV-fusion Phase 2: materialize a Cube template draft.  Emits the 5 cube
// tunable i64 func args (XBLOCK_M, XBLOCK_N, M_INNER, N_INNER, K_INNER) and
// stamps `ascendc.kernel_kind = "mix"` + `afir.cube_kind = "<kind>"` attrs on
// the func.  The TileParam entries land in plan.tileable so emitTilingInfos
// renders them into `auto_fuse.tiling_infos`.  Phase 4 will read these in
// LoopNestBuilder + GroupEmitter to emit the 3-level scf.for nest.
//
// Axis indices are placeholders (0=M, 1=N, 2=K) for now — Phase 4 will line
// them up with the linalg.matmul's iteration dims.
static TilePlan buildCubePlan(func::FuncOp func, const CollapsedGroupInfo &info,
                              const TilePlanDraft &draft,
                              OpBuilder &builder, Location loc) {
  TilePlan plan;
  plan.group    = &info;
  plan.cubeKind = draft.cubeKind;

  // 5 cube tunable args.  Defaults are AF-style for f16 matmul: 128/128/64
  // outer; 32/32 inner-MN; 16 inner-K (one cube fragment).
  auto addInner = [&](StringRef name, int64_t defaultVal, int axisIdx,
                      AxisRole role) {
    Value param = insertFuncArg(func, builder, loc, defaultVal, name);
    SmallVector<TileParam> grp;
    grp.push_back({name.str(), param, OpFoldResult(param), axisIdx,
                    TileLevel::Inner, role});
    plan.tileable.push_back(std::move(grp));
  };
  auto addOuterInner = [&](StringRef outerName, int64_t outerDefault,
                           StringRef innerName, int64_t innerDefault,
                           int axisIdx, AxisRole role) {
    Value outerSsa = insertFuncArg(func, builder, loc, outerDefault, outerName);
    Value innerSsa = insertFuncArg(func, builder, loc, innerDefault, innerName);
    SmallVector<TileParam> grp;
    grp.push_back({outerName.str(), outerSsa, OpFoldResult(outerSsa), axisIdx,
                    TileLevel::Outer, role});
    grp.push_back({innerName.str(), innerSsa, OpFoldResult(innerSsa), axisIdx,
                    TileLevel::Inner, role});
    plan.tileable.push_back(std::move(grp));
  };

  // Order: M outer/inner, N outer/inner, K inner.  Matches AF tile spec
  // (matmul-add-leakyrelu/step5_ascendc.mlir args arg4..arg8).
  addOuterInner("XBLOCK_M", 128, "M_INNER", 32, 0, AxisRole::Parallel);
  addOuterInner("XBLOCK_N", 128, "N_INNER", 32, 1, AxisRole::Parallel);
  addInner     ("K_INNER",        16,                2, AxisRole::Reduction);

  // Mark the func as a mix kernel; AnnotateMixMatmulSemantics + downstream
  // passes use `ascendc.kernel_kind` to gate behavior.
  func->setAttr("ascendc.kernel_kind", builder.getStringAttr("mix"));
  // Record the cube template explicitly so observers (lit / future verifier)
  // can pin the picked path without re-running the analysis.
  StringRef cubeKindName = "None";
  switch (draft.cubeKind) {
  case CubeKind::None:          cubeKindName = "None"; break;
  case CubeKind::MatmulOnly:    cubeKindName = "MatmulOnly"; break;
  case CubeKind::MatmulVecFuse: cubeKindName = "MatmulVecFuse"; break;
  }
  func->setAttr("afir.cube_kind", builder.getStringAttr(cubeKindName));
  // Mirror AnnotateMixMatmulSemantics so MixAbiExtractor's required-attr set
  // is satisfied without re-running that legacy pass on cube-emitted kernels.
  // Phase 1 CV-fusion only supports the canonical matmul shape: no transpose,
  // ND layout on all sides.  bias/epilogue come from a tensor-IR walk of the
  // trailing chain (vs. the prior "always Relu / never bias" hardcoding).
  MLIRContext *ctx = builder.getContext();
  func->setAttr("abi_matmul_op_kind",     StringAttr::get(ctx, "matmul"));
  func->setAttr("abi_matmul_trans_a",     BoolAttr::get(ctx, false));
  func->setAttr("abi_matmul_trans_b",     BoolAttr::get(ctx, false));
  func->setAttr("abi_matmul_layout_a",    StringAttr::get(ctx, "ND"));
  func->setAttr("abi_matmul_layout_b",    StringAttr::get(ctx, "ND"));
  func->setAttr("abi_matmul_layout_c",    StringAttr::get(ctx, "ND"));
  auto [hasBias, epilogueKindName] =
      classifyCubeEpilogueChain(func, draft.cubeKind);
  func->setAttr("abi_matmul_has_bias", BoolAttr::get(ctx, hasBias));
  func->setAttr("abi_matmul_epilogue_kind",
                StringAttr::get(ctx, epilogueKindName));
  return plan;
}

// Materialize one tiling-case draft into a TilePlan: ≈ Scheduler::BlockSplit
// (pickBlockAxis → blockFusedAxes / XBLOCK) + Scheduler::TileSplit (the in-order
// pass below over the collapsed axes, dispatching per axis kind) + the §3.5
// row-loop degradation.  This is the *only* part of TilePlanGen that mutates
// `func` (it appends the tunable tile-size arguments).
TilePlan buildPlan(func::FuncOp func, const CollapsedGroupInfo &info,
                          const AxisGrouping &g, const TilePlanDraft &draft,
                          OpBuilder &builder, Location loc) {
  // CV-fusion Phase 2: dispatch to the cube branch when the group is a Cube
  // template.  Cube plans don't share the vector axis-grouping pipeline below.
  if (draft.cubeKind != CubeKind::None)
    return buildCubePlan(func, info, draft, builder, loc);
  // P3b-2b: RCore TilePlan structure.  R axis (at draft.ubTilingAxisR) becomes
  // the block axis, dispatched on XBLOCK; an inner RBLOCK loop sweeps the
  // per-block R slice on each core.  LoopNestBuilder reuses its existing
  // Outer-block + Inner-with-parent-step machinery (the per-block inner ub is
  // `parentStep == XBLOCK` via the same code path that Common uses for ubY).
  // GroupEmitter codegen for RCore (per-block reduce + partial-output bump)
  // lands in P3b-2d; until then `genVectorTilePlan` guards with a clear error.
  if (draft.reduceIsBlock) {
    assert(g.yAxes.empty() &&
           "RCore restricted to true full-reduce in P3b-2 (P3b-4 widens)");
    assert(draft.ubTilingAxisR >= 0 && "RCore requires a ub-split R axis");

    TilePlan plan;
    plan.group = &info;
    plan.reduceTemplate = TilePlan::ReduceTemplate::RCore;
    plan.reduceIsBlock = true;
    (void)computeVectorizedDims(info, &plan);

    int rAxis = draft.ubTilingAxisR;
    plan.blockFusedAxes.push_back(rAxis);

    int xsubCount = 0, naxisCount = 0, rblockCount = 0;
    for (int i = 0; i < (int)info.collapsedAxes.size(); ++i) {
      const AxisClass &ax = g.axes[i];
      Value ext = getAxisExtentValue(builder, loc, info, i);

      if (ax.kind == AxisKind::X) {
        // Rare in a full-reduce, but mirror Common's handling.
        std::string name = llvm::formatv("XBLOCK_X_{0}", xsubCount++).str();
        Value param = insertFuncArg(func, builder, loc, 16, name);
        SmallVector<TileParam> group;
        group.push_back({name, param, OpFoldResult(ext), i, TileLevel::Inner,
                          AxisRole::Parallel});
        plan.tileable.push_back(std::move(group));
        if (plan.ubTilingAxisX < 0)
          plan.ubTilingAxisX = i;

      } else if (ax.kind == AxisKind::N) {
        std::string name = llvm::formatv("NAXIS_{0}", naxisCount++).str();
        plan.full.push_back({name, ext, OpFoldResult(ext), i, TileLevel::Full,
                              AxisRole::Parallel});

      } else if (ax.kind == AxisKind::Y) {
        llvm_unreachable("RCore: full-reduce has no Y axes (asserted above)");

      } else { // AxisKind::R
        if (i == rAxis) {
          // R as block axis: XBLOCK (Outer, ascendc.parallel) + RBLOCK (Inner).
          // LoopNestBuilder will give the Inner a ub of parentStep == XBLOCK
          // (per-block R slice), and the composed loopIVs[rAxis] = outer + inner.
          Value xblock = insertFuncArg(func, builder, loc, 128, "XBLOCK");
          std::string innerName =
              llvm::formatv("RBLOCK_{0}", rblockCount++).str();
          Value rblock = insertFuncArg(func, builder, loc, 64, innerName);
          SmallVector<TileParam> group;
          group.push_back({"XBLOCK", xblock, OpFoldResult(ext), i,
                            TileLevel::Outer, AxisRole::Reduction});
          group.push_back({innerName, rblock, OpFoldResult(xblock), i,
                            TileLevel::Inner, AxisRole::Reduction});
          plan.tileable.push_back(std::move(group));
          plan.ubTilingAxisR = i;
          plan.blockDimExprs.push_back(OpFoldResult(
              builder.create<arith::CeilDivSIOp>(loc, ext, xblock)));
        } else {
          // Secondary R axes (full-reduce typically has one) → full-load.
          std::string name =
              llvm::formatv("RBLOCK_{0}", rblockCount++).str();
          plan.full.push_back({name, ext, OpFoldResult(ext), i,
                                TileLevel::Full, AxisRole::Reduction});
        }
      }
    }
    return plan;
  }
  assert(draft.blockTilingId == 0 &&
         "blockTilingId != 0 only valid for RCore");

  TilePlan plan;
  plan.group = &info;
  if (!g.rAxes.empty())
    plan.reduceTemplate = draft.isFullLoad
                              ? TilePlan::ReduceTemplate::FullLoad
                              : TilePlan::ReduceTemplate::Common;
  plan.peelOuterR = draft.peelOuterR;

  DenseSet<int> vecDims = computeVectorizedDims(info, &plan);
  BlockPick     bp      = pickBlockAxis(g, vecDims);
  // The cost model may pick a non-block ub-Y axis; honor it by making *that*
  // axis the block axis (the pickBlockAxis default then becomes an ordinary
  // whole-loaded parallel axis).  When ubY < 0 (the §3.5 row-loop degradation,
  // or a group with no parallel axis) the block axis is the pickBlockAxis result
  // and gets the step-1 row loop instead of XBLOCK_SUB.
  int  blkAxis    = (draft.ubTilingAxisY >= 0) ? draft.ubTilingAxisY : bp.axis;
  bool blkRowLoop = bp.degradeToRowLoop; // implies draft.ubTilingAxisY < 0
  if (blkAxis >= 0)
    plan.blockFusedAxes.push_back(blkAxis);

  // --- ubSplit: one in-order pass over the collapsed axes (≈ TileSplit) ---
  // Counters preserved so func-arg insertion order / `auto_fuse.tiling_infos`
  // numbering match the previous implementation on the shapes that reach here.
  int rblockCount = 0, naxisCount = 0, xsubCount = 0;

  for (int i = 0; i < (int)info.collapsedAxes.size(); ++i) {
    const AxisClass &ax = g.axes[i];
    Value ext = getAxisExtentValue(builder, loc, info, i);

    if (ax.kind == AxisKind::X) {
      // Transpose X axis (input-side divergent): inner-tiled with its own
      // tunable; never the block axis (bindMultiCore == false).  Until the §5
      // transpose schedule generator gives X its own 16-fractal handling, every
      // X axis is an ordinary inner tile (and `draft.ubTilingAxisX` just names
      // the first one for the cost model).
      std::string name = llvm::formatv("XBLOCK_X_{0}", xsubCount++).str();
      Value param = insertFuncArg(func, builder, loc, 16, name);
      SmallVector<TileParam> group;
      group.push_back({name, param, OpFoldResult(ext), i, TileLevel::Inner,
                        AxisRole::Parallel});
      plan.tileable.push_back(std::move(group));
      if (plan.ubTilingAxisX < 0)
        plan.ubTilingAxisX = i;

    } else if (ax.kind == AxisKind::N) {
      // Transpose-N axis (trailing dim the permutation leaves in place): not
      // tiled, not looped — whole-dim slice (≈ AF's n_group).
      std::string name = llvm::formatv("NAXIS_{0}", naxisCount++).str();
      plan.full.push_back({name, ext, OpFoldResult(ext), i, TileLevel::Full,
                            AxisRole::Parallel});

    } else if (ax.kind == AxisKind::Y) {
      // Parallel axis (broadcast axes included — to the scheduler they are
      // ordinary Y axes; the lowering replicates the projecting operand
      // on-chip).
      if (i == blkAxis) {
        // Block axis (= ubY when the draft picked one, else the pickBlockAxis
        // default): XBLOCK (Outer) + inner level (XBLOCK_SUB, or step-1 row loop
        // under the §3.5 degradation).
        Value xblock = insertFuncArg(func, builder, loc, 128, "XBLOCK");
        SmallVector<TileParam> group;
        group.push_back({"XBLOCK", xblock, OpFoldResult(ext), i,
                          TileLevel::Outer, AxisRole::Parallel});
        if (blkRowLoop) {
          // Fixed step 1 (not a tunable func arg): one row per tile body.
          Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
          group.push_back({"XBLOCK_ROW", one, OpFoldResult(xblock), i,
                            TileLevel::Inner, AxisRole::Parallel});
        } else {
          Value xblockSub =
              insertFuncArg(func, builder, loc, 16, "XBLOCK_SUB");
          plan.ubTilingAxisY = i;
          group.push_back({"XBLOCK_SUB", xblockSub, OpFoldResult(xblock), i,
                            TileLevel::Inner, AxisRole::Parallel});
        }
        plan.tileable.push_back(std::move(group));
        plan.blockDimExprs.push_back(
            OpFoldResult(builder.create<arith::CeilDivSIOp>(loc, ext, xblock)));
      } else {
        // §3.4: every non-ub parallel axis is fully loaded — no plan entry, so
        // SliceComputer yields the whole-dim slice and there is no loop over it.
        // (This replaces the old per-axis `XBLOCK_SUB_n` and the broadcast-axis
        // `BCAST_n` step-1 / `BCAST_TILE_n` tunables; also the pre-existing
        // behavior for non-block parallel axes under the §3.5 degradation.)
      }

    } else { // AxisKind::R — reduction axis.
      std::string name = llvm::formatv("RBLOCK_{0}", rblockCount++).str();
      if (i == draft.ubTilingAxisR) {
        Value param = insertFuncArg(func, builder, loc, 64, name);
        SmallVector<TileParam> group;
        group.push_back({name, param, OpFoldResult(ext), i,
                          TileLevel::Inner, AxisRole::Reduction});
        plan.tileable.push_back(std::move(group));
        plan.ubTilingAxisR = i;
      } else {
        plan.full.push_back({name, ext, OpFoldResult(ext), i,
                              TileLevel::Full, AxisRole::Reduction});
      }
    }
  }
  return plan;
}

// Stamp tile-data legality constraints (≈ ATT tiling-data constraints, plan
// §6) onto `plan` for later emission into `auto_fuse.tiling_infos`:
//
//   Divides : XBLOCK_SUB | XBLOCK  (and analogues per tileable group)
//             — required by the existing tail-peel codegen path.
//   LeBytes : conservative on-chip footprint ≤ UB capacity
//             — `numBufs * elemBytes * Π per-axis-tile`.  Each axis's tile
//             is the inner tunable name when tileable, otherwise the static
//             extent / `argN_dimD` shape-key emitted by the SymExpr layer.
//             Skipped (no LeBytes emitted) if any axis fails to resolve;
//             that's also when the runtime UB-aware prune in CannTranslation
//             (`ub_cost_bytes_exprs`) takes over.
//
// These are *recorded* (and surfaced as JSON) — the constraint solver / cost
// model is the autotuner's job; `costEstimate` here just does a coarse static
// check via `staticTileFootprintBytes` already.
void populateConstraints(TilePlan &plan, const CollapsedGroupInfo &info,
                                 func::FuncOp func, unsigned elemBytes,
                                 SocConstants soc) {
  // Divides: every (Outer, Inner) pair within a tileable group.
  for (auto &group : plan.tileable) {
    const TileParam *outer = nullptr, *inner = nullptr;
    for (auto &tp : group) {
      if (tp.level == TileLevel::Outer)      outer = &tp;
      else if (tp.level == TileLevel::Inner) inner = &tp;
    }
    if (outer && inner)
      plan.constraints.push_back(
          {TileConstraint::Divides, inner->name, outer->name});
  }

  // LeBytes: only when every axis has a usable tile-size expression.
  auto dimSymsAttr = func->getAttrOfType<ArrayAttr>("afir.dim_symbols");
  std::optional<symshape::DimSymbolTable> symTable;
  if (dimSymsAttr)
    symTable = symshape::DimSymbolTable::fromAttr(dimSymsAttr);
  auto nameFor = [&](symshape::SymId id) -> std::string {
    if (!symTable) return "?";
    auto src = symTable->sourceOf(id);
    return "arg" + std::to_string(src.first) + "_dim" +
           std::to_string(src.second);
  };

  // Per-axis tile-size string.
  auto axisTileStr = [&](int axisIdx) -> std::string {
    for (auto &grp : plan.tileable)
      for (auto &tp : grp)
        if (tp.axisIdx == axisIdx && tp.level == TileLevel::Inner)
          return tp.name;
    const auto &ax = info.collapsedAxes[axisIdx];
    if (ax.staticSize != ShapedType::kDynamic)
      return std::to_string(ax.staticSize);
    if (ax.extent.isValid() && symTable)
      return ax.extent.emitC(nameFor);
    return ""; // unknown
  };

  // (Removed: the Divides{32, (extent-INNER)*elemBytes} tail-alignment reject
  // constraint.  The ragged tail now routes its GM<->UB copies through
  // DataCopyPad, which handles an unaligned f16 tail offset/length directly, so misaligned
  // tilings are correct rather than rejected.  Aligned tilings remain
  // naturally preferred by the footprint/cost model below.)

  std::string product;
  bool ok = true;
  for (int i = 0; i < (int)info.collapsedAxes.size(); ++i) {
    std::string s = axisTileStr(i);
    if (s.empty()) { ok = false; break; }
    product = product.empty() ? s : ("(" + product + " * " + s + ")");
  }
  if (!ok || product.empty()) return;

  int64_t numBufs = 2;
  if (!info.topoMembers.empty())
    numBufs = (int64_t)info.topoMembers[0]->getNumOperands() + 1;
  std::string footprint = "((" + std::to_string(numBufs * (int64_t)elemBytes) +
                          ") * " + product + ")";
  plan.constraints.push_back(
      {TileConstraint::LeBytes, footprint, std::to_string(soc.ubBytes)});
}

TilePlan genVectorTilePlan(func::FuncOp func,
                            const CollapsedGroupInfo &info,
                            OpBuilder &builder, Location loc,
                            bool enableReductionSplit,
                            llvm::StringRef socName) {
  const AxisGrouping &g = info.grouping; // computed by the Collapse pass
  unsigned elemBytes = operandElemBytes(info);
  DenseSet<int> vecDims = computeVectorizedDims(info);
  SocConstants soc = getSocConstants(socName);

  SmallVector<TilePlanDraft> drafts =
      enumerateTilingCases(g, info, enableReductionSplit, elemBytes);
  assert(!drafts.empty() && "enumerateTilingCases must yield at least one draft");

  // argmin score (≈ score_func selection); ties → enumeration order, which puts
  // the "current scheduler" choice first.  Only the winner is materialized —
  // buildPlan is the one place that mutates `func`.
  const TilePlanDraft *best = &drafts.front();
  double bestScore = costEstimate(g, info, vecDims, *best, elemBytes, soc);
  for (const TilePlanDraft &d : llvm::drop_begin(drafts)) {
    double s = costEstimate(g, info, vecDims, d, elemBytes, soc);
    if (s < bestScore) { bestScore = s; best = &d; }
  }
  // Release-safe barrier (was a debug-only `assert`): if every candidate
  // tiling is infeasible, building a kernel from the best-of-infeasible draft
  // silently produces a UB-overflowing / WIP-codegen kernel.  Fail loud instead
  // of miscompiling.  The passing pipeline always finds a feasible plan, so this
  // never fires for valid kernels.
  if (bestScore >= kInfeasible)
    llvm::report_fatal_error(
        "auto-fuse: no feasible tiling case (all candidate tilings exceed the "
        "UB budget or hit a WIP codegen path); refusing to build an infeasible "
        "kernel");

  TilePlan plan = buildPlan(func, info, g, *best, builder, loc);
  populateConstraints(plan, info, func, elemBytes, soc);
  // P3b-2c/d: LoopNestBuilder is expected to handle RCore via its existing
  // Outer-Inner-on-same-axis path (parentStep mechanism), but the GroupEmitter
  // codegen — per-block R extent on the new rFor + dimension-bumped partial
  // output — lands in P3b-2d.  Surface a clear error until then.
  if (plan.reduceTemplate == TilePlan::ReduceTemplate::RCore) {
    llvm::errs() << "[auto-fuse] RCore TilePlan built (P3b-2b), but "
                    "GroupEmitter codegen WIP (P3b-2d). Plan: "
                    "docs/superpowers/plans/2026-05-14-p3b-rcore-reduce-multicore.zh.md\n";
    /* P3b-2d: guard removed — RCore codegen lives in GroupEmitter now. */
  }
  return plan;
}

} // namespace mlir::afir
