#include "TilePlanGen.h"
#include "TileFuseUtils.h"
#include "Analysis/SymbolicShape/DimSymbolTable.h"
#include "Analysis/SymbolicShape/SymExpr.h"
#include "Conversion/VectorPlan/TilePlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <limits>

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

// ===========================================================================
// This file is the "schedule" stage of vector-plan-tile-fuse.  It mirrors
// AutoFuse's optimize/autoschedule (see
// docs/superpowers/plans/2026-05-11-port-af-scheduler-to-vector-plan.zh.md):
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

// Above this many bytes, a reduction axis clearly will not fit on-chip whole,
// so it must be ub-split (≈ AutoFuse's reduce-template feasibility check).
static constexpr int64_t kReductionTileBudgetBytes = 32 * 1024;
// Unified Buffer size on dav-c220 (Ascend 910B); a whole on-chip tile larger
// than this cannot possibly fit, so such a tiling case is deprioritised.
static constexpr int64_t kUBSizeBytes = 192 * 1024;
static const double kInfeasible = std::numeric_limits<double>::infinity();

// Conservative whole-tile on-chip footprint (bytes) for `draft`, when every
// untiled axis has a known static extent.  std::nullopt otherwise -- the real
// per-shape check then happens in the autotuner (via `footprint_expr`).  The
// tiled axes use buildPlan's default inner-tile sizes; the buffer count is the
// fused op's operand count plus a slack for compute temporaries (over-estimate
// in the count, so the result over-states the footprint -> conservative for the
// "definitely won't fit" judgement only as a soft penalty, not a hard reject).
static std::optional<int64_t>
staticTileFootprintBytes(const vector_plan::CollapsedGroupInfo &info,
                         const TilePlanDraft &draft, unsigned elemBytes) {
  constexpr int64_t kDefXBlockSub = 16, kDefRBlock = 64, kDefXBlockX = 16;
  int64_t elemsPerBuf = 1;
  for (auto [i, ax] : llvm::enumerate(info.collapsedAxes)) {
    int64_t sz;
    if ((int)i == draft.ubTilingAxisY)
      sz = kDefXBlockSub;
    else if ((int)i == draft.ubTilingAxisR)
      sz = kDefRBlock;
    else if ((int)i == draft.ubTilingAxisX)
      sz = kDefXBlockX;
    else {
      sz = ax.staticSize;
      if (sz == ShapedType::kDynamic)
        return std::nullopt;
    }
    elemsPerBuf *= sz;
  }
  int64_t numBufs = 2;
  if (!info.topoMembers.empty())
    numBufs = (int64_t)info.topoMembers[0]->getNumOperands() + 1;
  return numBufs * elemsPerBuf * (int64_t)elemBytes;
}

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
                       builder.getStringAttr("vector_plan.default_tile_size"),
                       builder.getI64IntegerAttr(defaultVal))});
  return newArg;
}

// Element byte width of the group's first member's first operand — used for the
// reduction-tile feasibility check (≈ AF's `dtype_size`).
static unsigned operandElemBytes(const CollapsedGroupInfo &info) {
  if (info.topoMembers.empty())
    return 4;
  if (auto st = dyn_cast<ShapedType>(info.topoMembers[0]->getOperand(0).getType()))
    if (st.getElementType().isIntOrFloat())
      return std::max(1u, st.getElementType().getIntOrFloatBitWidth() / 8);
  return 4;
}

namespace {

// (Axis classification — classifyAxes / transposePerm — moved to
// TileFuseUtils; the Collapse pass computes it and stores it in
// CollapsedGroupInfo::grouping, which buildPlan reads below.)

// ≈ AutoFuse's `tensor.attr.vectorized_axis` (the inner axes a vector op
// processes whole, that must not be looped/ub-tiled).  For a reduce member,
// that region is { its reduction iteration dims } ∪ { iteration dims that, in
// some operand's affine_map, appear *after* a reduction dim } — because the
// AscendC reduce intrinsic consumes a contiguous [R,A]/[A,R] tile, so anything
// inner to the reduction in the operand layout has to stay whole.  Non-reduce
// members contribute nothing here.  Returns the union over all members; if
// `plan` is non-null, also records each operand's vectorized iteration dims in
// `plan->vectorizedDims`.
DenseSet<int> computeVectorizedDims(const CollapsedGroupInfo &info,
                                     TilePlan *plan = nullptr) {
  DenseSet<int> vec;
  for (linalg::LinalgOp op : info.topoMembers) {
    auto iterTypes = op.getIteratorTypesArray();
    SmallVector<int> redDims;
    for (int d = 0; d < (int)iterTypes.size(); ++d)
      if (iterTypes[d] == utils::IteratorType::reduction)
        redDims.push_back(d);
    if (redDims.empty())
      continue;
    for (int d : redDims)
      vec.insert(d);
    auto maps     = op.getIndexingMapsArray();
    auto operands = op->getOperands();
    for (auto [operand, m] : llvm::zip(operands, maps)) {
      bool seenRed = false;
      SmallVector<int> opVecDims;
      for (AffineExpr e : m.getResults()) {
        auto de = dyn_cast<AffineDimExpr>(e);
        if (!de)
          continue;
        int d = (int)de.getPosition();
        if (llvm::is_contained(redDims, d)) {
          seenRed = true;
          opVecDims.push_back(d); // the reduction dim itself is vectorized
          continue;
        }
        if (seenRed) {
          vec.insert(d);
          opVecDims.push_back(d);
        }
      }
      if (plan && !opVecDims.empty())
        plan->vectorizedDims[operand] = std::move(opVecDims);
    }
  }
  return vec;
}

struct BlockPick {
  int  axis = -1;             // the block-dispatch axis
  bool degradeToRowLoop = false;
};

// ≈ Scheduler::BlockSplit + the §3.5 "no usable inner tile axis ⇒ block-axis
// row loop" degradation.  P1/P3a: the block axis is the first non-broadcast
// parallel axis that is *not* inside a reduce's vectorized region; the
// degradation fires when some parallel axis after it *is* in that region (then
// pairing the block axis with a normal XBLOCK_SUB inner level would re-create a
// non-contiguous operand slice — e.g. Case B's `out[d0,d2]=sum_{d1}x[d0,d1,d2]`,
// where d2 is vectorized).  P2 replaces the "first axis" with a fused leading run.
BlockPick pickBlockAxis(const AxisGrouping &g, const DenseSet<int> &vecDims) {
  BlockPick bp;
  for (int i : g.yAxes) {
    if (g.axes[i].isBroadcastConst || vecDims.count(i))
      continue;
    bp.axis = i;
    break;
  }
  if (bp.axis < 0) {
    // No non-vectorized non-broadcast parallel axis (e.g. `[R, P]`-style, where
    // the only parallel axis is inner to the reduction): fall back to the first
    // non-broadcast parallel axis — same (unsupported, non-contiguous) outcome
    // the previous code produced; a proper fix for those shapes comes later.
    for (int i : g.yAxes)
      if (!g.axes[i].isBroadcastConst) {
        bp.axis = i;
        break;
      }
  }
  if (bp.axis >= 0)
    for (int i : g.yAxes)
      if (i > bp.axis && !g.axes[i].isBroadcastConst && vecDims.count(i)) {
        bp.degradeToRowLoop = true;
        break;
      }
  return bp;
}

// ≈ AutoFuse's GenTilingCase + PruneTilingCase.  Walks the cartesian product of
// (ub_tiling_id over each axis group), emits one TilePlanDraft per point — plus
// an RCore variant (`reduceIsBlock`) when a reduce axis is ub-split — then
// prunes degenerate single-tile-axis cases.  **P5c** enumerates `ubY` over all
// non-broadcast parallel axes (block axis first) and `ubR` over { R whole, each
// oversized / `--enable-reduction-split` R axis }; `ubX` is still fixed to the
// first transpose-X axis (a real choice needs the >16-fractal split).  Drafts
// whose codegen isn't implemented yet — a non-block ub-Y axis, RCore, FullLoad —
// are still enumerated but `costEstimate` keeps them infeasible, so the picked
// plan stays exactly what the previous scheduler produced (ubY = block axis,
// auto-split R when oversized).
//
// This is intentionally a *pure* function — it never mutates `func` — so
// `pickBest` can score every candidate (blockDim, reduce template, ub-axis pick
// are all determinable from the grouping + collapsed sizes) before the winning
// one is materialized by `buildPlan`.
SmallVector<TilePlanDraft>
enumerateTilingCases(const AxisGrouping &g, const CollapsedGroupInfo &info,
                     bool enableReductionSplit, unsigned elemBytes) {
  DenseSet<int> vecDims = computeVectorizedDims(info);
  BlockPick     bp      = pickBlockAxis(g, vecDims);

  int ubX = g.xAxes.empty() ? -1 : g.xAxes.front();

  // ubY candidates ≈ GenTilingCase over y_group: every parallel axis (broadcast
  // axes included — they're ordinary Y axes to the scheduler), the block axis
  // first (so ties in costEstimate keep the current pick).  The row-loop
  // degradation forces ubY = -1 (the block axis row-loops, no inner Y tile).
  SmallVector<int> ubYs;
  if (bp.degradeToRowLoop || bp.axis < 0) {
    ubYs.push_back(bp.degradeToRowLoop ? -1 : bp.axis);
  } else {
    ubYs.push_back(bp.axis);
    for (int y : g.yAxes)
      if (y != bp.axis)
        ubYs.push_back(y);
  }

  // ubR candidates ≈ GenTilingCase over r_group: R kept whole, or one of the R
  // axes ub-split.  --enable-reduction-split forces a split (no "whole" case);
  // a true full-reduce (no parallel axes) likewise must split — R kept whole
  // would mean block_dim=1 with no parallelism, only path to multi-core is
  // RCore over an R-split (the RCore variants are appended below).  Otherwise
  // an R axis is only a split candidate when it's clearly too big to fit
  // on-chip whole.  The row-loop degradation keeps R whole (as before).
  SmallVector<int> ubRs;
  if (bp.degradeToRowLoop || g.rAxes.empty()) {
    ubRs.push_back(-1);
  } else if (enableReductionSplit || g.yAxes.empty()) {
    for (int r : g.rAxes)
      ubRs.push_back(r);
  } else {
    ubRs.push_back(-1);
    for (int r : g.rAxes) {
      int64_t sz = info.collapsedAxes[r].staticSize;
      if (sz != ShapedType::kDynamic &&
          sz * (int64_t)elemBytes > kReductionTileBudgetBytes)
        ubRs.push_back(r);
    }
  }

  SmallVector<TilePlanDraft> drafts;
  for (int y : ubYs)
    for (int r : ubRs) {
      TilePlanDraft d;
      d.ubTilingAxisY = y;
      d.ubTilingAxisX = ubX;
      d.ubTilingAxisR = r;
      drafts.push_back(d);
      if (r != -1) {
        // RCore variant (reduce axis also split across cores).  Enumerated for
        // completeness; ∞-scored in costEstimate until the two-stage codegen exists.
        TilePlanDraft rc = d;
        rc.reduceIsBlock = true;
        rc.blockTilingId = 1;
        drafts.push_back(rc);
      }
    }

  // PruneTilingCase: in the single-tile-axis scenario (only ubY is a tile axis)
  // a draft whose ubY axis has static extent 1 is pointless — drop it if there
  // is another draft to fall back to.
  if (drafts.size() > 1) {
    llvm::erase_if(drafts, [&](const TilePlanDraft &d) {
      return d.ubTilingAxisX < 0 && d.ubTilingAxisR < 0 &&
             d.ubTilingAxisY >= 0 &&
             info.collapsedAxes[d.ubTilingAxisY].staticSize == 1;
    });
    if (drafts.empty()) // shouldn't happen, but never return nothing
      drafts.push_back(TilePlanDraft{});
  }

  // TODO(P5+): also enumerate a FullLoad grouping variant (classifyAxes with
  // reduce→N) when the reduction tile fits whole.
  return drafts;
}

// ≈ AutoFuse's score_func (argmin selects; ties → enumeration order, which puts
// the "ubY = block axis, R whole" draft first).  Feasibility-only for now: ∞ for
// a kept-whole reduction axis that can't fit on-chip, ∞ for an RCore/FullLoad
// reduce template (no codegen yet), and ∞ for a non-block ub-Y axis (buildPlan
// *can* materialize it — the "block-axis swap" below — but with the trivial cost
// here it would never be a sensible pick, e.g. it would block-dispatch a tiny
// broadcast axis instead of the big parallel one; a real §3.6 cost — blockDim
// distance to #AICores, vectorized bytes, etc. — lands with P6's UB-peak model
// and is what should override this).
double costEstimate(const AxisGrouping &g, const CollapsedGroupInfo &info,
                    const DenseSet<int> &vecDims,
                    const TilePlanDraft &draft, unsigned elemBytes) {
  // RCore (R axis as block axis, two-stage partial→combine codegen).  P3b-2a:
  // open the gate for true full-reduce (no parallel axes to dispatch over —
  // RCore is the only path to block_dim>1) and keep it ∞ everywhere else
  // (Common with a real parallel block axis is always preferable until P3b-4's
  // real cost model can compare them).  buildPlan codegen for RCore lands in
  // P3b-2b/c/d; until then buildPlan asserts with a clear "WIP" message.
  if (draft.reduceIsBlock) {
    if (g.yAxes.empty() && draft.ubTilingAxisR >= 0)
      return 0.0;
    return kInfeasible;
  }
  // Full-reduce (no parallel axis) on the Common template — whether R kept
  // whole (block_dim=1, no parallelism) or R ub-split (RBLOCK with no parallel
  // outer scf.for) — both hit the R1 bug in GroupEmitter (setInsertionPointToEnd
  // lands on the func entry block past func.return).  Reject so RCore is the
  // unique pick when full-reduce.
  if (g.yAxes.empty())
    return kInfeasible;
  if (draft.ubTilingAxisY >= 0 &&
      draft.ubTilingAxisY != pickBlockAxis(g, vecDims).axis)
    return kInfeasible; // non-block ub-Y: see comment above.
  for (int r : g.rAxes) {
    if (r == draft.ubTilingAxisR)
      continue; // this R axis is ub-split → its on-chip tile is RBLOCK-bounded.
    // A statically oversized R axis kept whole can't fit; a dynamic one we can't
    // prove either way, so (matching the previous scheduler) leave it Full.
    int64_t sz = info.collapsedAxes[r].staticSize;
    if (sz != ShapedType::kDynamic &&
        sz * (int64_t)elemBytes > kReductionTileBudgetBytes)
      return kInfeasible; // R kept whole but won't fit.
  }
  // ≈ AF's UB-peak penalty (w3): when the whole-tile footprint is statically
  // known and exceeds the UB, deprioritise this draft.  A soft penalty (not
  // kInfeasible) so the pass never runs out of feasible drafts; the autotuner
  // does the per-shape check via `footprint_expr` in vector_plan.tiling_infos.
  if (auto fp = staticTileFootprintBytes(info, draft, elemBytes);
      fp && *fp > kUBSizeBytes)
    return 1.0e9 + (double)*fp; // larger overflow → larger penalty
  return 0.0;
}

} // namespace

// Materialize one tiling-case draft into a TilePlan: ≈ Scheduler::BlockSplit
// (pickBlockAxis → blockFusedAxes / XBLOCK) + Scheduler::TileSplit (the in-order
// pass below over the collapsed axes, dispatching per axis kind) + the §3.5
// row-loop degradation.  This is the *only* part of TilePlanGen that mutates
// `func` (it appends the tunable tile-size arguments).
static TilePlan buildPlan(func::FuncOp func, const CollapsedGroupInfo &info,
                          const AxisGrouping &g, const TilePlanDraft &draft,
                          OpBuilder &builder, Location loc) {
  // P3b-2a wires the cost-model gate so RCore is selectable for full-reduce,
  // but the buildPlan / LoopNestBuilder / GroupEmitter codegen lands in
  // P3b-2b/c/d.  Until then, surface a clear error rather than producing bad IR.
  if (draft.reduceIsBlock) {
    llvm::errs() << "[vector-plan] RCore reduce template selected but codegen "
                    "WIP (P3b-2b/c/d). Plan: docs/superpowers/plans/"
                    "2026-05-14-p3b-rcore-reduce-multicore.zh.md\n";
    llvm::report_fatal_error("RCore codegen not yet implemented (P3b-2a)");
  }
  assert(draft.blockTilingId == 0 &&
         "blockTilingId != 0 only valid for RCore");

  TilePlan plan;
  plan.group = &info;
  if (!g.rAxes.empty())
    plan.reduceTemplate = TilePlan::ReduceTemplate::Common;

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
  // Counters preserved so func-arg insertion order / `vector_plan.tiling_infos`
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

TilePlan genVectorTilePlan(func::FuncOp func,
                            const CollapsedGroupInfo &info,
                            OpBuilder &builder, Location loc,
                            bool enableReductionSplit) {
  const AxisGrouping &g = info.grouping; // computed by the Collapse pass
  unsigned elemBytes = operandElemBytes(info);
  DenseSet<int> vecDims = computeVectorizedDims(info);

  SmallVector<TilePlanDraft> drafts =
      enumerateTilingCases(g, info, enableReductionSplit, elemBytes);
  assert(!drafts.empty() && "enumerateTilingCases must yield at least one draft");

  // argmin score (≈ score_func selection); ties → enumeration order, which puts
  // the "current scheduler" choice first.  Only the winner is materialized —
  // buildPlan is the one place that mutates `func`.
  const TilePlanDraft *best = &drafts.front();
  double bestScore = costEstimate(g, info, vecDims, *best, elemBytes);
  for (const TilePlanDraft &d : llvm::drop_begin(drafts)) {
    double s = costEstimate(g, info, vecDims, d, elemBytes);
    if (s < bestScore) { bestScore = s; best = &d; }
  }
  assert(bestScore < kInfeasible && "no feasible tiling case");

  return buildPlan(func, info, g, *best, builder, loc);
}

void emitTilingInfos(func::FuncOp func, const TilePlan &plan) {
  MLIRContext *ctx = func.getContext();
  auto moduleOp = func->getParentOfType<ModuleOp>();
  if (!moduleOp) return;

  Type i32Ty = IntegerType::get(ctx, 32);
  Type i64Ty = IntegerType::get(ctx, 64);

  SmallVector<Attribute> fields;
  int32_t abiIndex = 0;

  for (auto &group : plan.tileable) {
    for (const auto &tp : group) {
      auto ba = dyn_cast<BlockArgument>(tp.ssa);
      if (!ba) continue;

      int64_t defaultVal = 0;
      if (auto attr = func.getArgAttrOfType<IntegerAttr>(
              ba.getArgNumber(), "vector_plan.default_tile_size"))
        defaultVal = attr.getInt();

      // Static extent of the axis this param tiles -- -1 when dynamic.  Lets
      // afir-translate / the autotuner cap the search range at the axis size.
      int64_t axisSize = -1;
      if (plan.group && tp.axisIdx >= 0 &&
          tp.axisIdx < (int)plan.group->collapsedAxes.size()) {
        int64_t s = plan.group->collapsedAxes[tp.axisIdx].staticSize;
        if (s != ShapedType::kDynamic)
          axisSize = s;
      }

      assert(ba.getArgNumber() <= (unsigned)INT32_MAX && "arg_index overflow");
      NamedAttrList fieldAttrs;
      fieldAttrs.append("abi_index",
                        IntegerAttr::get(i32Ty, abiIndex));
      fieldAttrs.append("arg_index",
                        IntegerAttr::get(i32Ty, (int32_t)ba.getArgNumber()));
      fieldAttrs.append("axis_size", IntegerAttr::get(i64Ty, axisSize));
      fieldAttrs.append("default_value",
                        IntegerAttr::get(i64Ty, defaultVal));
      fieldAttrs.append("kind", StringAttr::get(ctx, "tunable"));
      fieldAttrs.append("name", StringAttr::get(ctx, tp.name));
      fields.push_back(fieldAttrs.getDictionary(ctx));
      ++abiIndex;
    }
  }

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
  if (!blockDimExpr.empty())
    func->setAttr("afir.block_dim_expr", StringAttr::get(ctx, blockDimExpr));

  NamedAttrList entryAttrs;
  entryAttrs.append("fields", ArrayAttr::get(ctx, fields));
  entryAttrs.append("kernel_id", StringAttr::get(ctx, func.getName()));
  if (!blockDimExpr.empty())
    entryAttrs.append("block_dim_expr", StringAttr::get(ctx, blockDimExpr));

  StringRef attrName = "vector_plan.tiling_infos";
  SmallVector<Attribute> infos;
  if (auto existing = moduleOp->getAttrOfType<ArrayAttr>(attrName))
    llvm::append_range(infos, existing.getValue());
  infos.push_back(entryAttrs.getDictionary(ctx));
  moduleOp->setAttr(attrName, ArrayAttr::get(ctx, infos));
}

} // namespace mlir::afir
