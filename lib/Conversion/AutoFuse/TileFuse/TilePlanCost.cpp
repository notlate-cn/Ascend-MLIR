#include "TilePlanGenInternal.h"
#include "TileFuseUtils.h"
#include "Conversion/AutoFuse/GroupInfo.h"
#include "Conversion/AutoFuse/TilePlan.h"
#include "Target/CannKernel/SocSpec.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::auto_fuse;

namespace mlir::afir {

// Above this many bytes, a reduction axis clearly will not fit on-chip whole,
// so it must be ub-split.
static constexpr int64_t kReductionTileBudgetBytes = 32 * 1024;
const double kInfeasible = std::numeric_limits<double>::infinity();

SocConstants getSocConstants(llvm::StringRef socName) {
  if (auto spec = afir::cannkernel::getSocSpec(socName))
    return {(int64_t)spec->totalUbSize, (int64_t)spec->numAICores};
  return {192 * 1024, 40}; // Ascend910B1 defaults
}

// Element byte width of the group's first member's first operand — used for the
// reduction-tile feasibility check (≈ AF's `dtype_size`).
unsigned operandElemBytes(const CollapsedGroupInfo &info) {
  if (info.topoMembers.empty())
    return 4;
  if (auto st = dyn_cast<ShapedType>(info.topoMembers[0]->getOperand(0).getType()))
    if (st.getElementType().isIntOrFloat())
      return std::max(1u, st.getElementType().getIntOrFloatBitWidth() / 8);
  return 4;
}

// Conservative whole-tile on-chip footprint (bytes) for `draft`, when every
// untiled axis has a known static extent.  std::nullopt otherwise -- the real
// per-shape check then happens in the autotuner (via `footprint_expr`).  The
// tiled axes use buildPlan's default inner-tile sizes; the buffer count is the
// fused op's operand count plus a slack for compute temporaries (over-estimate
// in the count, so the result over-states the footprint -> conservative for the
// "definitely won't fit" judgement only as a soft penalty, not a hard reject).
static std::optional<int64_t>
staticTileFootprintBytes(const auto_fuse::CollapsedGroupInfo &info,
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

// Returns the iter-axis index of the outermost reduction axis that is followed
// by a parallel axis which is in turn followed by another reduction axis (the
// "displaced R" of AF's `IsNeedMultiReduce`, reduce_api_call_base.cpp:118).
// For iter [reduction, parallel, reduction] this returns 0 (r1); for any
// strictly-trailing or strictly-leading R-block it returns -1.
//
// Collapse already merges contiguous same-role axes, so by the time we see
// the AxisGrouping a displaced R cannot be made contiguous by re-ordering.
// This is the trigger for the step=1 peel-outer-R emit path.
static int firstDisplacedReduceAxis(const AxisGrouping &g) {
  int n = (int)g.axes.size();
  for (int i = 0; i < n; ++i) {
    if (g.axes[i].kind != AxisKind::R) continue;
    // Is there a parallel-then-R suffix after i?
    bool sawParallel = false;
    for (int j = i + 1; j < n; ++j) {
      if (g.axes[j].kind == AxisKind::Y || g.axes[j].kind == AxisKind::X)
        sawParallel = true;
      else if (g.axes[j].kind == AxisKind::R && sawParallel)
        return i;
    }
  }
  return -1;
}

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
                                     TilePlan *plan) {
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
    // R-kept-whole (r=-1) is covered by the FullLoad draft pushed below
    // (same IR, FullLoad-tagged plan). Don't emit a redundant Common-no-ubR
    // draft here — the picker would otherwise emit two identical variants.
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

  // FullLoad variant (≈ AF kAllLoad). Reduce axis kept whole; the resulting
  // IR is identical to a Common draft with ubTilingAxisR=-1, so the main
  // enumeration loop drops the r=-1 Common case to avoid an exact duplicate.
  // costEstimate accepts this whenever every R axis fits whole on-chip.
  if (!g.rAxes.empty() && !g.yAxes.empty() && !bp.degradeToRowLoop) {
    for (int y : ubYs) {
      TilePlanDraft fl;
      fl.ubTilingAxisY = y;
      fl.ubTilingAxisX = ubX;
      fl.ubTilingAxisR = -1;
      fl.isFullLoad = true;
      drafts.push_back(fl);
    }
  }

  // Peel-outer-R variant (≈ AF IsNeedMultiReduce): when the group has a
  // displaced reduction axis (R-then-P-then-R in iter order), no single
  // reduce_sum_2d_l2 over a (parallel ++ reduction) flat buffer is correct —
  // FullLoad sims wrong because ComputeConversion's layout heuristic mis-picks
  // RA when the buffer is actually AR.  Emit an explicit "peel the outermost
  // displaced R as a step=1 outer scf.for" draft instead; the GroupEmitter
  // rank-reduces that axis out of the inner generic so what reaches
  // ComputeConversion is the well-tested single-P/single-R shape.
  if (!g.yAxes.empty() && !bp.degradeToRowLoop) {
    int peeled = firstDisplacedReduceAxis(g);
    if (peeled >= 0) {
      for (int y : ubYs) {
        TilePlanDraft pd;
        pd.ubTilingAxisY = y;
        pd.ubTilingAxisX = ubX;
        pd.ubTilingAxisR = -1;
        pd.peelOuterR    = peeled;
        drafts.push_back(pd);
      }
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
                    const TilePlanDraft &draft, unsigned elemBytes,
                    SocConstants soc,
                    bool relaxNonBlockUbY) {
  // When the group has a displaced reduction axis (R-then-P-then-R), the
  // peel-outer-R draft is the only correct codegen path — every other variant
  // hands ComputeConversion a flat (parallel ++ reduction) buffer whose layout
  // heuristic mis-picks RA when AR is correct.  Gate it here so a vanilla
  // FullLoad / Common-no-split / RBLOCK-split never wins on these shapes.
  bool hasDisplaced = firstDisplacedReduceAxis(g) >= 0;
  if (hasDisplaced && draft.peelOuterR < 0)
    return kInfeasible;
  if (draft.peelOuterR >= 0) {
    if (!hasDisplaced)
      return kInfeasible; // peel only meaningful for displaced R
    // Cheap: just a §3.5-style fixed-cost score so the enumeration order
    // (block-axis ubY first) determines tiebreaks.
    return 0.0;
  }
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
  // FullLoad (≈ AF kAllLoad, plan §4): R kept whole, single-shot
  // reduce_sum_2d_l2 over the full R tile. IR-identical to a Common draft with
  // ubTilingAxisR=-1 — enumerateTilingCases drops the redundant Common-no-ubR
  // so this is the only path that reaches buildPlan when R isn't ub-split.
  // Feasibility = every R axis fits whole on-chip.
  if (draft.isFullLoad) {
    for (int r : g.rAxes) {
      int64_t sz = info.collapsedAxes[r].staticSize;
      if (sz != ShapedType::kDynamic &&
          sz * (int64_t)elemBytes > kReductionTileBudgetBytes)
        return kInfeasible;
    }
    return 0.0;
  }
  // Full-reduce (no parallel axis) on the Common template — whether R kept
  // whole (block_dim=1, no parallelism) or R ub-split (RBLOCK with no parallel
  // outer scf.for) — both hit the R1 bug in GroupEmitter (setInsertionPointToEnd
  // lands on the func entry block past func.return).  Reject so RCore is the
  // unique pick when full-reduce.
  if (g.yAxes.empty())
    return kInfeasible;
  if (draft.ubTilingAxisY >= 0 &&
      draft.ubTilingAxisY != pickBlockAxis(g, vecDims).axis) {
    // non-block ub-Y: see comment above. P1b/P6 path opens this up so the
    // autotuner can empirically compare it against the block-axis ubY pick;
    // we mark it as a soft penalty instead of kInfeasible so it survives the
    // feasibility filter and shows up as a multi-variant codegen target.
    if (relaxNonBlockUbY) return 1.0;
    return kInfeasible;
  }
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
  // does the per-shape check via `footprint_expr` in auto_fuse.tiling_infos.
  auto fpOpt = staticTileFootprintBytes(info, draft, elemBytes);
  if (fpOpt && *fpOpt > soc.ubBytes)
    return 1.0e9 + (double)*fpOpt; // larger overflow → larger penalty

  // §3.6 w1: under-saturation penalty — estimate blockDim from the chosen
  // block axis's static extent / default XBLOCK; when blockDim < #AICores,
  // some cores are idle.  Skipped (no penalty) when extent or block axis is
  // unknown — that's the dynamic-shape case where the cost can't decide.
  double score = 0.0;
  {
    BlockPick bp = pickBlockAxis(g, vecDims);
    int blkAxis = (draft.ubTilingAxisY >= 0) ? draft.ubTilingAxisY : bp.axis;
    if (blkAxis >= 0 && blkAxis < (int)info.collapsedAxes.size()) {
      int64_t ext = info.collapsedAxes[blkAxis].staticSize;
      if (ext != ShapedType::kDynamic && ext > 0) {
        constexpr int64_t kDefXBlock = 128; // matches buildPlan default
        int64_t blockDim = (ext + kDefXBlock - 1) / kDefXBlock;
        if (blockDim < soc.numAICores)
          score += (double)(soc.numAICores - blockDim) / (double)soc.numAICores;
      }
    }
  }

  // §3.6 w4: vectorized-bytes bonus — bigger on-chip tile = better SIMD
  // utilisation, up to the UB cap.  Capped at a small magnitude (≤ -0.5)
  // so it can't compete with the w3 over-budget penalty (≥ 1e9) and stays
  // below the w1 idle-core penalty (≤ 1.0).  No effect when footprint is
  // unknown.
  if (fpOpt && *fpOpt <= soc.ubBytes)
    score -= 0.5 * (double)*fpOpt / (double)soc.ubBytes;

  return score;
}

} // namespace mlir::afir
