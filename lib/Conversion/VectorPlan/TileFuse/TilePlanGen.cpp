#include "TilePlanGen.h"
#include "TileFuseUtils.h"
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

// ===========================================================================
// This file is the "schedule" stage of vector-plan-tile-fuse.  It mirrors
// AutoFuse's optimize/autoschedule (see
// docs/superpowers/plans/2026-05-11-port-af-scheduler-to-vector-plan.zh.md):
//
//   genVectorTilePlan
//     ├─ classifyAxes        ≈ TilingGroup::GenTilingGroup   (axis X/Y/R/N)
//     ├─ pickBlockAxis       ≈ Scheduler::BlockSplit (+ the §3.5 row-loop
//     │                         degradation when no ub axis is available)
//     └─ emit one TilePlan   ≈ Scheduler::TileSplit (ubSplit) over the axes
//
// Status: P1 (refactor into classifyAxes / pickBlockAxis / in-order ubSplit)
// + P3a (compute per-operand `vectorizedDims` and derive the row-loop
// degradation from "the block axis is followed by a parallel axis inside a
// reduce's vectorized region" instead of the old `splitParallel` heuristic —
// behavior-identical for current shapes, but principled).  Still pending:
// FullLoad/RCore reduce templates (P3b), leading-run block fusion + transpose
// templates (P2+P4), tiling-case enumeration + cost model (P5), UB peak-memory
// constraints (P6).  See the plan doc.
// ===========================================================================

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

namespace {

// (Axis classification — classifyAxes / transposePerm — moved to
// TileFuseUtils; the Collapse pass computes it and stores it in
// CollapsedGroupInfo::grouping, which genVectorTilePlan reads below.)

// ≈ AutoFuse's `tensor.attr.vectorized_axis` (the inner axes a vector op
// processes whole, that must not be looped/ub-tiled).  For a reduce member,
// that region is { its reduction iteration dims } ∪ { iteration dims that, in
// some operand's affine_map, appear *after* a reduction dim } — because the
// AscendC reduce intrinsic consumes a contiguous [R,A]/[A,R] tile, so anything
// inner to the reduction in the operand layout has to stay whole.  Non-reduce
// members contribute nothing here.  Returns the union over all members; also
// fills `plan.vectorizedDims` per operand Value (in iteration-dim ids).
DenseSet<int> computeVectorizedDims(const CollapsedGroupInfo &info,
                                     TilePlan &plan) {
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
      if (!opVecDims.empty())
        plan.vectorizedDims[operand] = std::move(opVecDims);
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
// where d2 is vectorized).  For the shapes that reach here this is equivalent
// to the old `numParallel≥2 && numReduction≥1`, but it states the real reason.
// P2 replaces the "first axis" with a fused leading run.
BlockPick pickBlockAxis(const AxisGrouping &g, const DenseSet<int> &vecDims) {
  BlockPick bp;
  for (int i : g.yAxes) {
    if (g.axes[i].isBroadcastSplit || vecDims.count(i))
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
      if (!g.axes[i].isBroadcastSplit) {
        bp.axis = i;
        break;
      }
  }
  if (bp.axis >= 0)
    for (int i : g.yAxes)
      if (i > bp.axis && !g.axes[i].isBroadcastSplit && vecDims.count(i)) {
        bp.degradeToRowLoop = true;
        break;
      }
  return bp;
}

} // namespace

TilePlan genVectorTilePlan(func::FuncOp func,
                            const CollapsedGroupInfo &info,
                            OpBuilder &builder, Location loc,
                            bool enableReductionSplit,
                            int64_t maxFullLoopIters) {
  const AxisGrouping &g = info.grouping; // computed by the Collapse pass

  TilePlan plan;
  plan.group = &info;
  if (!g.rAxes.empty())
    plan.reduceTemplate = TilePlan::ReduceTemplate::Common; // P3a: only Common

  DenseSet<int> vecDims = computeVectorizedDims(info, plan);
  BlockPick     bp      = pickBlockAxis(g, vecDims);
  if (bp.axis >= 0)
    plan.blockFusedAxes.push_back(bp.axis);

  // Auto reduction-split: if the user did not request --enable-reduction-split
  // but a reduction axis is so large that its on-chip tile clearly won't fit
  // (R · elem_bytes > budget), switch that axis to the RBLOCK split path.
  DenseSet<int> autoSplitR;
  if (!enableReductionSplit && !bp.degradeToRowLoop && !info.topoMembers.empty()) {
    unsigned elemBytes = 4;
    if (auto st = dyn_cast<ShapedType>(
            info.topoMembers[0]->getOperand(0).getType()))
      if (st.getElementType().isIntOrFloat())
        elemBytes = std::max(1u, st.getElementType().getIntOrFloatBitWidth() / 8);
    constexpr int64_t kReductionTileBudgetBytes = 32 * 1024;
    for (int i : g.rAxes) {
      int64_t sz = info.collapsedAxes[i].staticSize;
      if (sz != ShapedType::kDynamic &&
          sz * (int64_t)elemBytes > kReductionTileBudgetBytes)
        autoSplitR.insert(i);
    }
  }

  // --- ubSplit: one in-order pass over the collapsed axes (≈ TileSplit) ---
  // Counters preserved from the previous implementation so func-arg insertion
  // order and `vector_plan.tiling_infos` numbering are byte-identical.
  int parallelSeen = 0, bcastCount = 0, bcastTileCount = 0, rblockCount = 0;
  int naxisCount = 0, xsubCount = 0;

  for (int i = 0; i < (int)info.collapsedAxes.size(); ++i) {
    const AxisClass &ax = g.axes[i];
    Value ext = getAxisExtentValue(builder, loc, info, i);

    if (ax.kind == AxisKind::X) {
      // Transpose X axis (input-side divergent): inner-tiled with its own
      // tunable; never the block axis (bindMultiCore == false).  Until the §5
      // transpose schedule generator gives X its own 16-fractal handling, this
      // is just an ordinary inner tile.
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
      // tiled, not looped — whole-dim slice (≈ AF's n_group / a "vectorized"
      // axis the transpose primitive processes whole).
      std::string name = llvm::formatv("NAXIS_{0}", naxisCount++).str();
      plan.full.push_back({name, ext, OpFoldResult(ext), i, TileLevel::Full,
                            AxisRole::Parallel});

    } else if (ax.kind == AxisKind::Y && !ax.isBroadcastSplit) {
      if (i == bp.axis) {
        // Block axis: XBLOCK (Outer) + inner level (XBLOCK_SUB, or step-1 row
        // loop under the degradation).
        Value xblock = insertFuncArg(func, builder, loc, 128, "XBLOCK");
        SmallVector<TileParam> group;
        group.push_back({"XBLOCK", xblock, OpFoldResult(ext), i,
                          TileLevel::Outer, AxisRole::Parallel});
        if (bp.degradeToRowLoop) {
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
      } else if (bp.degradeToRowLoop) {
        // Non-block parallel axis under the row-loop degradation: no plan
        // entry → SliceComputer yields the full-dim slice and no loop.
      } else {
        // Non-block parallel axis: inner-tiled only.
        std::string name =
            llvm::formatv("XBLOCK_SUB_{0}", parallelSeen - 1).str();
        Value param = insertFuncArg(func, builder, loc, 16, name);
        SmallVector<TileParam> group;
        group.push_back({name, param, OpFoldResult(ext), i,
                          TileLevel::Inner, AxisRole::Parallel});
        plan.tileable.push_back(std::move(group));
      }
      ++parallelSeen;

    } else if (ax.isBroadcastSplit) {
      // BCast axis: Full step-1 when small, inner-tiled (BCAST_TILE) otherwise.
      bool escape =
          (info.collapsedAxes[i].staticSize == ShapedType::kDynamic) ||
          (info.collapsedAxes[i].staticSize > maxFullLoopIters);
      if (!escape) {
        std::string name = llvm::formatv("BCAST_{0}", bcastCount).str();
        Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
        plan.full.push_back({name, step, OpFoldResult(ext), i,
                              TileLevel::Full, AxisRole::Parallel});
      } else {
        std::string name =
            llvm::formatv("BCAST_TILE_{0}", bcastTileCount++).str();
        Value param = insertFuncArg(func, builder, loc, 16, name);
        SmallVector<TileParam> group;
        group.push_back({name, param, OpFoldResult(ext), i,
                          TileLevel::Inner, AxisRole::Parallel});
        plan.tileable.push_back(std::move(group));
      }
      ++bcastCount;

    } else { // AxisKind::R — reduction axis.
      std::string name = llvm::formatv("RBLOCK_{0}", rblockCount++).str();
      if (!enableReductionSplit && !autoSplitR.count(i)) {
        plan.full.push_back({name, ext, OpFoldResult(ext), i,
                              TileLevel::Full, AxisRole::Reduction});
      } else {
        Value param = insertFuncArg(func, builder, loc, 64, name);
        SmallVector<TileParam> group;
        group.push_back({name, param, OpFoldResult(ext), i,
                          TileLevel::Inner, AxisRole::Reduction});
        plan.tileable.push_back(std::move(group));
        plan.ubTilingAxisR = i;
      }
    }
  }
  return plan;
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

      assert(ba.getArgNumber() <= (unsigned)INT32_MAX && "arg_index overflow");
      NamedAttrList fieldAttrs;
      fieldAttrs.append("abi_index",
                        IntegerAttr::get(i32Ty, abiIndex));
      fieldAttrs.append("arg_index",
                        IntegerAttr::get(i32Ty, (int32_t)ba.getArgNumber()));
      fieldAttrs.append("default_value",
                        IntegerAttr::get(i64Ty, defaultVal));
      fieldAttrs.append("kind", StringAttr::get(ctx, "tunable"));
      fieldAttrs.append("name", StringAttr::get(ctx, tp.name));
      fields.push_back(fieldAttrs.getDictionary(ctx));
      ++abiIndex;
    }
  }

  NamedAttrList entryAttrs;
  entryAttrs.append("fields", ArrayAttr::get(ctx, fields));
  entryAttrs.append("kernel_id", StringAttr::get(ctx, func.getName()));

  StringRef attrName = "vector_plan.tiling_infos";
  SmallVector<Attribute> infos;
  if (auto existing = moduleOp->getAttrOfType<ArrayAttr>(attrName))
    llvm::append_range(infos, existing.getValue());
  infos.push_back(entryAttrs.getDictionary(ctx));
  moduleOp->setAttr(attrName, ArrayAttr::get(ctx, infos));
}

} // namespace mlir::afir
