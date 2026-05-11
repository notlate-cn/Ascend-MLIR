#include "TilePlanGen.h"
#include "TileFuseUtils.h"
#include "Conversion/VectorPlan/TilePlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
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
// Phase 1 keeps this behavior-identical to the previous monolithic version:
// only Y/R axis kinds are produced, the block axis is the *first* non-broadcast
// parallel axis (no fusing of a leading run yet — P2), and the row-loop
// degradation fires exactly where the old `splitParallel` branch did.  Later
// phases add: leading-run block fusion (P2), full reduce model + per-operand
// vectorized-dims (P3), transpose templates (P4), tiling-case enumeration +
// cost model (P5), and UB peak-memory constraints (P6).
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

// ≈ TilingGroup::GenTilingGroup — classify each collapsed iteration axis.
// P1: non-broadcast parallel → Y, reduction → R, broadcast → Y (with the
// isBroadcastSplit flag).  X/N (transpose / concat / split / gather) are
// future work; the ops that produce them don't reach this path today.
AxisGrouping classifyAxes(const CollapsedGroupInfo &info) {
  AxisGrouping g;
  DenseSet<int> bcastSet(info.broadcastAxes.begin(), info.broadcastAxes.end());
  g.axes.resize(info.collapsedAxes.size());
  for (int i = 0; i < (int)info.collapsedAxes.size(); ++i) {
    AxisClass &ax = g.axes[i];
    ax.origPos = i;
    g.axesOrder.push_back(i);
    if (info.collapsedAxes[i].role == AxisRole::Reduction) {
      ax.kind = AxisKind::R;
      ax.isReduceSplit = true;
      g.rAxes.push_back(i);
    } else {
      ax.kind = AxisKind::Y;
      ax.isBroadcastSplit = bcastSet.count(i);
      ax.bindMultiCore = !ax.isBroadcastSplit;
      g.yAxes.push_back(i);
    }
  }
  return g;
}

struct BlockPick {
  int  axis = -1;             // first non-broadcast parallel axis (the block axis)
  bool degradeToRowLoop = false;
};

// ≈ Scheduler::BlockSplit + the §3.5 "no valid ub axis ⇒ block-axis row loop"
// degradation.  P1: the block axis is the first non-broadcast parallel axis.
// The degradation fires exactly when AutoFuse's reduce model would lock every
// other parallel axis inside the reduce's vectorized region — which, for the
// op shapes that reach here, is precisely "≥2 non-broadcast parallel axes
// separated by ≥1 reduction axis" (the old `splitParallel` condition).  P3
// replaces this with the real per-operand vectorized-dims computation.
BlockPick pickBlockAxis(const AxisGrouping &g) {
  BlockPick bp;
  int numParallelNonBcast = 0;
  for (int i : g.yAxes) {
    if (g.axes[i].isBroadcastSplit)
      continue;
    ++numParallelNonBcast;
    if (bp.axis < 0)
      bp.axis = i;
  }
  bp.degradeToRowLoop = numParallelNonBcast >= 2 && !g.rAxes.empty();
  return bp;
}

} // namespace

TilePlan genVectorTilePlan(func::FuncOp func,
                            const CollapsedGroupInfo &info,
                            OpBuilder &builder, Location loc,
                            bool enableReductionSplit,
                            int64_t maxFullLoopIters) {
  AxisGrouping g  = classifyAxes(info);
  BlockPick    bp = pickBlockAxis(g);

  TilePlan plan;
  plan.group = &info;
  if (!g.rAxes.empty())
    plan.reduceTemplate = TilePlan::ReduceTemplate::Common; // P1: only Common
  if (bp.axis >= 0)
    plan.blockFusedAxes.push_back(bp.axis);

  // --- ubSplit: one in-order pass over the collapsed axes (≈ TileSplit) ---
  // Counters preserved from the previous implementation so func-arg insertion
  // order and `vector_plan.tiling_infos` numbering are byte-identical.
  int parallelSeen = 0, bcastCount = 0, bcastTileCount = 0, rblockCount = 0;

  for (int i = 0; i < (int)info.collapsedAxes.size(); ++i) {
    const AxisClass &ax = g.axes[i];
    Value ext = getAxisExtentValue(builder, loc, info, i);

    if (ax.kind == AxisKind::Y && !ax.isBroadcastSplit) {
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
      if (!enableReductionSplit) {
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
