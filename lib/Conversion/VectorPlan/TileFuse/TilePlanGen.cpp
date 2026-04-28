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

TilePlan genVectorTilePlan(func::FuncOp func,
                            const CollapsedGroupInfo &info,
                            OpBuilder &builder, Location loc,
                            bool enableReductionSplit,
                            int64_t maxFullLoopIters) {
  TilePlan plan;
  plan.group = &info;

  DenseSet<int> bcastSet(info.broadcastAxes.begin(), info.broadcastAxes.end());
  int parallelCount = 0, bcastCount = 0, bcastTileCount = 0, rblockCount = 0;

  for (int i = 0; i < (int)info.collapsedAxes.size(); ++i) {
    const AxisInfo &ax = info.collapsedAxes[i];
    Value ext = getAxisExtentValue(builder, loc, info, i);

    if (ax.role == AxisRole::Parallel && !bcastSet.count(i)) {
      // Non-BCast parallel axis.
      if (parallelCount == 0) {
        Value xblock    = insertFuncArg(func, builder, loc, 128, "XBLOCK");
        Value xblockSub = insertFuncArg(func, builder, loc, 16,  "XBLOCK_SUB");
        SmallVector<TileParam> group;
        group.push_back({"XBLOCK",     xblock,    OpFoldResult(ext),    i,
                          TileLevel::Outer, AxisRole::Parallel});
        group.push_back({"XBLOCK_SUB", xblockSub, OpFoldResult(xblock), i,
                          TileLevel::Inner, AxisRole::Parallel});
        plan.tileable.push_back(std::move(group));
        // blockDimExprs: ceildiv(extent, XBLOCK).
        Value blockCount =
            builder.create<arith::CeilDivSIOp>(loc, ext, xblock);
        plan.blockDimExprs.push_back(OpFoldResult(blockCount));
      } else {
        std::string name =
            llvm::formatv("XBLOCK_SUB_{0}", parallelCount - 1).str();
        Value param = insertFuncArg(func, builder, loc, 16, name);
        SmallVector<TileParam> group;
        group.push_back({name, param, OpFoldResult(ext), i,
                          TileLevel::Inner, AxisRole::Parallel});
        plan.tileable.push_back(std::move(group));
      }
      ++parallelCount;

    } else if (bcastSet.count(i)) {
      // BCast axis: placeholder Full (Task 4 replaces this with proper logic).
      bool escape = (ax.staticSize == ShapedType::kDynamic) ||
                    (ax.staticSize > maxFullLoopIters);
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

    } else {
      // Reduction axis placeholder (Task 6 handles split).
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
