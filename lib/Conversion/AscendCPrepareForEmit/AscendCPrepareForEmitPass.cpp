//===- AscendCPrepareForEmitPass.cpp - Prepare func for ascir-translate ===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This pass transforms a func.func produced by --linalg-to-ascendc /
// --ascendc-parallelize into the form expected by ascir-translate:
//
//   Input signature:
//     @fc_relu(%A: memref<?x?xf32, strided<[?,?], offset:?>>,
//              %B: memref<?x?xf32, strided<[?,?], offset:?>>,
//              ...
//              %TB_M: i64, %TB_N: i64, %Tb_M: i64, %Tb_N: i64, %t_K: i64)
//       -> memref<...>
//
//   Output signature:
//     @fc_relu(%A: memref<?x?xf32, strided<[?,?], offset:?>>,   // unchanged
//              %B: memref<?x?xf32, strided<[?,?], offset:?>>,
//              ...
//              %tiling: memref<?x!emitasc.py_struct<"TilingData",...>, 22>)
//       attributes {ascendc.aicore, ascendc.global}
//
// Transformation steps:
//   1. Collect all memref.dim %argX, %cI ops in the function body.
//      Each unique (argIdx, dimIdx) pair becomes an i64 field in TilingData,
//      replacing the memref.dim with emitasc.member reads (index-cast to index).
//   2. Collect all i64 tiling args and replace with a TilingData GM pointer.
//   3. At function entry, emit emitasc.copy_struct to read TilingData from GM,
//      then emitasc.member to extract each field.
//   4. Drop the function return value (kernel returns void).
//   5. Set {ascendc.aicore, ascendc.global} attributes on the function.
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#define GEN_PASS_DECL_ASCENDCPREPAREFOREMITPASS
#define GEN_PASS_DEF_ASCENDCPREPAREFOREMITPASS
#include "Conversion/Passes.h.inc"

#define DEBUG_TYPE "ascendc-prepare-for-emit"

using namespace mlir;
using namespace mlir::ascendc;
using namespace mlir::emitasc;

namespace mlir::afir {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// GM memory_space value used by ascir-translate for __gm__ pointers.
static constexpr int64_t kGMSpace = 22;

/// Build the TilingData PyStruct type with the given field names and i64 types.
static emitasc::PyStructType buildTilingDataType(MLIRContext *ctx,
                                                 ArrayRef<StringRef> names) {
  SmallVector<Attribute> typeAttrs, nameAttrs;
  Type i64 = IntegerType::get(ctx, 64);
  for (auto name : names) {
    typeAttrs.push_back(TypeAttr::get(i64));
    nameAttrs.push_back(StringAttr::get(ctx, name));
  }
  return emitasc::PyStructType::get(
      ctx, StringAttr::get(ctx, "TilingData"),
      ArrayAttr::get(ctx, typeAttrs), ArrayAttr::get(ctx, nameAttrs));
}

//===----------------------------------------------------------------------===//
// Main transformation
//===----------------------------------------------------------------------===//

/// Represents a memref.dim query on a block argument: (argNumber, dimIndex).
struct DimKey {
  unsigned argNumber;
  int64_t dimIndex;
  bool operator==(const DimKey &o) const {
    return argNumber == o.argNumber && dimIndex == o.dimIndex;
  }
};

static LogicalResult prepareFunc(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  Block &entry = func.getBody().front();
  Type indexTy = IndexType::get(ctx);
  Type i64Ty = IntegerType::get(ctx, 64);

  // ── 1. Collect memref.dim uses on block arguments ────────────────────────
  // Scan the whole function for `memref.dim %argX, %cI` where %argX is a
  // BlockArgument and %cI is an arith.constant index.  Collect unique
  // (argNumber, dimIndex) pairs in stable order and remember the ops.
  SmallVector<DimKey> dimKeys;        // unique keys, insertion order
  SmallVector<memref::DimOp> dimOps; // one entry per op (may repeat key)

  auto addDimKey = [&](unsigned argNum, int64_t dimIdx) {
    DimKey key{argNum, dimIdx};
    if (llvm::none_of(dimKeys, [&](const DimKey &k) { return k == key; }))
      dimKeys.push_back(key);
  };

  func.walk([&](memref::DimOp dimOp) {
    auto arg = dyn_cast<BlockArgument>(dimOp.getSource());
    if (!arg)
      return;
    // Extract the constant integer value from the index operand.
    // arith.constant with index type stores an IntegerAttr with IndexType.
    Value indexOperand = dimOp.getIndex();
    auto constOp = indexOperand.getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      return;
    auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
    if (!intAttr)
      return;
    int64_t dimIdxConst = intAttr.getValue().getSExtValue();
    addDimKey(arg.getArgNumber(), dimIdxConst);
    dimOps.push_back(dimOp);
  });

  // Also collect dim[1] (row stride = num columns) for every block argument
  // that appears as the base of a memref.subview feeding set_global_buffer.
  // These dims are needed to compute the flat offset for the GM pointer.
  func.walk([&](memref::SubViewOp subview) {
    auto baseArg = dyn_cast<BlockArgument>(subview.getSource());
    if (!baseArg)
      return;
    // Only care about subviews that feed into set_global_buffer.
    bool feedsSgb = llvm::any_of(subview->getUsers(), [](Operation *user) {
      return user->getName().getStringRef() ==
             "ascendc.global_tensor.set_global_buffer";
    });
    if (!feedsSgb)
      return;
    // Ensure dim[0] and dim[1] are both covered so flat offset can be computed.
    addDimKey(baseArg.getArgNumber(), 0);
    addDimKey(baseArg.getArgNumber(), 1);
  });

  // For any newly added dimKeys that don't have a corresponding memref.dim op
  // yet, insert one at the function entry so it gets picked up in step 7.
  // (These are needed when subview bases were not queried by existing dim ops.)
  {
    OpBuilder eb(ctx);
    eb.setInsertionPointToStart(&entry);
    // Build a map of already-present (argNum, dimIdx) → dimOp.
    for (const DimKey &key : dimKeys) {
      bool alreadyPresent = llvm::any_of(dimOps, [&](memref::DimOp d) {
        auto a = dyn_cast<BlockArgument>(d.getSource());
        if (!a) return false;
        auto c = d.getIndex().getDefiningOp<arith::ConstantOp>();
        if (!c) return false;
        auto ia = dyn_cast<IntegerAttr>(c.getValue());
        if (!ia) return false;
        return a.getArgNumber() == key.argNumber &&
               ia.getValue().getSExtValue() == key.dimIndex;
      });
      if (!alreadyPresent) {
        BlockArgument base = entry.getArgument(key.argNumber);
        Value cIdx = eb.create<arith::ConstantIndexOp>(func.getLoc(), key.dimIndex);
        auto newDimOp = eb.create<memref::DimOp>(func.getLoc(), base, cIdx);
        dimOps.push_back(newDimOp);
      }
    }
  }

  // ── 2. Collect i64 tiling args ───────────────────────────────────────────
  SmallVector<BlockArgument> i64Args;
  for (BlockArgument arg : entry.getArguments()) {
    if (arg.getType().isInteger(64))
      i64Args.push_back(arg);
  }

  // ── 3. Build TilingData field names ─────────────────────────────────────
  // Order: i64 tile-size args first, then dim fields.
  static const char *kDefaultTileNames[] = {"TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"};
  static const unsigned kDefaultCount =
      sizeof(kDefaultTileNames) / sizeof(kDefaultTileNames[0]);

  // Build all names upfront in a stable vector so StringRefs stay valid.
  SmallVector<std::string> tilingNameStorage;
  tilingNameStorage.reserve(i64Args.size() + dimKeys.size());

  // i64 tile-size args
  for (unsigned i = 0; i < i64Args.size(); ++i)
    tilingNameStorage.push_back(i < kDefaultCount ? kDefaultTileNames[i] : "field");
  // dim fields: "dim_argN_D"
  for (const DimKey &key : dimKeys)
    tilingNameStorage.push_back("dim_arg" + std::to_string(key.argNumber) +
                                "_" + std::to_string(key.dimIndex));

  // StringRefs into the stable storage (no realloc after reserve).
  SmallVector<StringRef> tilingNames;
  for (const std::string &s : tilingNameStorage)
    tilingNames.push_back(s);

  // ── 4. Build TilingData struct type and GM pointer arg type ──────────────
  bool hasTilingData = !tilingNames.empty();
  emitasc::PyStructType tilingStructTy;
  Type tilingArgTy;
  if (hasTilingData) {
    tilingStructTy = buildTilingDataType(ctx, tilingNames);
    tilingArgTy = MemRefType::get(
        {ShapedType::kDynamic}, tilingStructTy, MemRefLayoutAttrInterface{},
        IntegerAttr::get(IntegerType::get(ctx, 32), kGMSpace));
  }

  // ── 5. Insert TilingData block arg and extract all fields ────────────────
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(&entry);

  SmallVector<Value> tilingFieldVals; // indexed by tilingNames position
  if (hasTilingData) {
    BlockArgument tilingArg = entry.addArgument(tilingArgTy, func.getLoc());
    Value localStruct =
        builder.create<emitasc::CopyStructOp>(func.getLoc(), tilingStructTy, tilingArg);
    for (unsigned i = 0; i < tilingNames.size(); ++i) {
      tilingFieldVals.push_back(builder.create<emitasc::MemberOp>(
          func.getLoc(), i64Ty, localStruct,
          builder.getStringAttr(tilingNames[i])));
    }
  }

  // ── 6. Replace i64 arg uses with tiling fields ──────────────────────────
  for (unsigned i = 0; i < i64Args.size(); ++i)
    i64Args[i].replaceAllUsesWith(tilingFieldVals[i]);

  // ── 7. Replace memref.dim uses with index-cast of tiling fields ──────────
  // dimKeys[k] corresponds to tilingFieldVals[i64Args.size() + k].
  unsigned dimFieldBase = i64Args.size();

  // Helper: look up the i64 tiling field Value for a (argNumber, dimIndex) key.
  // Returns a null Value if the key was not collected.
  auto getDimI64Value = [&](unsigned argNum, int64_t dimIdx) -> Value {
    DimKey key{argNum, dimIdx};
    auto it = llvm::find_if(dimKeys, [&](const DimKey &d) { return d == key; });
    if (it == dimKeys.end())
      return {};
    unsigned k = it - dimKeys.begin();
    return tilingFieldVals[dimFieldBase + k];
  };

  for (memref::DimOp dimOp : dimOps) {
    auto arg = cast<BlockArgument>(dimOp.getSource());
    auto constOp = dimOp.getIndex().getDefiningOp<arith::ConstantOp>();
    int64_t dimIdxVal = cast<IntegerAttr>(constOp.getValue()).getValue().getSExtValue();
    DimKey key{arg.getArgNumber(), dimIdxVal};
    unsigned k = llvm::find_if(dimKeys, [&](const DimKey &d) { return d == key; }) -
                 dimKeys.begin();
    Value i64Val = tilingFieldVals[dimFieldBase + k];
    // Cast i64 → index to replace the memref.dim (which returns index).
    OpBuilder b(dimOp);
    Value idxVal = b.create<arith::IndexCastOp>(dimOp.getLoc(), indexTy, i64Val);
    dimOp.replaceAllUsesWith(idxVal);
    dimOp.erase();
  }

  // ── 7a. Promote top-level GM memref.alloc to function arguments ─────────────
  // Must run before 7b so that subview chains from the promoted arg are
  // traversable when resolving set_global_buffer ops.
  {
    SmallVector<memref::AllocOp> gmAllocs;
    for (Operation &op : entry.without_terminator()) {
      auto allocOp = dyn_cast<memref::AllocOp>(&op);
      if (!allocOp)
        continue;
      auto mrt = cast<MemRefType>(allocOp.getResult().getType());
      if (mrt.getMemorySpaceAsInt() == 0)
        gmAllocs.push_back(allocOp);
    }
    for (memref::AllocOp allocOp : gmAllocs) {
      auto origTy = cast<MemRefType>(allocOp.getResult().getType());
      SmallVector<int64_t> strides(origTy.getRank(), 1);
      auto stridedLayout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides);
      auto stridedTy = MemRefType::get(origTy.getShape(), origTy.getElementType(),
                                        stridedLayout);
      BlockArgument newArg = entry.addArgument(stridedTy, allocOp.getLoc());
      OpBuilder b(allocOp);
      Value casted = b.create<memref::CastOp>(allocOp.getLoc(), origTy, newArg);
      allocOp.getResult().replaceAllUsesWith(casted);
      allocOp.erase();
    }
  }

  // ── 7b. Replace subview+set_global_buffer with flat-pointer set_global_buffer
  //
  // Handles two patterns:
  //   2D: subview %argX[row, col] [s] [1,1]  → flat_offset = row*col_stride + col
  //   1D: subview chain: subview(subview(argX, [off1]), [off2])
  //                                          → flat_offset = off1 + off2 (additive)
  //
  // Output:
  //   emitasc.reinterpret_cast %argX → memref<?xElem, 22>
  //   ascendc.global_tensor.set_global_buffer %gt, %flat, flat_offset_i32
  //
  Type i32Ty = IntegerType::get(ctx, 32);

  // Helper: materialize an OpFoldResult as an index Value.
  auto materializeOffset = [&](OpBuilder &b, Location loc,
                                OpFoldResult ofr) -> Value {
    if (auto attr = ofr.dyn_cast<Attribute>()) {
      int64_t v = cast<IntegerAttr>(attr).getValue().getSExtValue();
      return b.create<arith::ConstantIndexOp>(loc, v);
    }
    return ofr.get<Value>();
  };

  // Helper: walk up a chain of subview ops to find the root BlockArgument,
  // accumulating a 1D flat offset along the way.  Returns null if the chain
  // does not terminate at a BlockArgument, or if any subview has rank > 1
  // and cannot be reduced to a 1D offset here (2D handled separately below).
  auto resolveSubviewChain1D =
      [&](memref::SubViewOp leaf, OpBuilder &b,
          Location loc) -> std::pair<BlockArgument, Value> {
    Value accOffset = b.create<arith::ConstantIndexOp>(loc, 0);
    Value cur = leaf.getResult();
    while (true) {
      auto sv = cur.getDefiningOp<memref::SubViewOp>();
      if (!sv)
        return {BlockArgument{}, Value{}};
      SmallVector<OpFoldResult> offs = sv.getMixedOffsets();
      if (offs.size() != 1)
        return {BlockArgument{}, Value{}}; // not 1D, give up
      Value off = materializeOffset(b, loc, offs[0]);
      accOffset = b.create<arith::AddIOp>(loc, accOffset, off);
      Value src = sv.getSource();
      // See through memref.cast to the underlying value.
      if (auto castOp = src.getDefiningOp<memref::CastOp>())
        src = castOp.getSource();
      if (auto ba = dyn_cast<BlockArgument>(src))
        return {ba, accOffset};
      cur = src; // continue up the chain
    }
  };

  SmallVector<ascendc::GlobalTensorSetGlobalBufferOp> setGlobalBufferOps;
  func.walk([&](ascendc::GlobalTensorSetGlobalBufferOp op) {
    if (!isa<memref::SubViewOp>(op.getBuffer().getDefiningOp()))
      return;
    setGlobalBufferOps.push_back(op);
  });

  for (ascendc::GlobalTensorSetGlobalBufferOp sgbOp : setGlobalBufferOps) {
    auto subview = sgbOp.getBuffer().getDefiningOp<memref::SubViewOp>();
    if (!subview)
      continue;

    OpBuilder b(sgbOp);
    Location loc = sgbOp.getLoc();

    BlockArgument baseArg;
    Value flatOffset;

    SmallVector<OpFoldResult> mixedOffsets = subview.getMixedOffsets();

    if (mixedOffsets.size() >= 2) {
      // ── 2D case: walk up a chain of 2D subviews to find the root BlockArg.
      // Accumulate row/col offsets at each level; the base stride is taken
      // from the root BlockArgument's dim-1 (column stride).
      Value accRow = b.create<arith::ConstantIndexOp>(loc, 0);
      Value accCol = b.create<arith::ConstantIndexOp>(loc, 0);
      Value cur = subview.getResult();
      bool ok = true;
      while (true) {
        auto sv = cur.getDefiningOp<memref::SubViewOp>();
        if (!sv) {
          ok = false;
          break;
        }
        SmallVector<OpFoldResult> offs = sv.getMixedOffsets();
        if (offs.size() < 2) {
          ok = false;
          break;
        }
        Value rowOff = materializeOffset(b, loc, offs[0]);
        Value colOff = materializeOffset(b, loc, offs[1]);
        accRow = b.create<arith::AddIOp>(loc, accRow, rowOff);
        accCol = b.create<arith::AddIOp>(loc, accCol, colOff);
        Value src = sv.getSource();
        // See through memref.cast.
        if (auto castOp = src.getDefiningOp<memref::CastOp>())
          src = castOp.getSource();
        if (auto ba = dyn_cast<BlockArgument>(src)) {
          baseArg = ba;
          break;
        }
        cur = src;
      }
      if (!ok || !baseArg)
        continue;
      Value colStrideI64 = getDimI64Value(baseArg.getArgNumber(), 1);
      if (!colStrideI64)
        continue;
      Value colStride = b.create<arith::IndexCastOp>(loc, indexTy, colStrideI64);
      Value rowTimesStride = b.create<arith::MulIOp>(loc, accRow, colStride);
      flatOffset = b.create<arith::AddIOp>(loc, rowTimesStride, accCol);
    } else {
      // ── 1D case: walk the subview chain ──────────────────────────────────
      auto [ba, acc] = resolveSubviewChain1D(subview, b, loc);
      if (!ba)
        continue;
      baseArg = ba;
      flatOffset = acc;
    }

    Value flatOffsetI32 = b.create<arith::IndexCastOp>(loc, i32Ty, flatOffset);

    // Cast base memref to flat GM pointer: memref<?xElem, 22>.
    auto baseMemRefTy = cast<MemRefType>(baseArg.getType());
    Type elemTy = baseMemRefTy.getElementType();
    auto flatTy = MemRefType::get(
        {ShapedType::kDynamic}, elemTy, MemRefLayoutAttrInterface{},
        IntegerAttr::get(IntegerType::get(ctx, 32), kGMSpace));
    Value flatBase = b.create<emitasc::ReinterpretCastOp>(loc, flatTy, baseArg);

    b.create<ascendc::GlobalTensorSetGlobalBufferOp>(
        loc, sgbOp.getTensor(), flatBase, flatOffsetI32);
    sgbOp.erase();

    if (subview.use_empty())
      subview.erase();
  }

  // Clean up any dead subview chains left over (e.g. outer subviews that
  // became unused after the inner ones were replaced above).
  {
    SmallVector<memref::SubViewOp> deadSubviews;
    func.walk([&](memref::SubViewOp sv) {
      if (sv.use_empty())
        deadSubviews.push_back(sv);
    });
    for (memref::SubViewOp sv : deadSubviews)
      sv.erase();
  }

  // ── 8. Erase old i64 block args (reverse order) ─────────────────────────
  SmallVector<unsigned> toErase;
  for (BlockArgument arg : i64Args)
    toErase.push_back(arg.getArgNumber());
  llvm::sort(toErase, std::greater<unsigned>());
  for (unsigned idx : toErase)
    entry.eraseArgument(idx);

  // ── 9. Update function type ──────────────────────────────────────────────
  SmallVector<Type> newArgTypes;
  for (BlockArgument arg : entry.getArguments())
    newArgTypes.push_back(arg.getType());
  func.setFunctionType(FunctionType::get(ctx, newArgTypes, /*results=*/{}));

  // ── 10. Add {ascendc.aicore, ascendc.global} attributes ─────────────────
  func->setAttr("ascendc.aicore", UnitAttr::get(ctx));
  func->setAttr("ascendc.global", UnitAttr::get(ctx));

  // ── 11. Remove return values (kernel returns void) ───────────────────────
  func.walk([&](func::ReturnOp ret) {
    if (ret.getNumOperands() > 0) {
      OpBuilder b(ret);
      b.create<func::ReturnOp>(ret.getLoc());
      ret.erase();
    }
  });

  // ── 12. Lower affine.min → arith.minsi ──────────────────────────────────
  // ascir-translate does not support affine ops; lower them to arith here.
  {
    SmallVector<affine::AffineMinOp> minOps;
    func.walk([&](affine::AffineMinOp op) { minOps.push_back(op); });
    for (affine::AffineMinOp minOp : minOps) {
      OpBuilder b(minOp);
      Location loc = minOp.getLoc();
      AffineMap map = minOp.getAffineMap();
      ValueRange mapOperands = minOp.getOperands();

      // Evaluate each result expression of the map.
      SmallVector<Value> results;
      for (AffineExpr expr : map.getResults()) {
        // Expand the affine expression to arith ops.
        Value val = mlir::affine::expandAffineExpr(b, loc, expr, mapOperands.take_front(map.getNumDims()),
                                                   mapOperands.drop_front(map.getNumDims()));
        results.push_back(val);
      }
      // Reduce to a single minimum.
      Value minVal = results[0];
      for (unsigned i = 1; i < results.size(); ++i)
        minVal = b.create<arith::MinSIOp>(loc, minVal, results[i]);
      minOp.getResult().replaceAllUsesWith(minVal);
      minOp.erase();
    }
  }

  // ── 13. Declare TilingData struct at module level ────────────────────────
  if (hasTilingData) {
    if (auto moduleOp = func->getParentOfType<ModuleOp>()) {
      OpBuilder modBuilder(ctx);
      modBuilder.setInsertionPointToStart(moduleOp.getBody());
      modBuilder.create<emitasc::DeclarePyStructOp>(
          func.getLoc(), TypeAttr::get(tilingStructTy));
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct AscendCPrepareForEmitPass
    : public ::impl::AscendCPrepareForEmitPassBase<AscendCPrepareForEmitPass> {
  using AscendCPrepareForEmitPassBase::AscendCPrepareForEmitPassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (failed(prepareFunc(func)))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createAscendCPrepareForEmitPass() {
  return std::make_unique<AscendCPrepareForEmitPass>();
}

} // namespace mlir::afir
