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

#include "Analysis/SymbolicShape/DimSymbolTable.h"
#include "Analysis/SymbolicShape/SymExpr.h"
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
  SmallVector<DimKey> dimKeys;        // unique canonical keys, insertion order
  SmallVector<memref::DimOp> dimOps; // one entry per op (may repeat key)

  // When afir-symbolize-shapes ran, two (argN, dimIdx) pairs that the linalg op
  // proved equal share one root symbol -- fold both onto the root's (arg, dim)
  // so they collapse to a single TilingData field instead of e.g. emitting both
  // dim_arg3_0 and dim_arg0_0.  Falls back to identity when the attrs aren't
  // present (fully-static kernel, or symbolize didn't run).
  auto dimSymsAttr = func->getAttrOfType<ArrayAttr>("afir.dim_symbols");
  std::optional<mlir::afir::symshape::DimSymbolTable> symTable;
  if (dimSymsAttr)
    symTable = mlir::afir::symshape::DimSymbolTable::fromAttr(dimSymsAttr);
  auto canonicalize = [&](unsigned argN, int64_t dimIdx) -> DimKey {
    if (symTable && dimIdx >= 0) {
      if (auto a = func.getArgAttrOfType<StringAttr>(argN, "afir.symbolic_shape")) {
        if (auto list = mlir::afir::symshape::parseSymExprList(a.getValue())) {
          if ((size_t)dimIdx < list->size()) {
            const auto &e = (*list)[dimIdx];
            if (e.getKind() == mlir::afir::symshape::SymExpr::Kind::Sym &&
                e.getSym() < symTable->numRoots()) {
              auto src = symTable->sourceOf(e.getSym());
              return DimKey{src.first, (int64_t)src.second};
            }
          }
        }
      }
    }
    return DimKey{argN, dimIdx};
  };

  auto addDimKey = [&](unsigned argNum, int64_t dimIdx) {
    DimKey key = canonicalize(argNum, dimIdx);
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
  // Walk the subview chain transitively (subview-of-subview, through casts)
  // to find the root BlockArgument.
  func.walk([&](ascendc::GlobalTensorSetGlobalBufferOp sgbOp) {
    Value buf = sgbOp.getBuffer();
    // Walk through subview/cast chain to find the root BlockArgument.
    while (buf) {
      if (auto ba = dyn_cast<BlockArgument>(buf)) {
        if (auto memTy = dyn_cast<MemRefType>(ba.getType())) {
          for (int64_t d = 0; d < memTy.getRank(); ++d)
            addDimKey(ba.getArgNumber(), d);
        }
        break;
      }
      if (auto sv = buf.getDefiningOp<memref::SubViewOp>()) {
        buf = sv.getSource();
        continue;
      }
      if (auto castOp = buf.getDefiningOp<memref::CastOp>()) {
        buf = castOp.getSource();
        continue;
      }
      break; // unrecognized op, stop
    }
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

  // ── 2. Collect tiling args ────────────────────────────────────────────────
  // Phase B: read from vector_plan.tiling_infos if present.
  // Phase A fallback: scan i64 block args with positional default names.
  SmallVector<BlockArgument> tilingArgs;
  SmallVector<std::string>   tilingArgNames;

  if (auto moduleOp = func->getParentOfType<ModuleOp>()) {
    if (auto tilingInfosAttr =
            moduleOp->getAttrOfType<ArrayAttr>("vector_plan.tiling_infos")) {
      for (Attribute infoAttr : tilingInfosAttr) {
        auto info = dyn_cast<DictionaryAttr>(infoAttr);
        if (!info) continue;
        auto kid = dyn_cast_or_null<StringAttr>(info.get("kernel_id"));
        if (!kid || kid.getValue() != func.getName())
          continue;
        auto fieldsAttr = dyn_cast_or_null<ArrayAttr>(info.get("fields"));
        if (!fieldsAttr) break;
        for (Attribute fa : fieldsAttr) {
          auto field = dyn_cast<DictionaryAttr>(fa);
          if (!field) continue;
          // Schema v2: shape_derived fields lack arg_index; skip them here.
          auto argIdxAttr = dyn_cast_or_null<IntegerAttr>(field.get("arg_index"));
          auto nameAttr   = dyn_cast_or_null<StringAttr>(field.get("name"));
          if (!argIdxAttr || !nameAttr) continue;
          unsigned argIdx = (unsigned)argIdxAttr.getValue().getSExtValue();
          if (argIdx >= entry.getNumArguments()) {
            func.emitError("vector_plan.tiling_infos arg_index ")
                << argIdx << " out of range for func " << func.getName();
            return failure();
          }
          tilingArgs.push_back(cast<BlockArgument>(entry.getArgument(argIdx)));
          tilingArgNames.push_back(nameAttr.getValue().str());
        }
        break;
      }
    }
  }
  // Phase A fallback: all i64 block args.
  if (tilingArgs.empty()) {
    static const char *kPhaseANames[] = {"TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"};
    unsigned i = 0;
    for (BlockArgument arg : entry.getArguments()) {
      if (arg.getType().isInteger(64)) {
        tilingArgs.push_back(arg);
        tilingArgNames.push_back(
            i < std::size(kPhaseANames) ? kPhaseANames[i++] : "field");
      }
    }
  }

  // ── 3. Build TilingData field names ──────────────────────────────────────
  // Phase B: use names from tiling.infos (tilingArgNames already populated).
  // Phase A fallback: tilingArgNames populated from positional defaults above.
  SmallVector<std::string> tilingNameStorage;
  tilingNameStorage.reserve(tilingArgs.size() + dimKeys.size());
  for (const std::string &n : tilingArgNames)
    tilingNameStorage.push_back(n);
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

  // ── 6. Replace tiling arg uses with tiling fields ────────────────────────
  // Invariant: tilingArgs[i] ↔ tilingArgNames[i] ↔ tilingFieldVals[i].
  assert(tilingArgs.empty() || hasTilingData);
  // Phase B args are index-typed: cast i64 member value → index before replace.
  for (unsigned i = 0; i < tilingArgs.size(); ++i) {
    Value fieldVal = tilingFieldVals[i]; // always i64 from emitasc.member
    if (tilingArgs[i].getType().isIndex()) {
      // builder insertion point is still at start of entry after step 5.
      fieldVal = builder.create<arith::IndexCastOp>(
          func.getLoc(), IndexType::get(ctx), fieldVal);
    }
    tilingArgs[i].replaceAllUsesWith(fieldVal);
  }

  // ── 7. Replace memref.dim uses with index-cast of tiling fields ──────────
  // dimKeys[k] corresponds to tilingFieldVals[tilingArgs.size() + k].
  unsigned dimFieldBase = tilingArgs.size();

  // Helper: look up the i64 tiling field Value for a (argNumber, dimIndex) key.
  // Returns a null Value if the key was not collected.
  auto getDimI64Value = [&](unsigned argNum, int64_t dimIdx) -> Value {
    DimKey key = canonicalize(argNum, dimIdx);
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
    DimKey key = canonicalize(arg.getArgNumber(), dimIdxVal);
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
  //
  // Also captures (argNumber → dynamic dim sizes) so that 7b can compute flat
  // offsets for subviews that reference the promoted args without needing a
  // memref.dim op (which ascir-translate cannot emit).
  DenseMap<unsigned, SmallVector<Value>> promotedArgDynSizes;
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
      SmallVector<int64_t> strides(origTy.getRank(), ShapedType::kDynamic);
      auto stridedLayout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides);
      auto stridedTy = MemRefType::get(origTy.getShape(), origTy.getElementType(),
                                        stridedLayout);
      // Save dynamic sizes before erasing.
      SmallVector<Value> dynSizes(allocOp.getDynamicSizes());
      BlockArgument newArg = entry.addArgument(stridedTy, allocOp.getLoc());
      promotedArgDynSizes[newArg.getArgNumber()] = dynSizes;
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
    if (!op.getBuffer().getDefiningOp())
      return;
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
      // Get col stride: prefer pre-collected tiling field (cheaper), fall back
      // to the dynamic size captured during alloc promotion in step 7a (needed
      // for allocs promoted after the dim-collection phase, which have no
      // corresponding tiling field).
      Value colStride;
      if (Value colStrideI64 = getDimI64Value(baseArg.getArgNumber(), 1)) {
        colStride = b.create<arith::IndexCastOp>(loc, indexTy, colStrideI64);
      } else {
        auto dynIt = promotedArgDynSizes.find(baseArg.getArgNumber());
        if (dynIt != promotedArgDynSizes.end() && dynIt->second.size() >= 2) {
          colStride = dynIt->second[1];
        } else {
          // Fallback: read row stride from the subview's type directly.
          auto svTy = cast<MemRefType>(subview.getResult().getType());
          auto [typeStrides, typeOffset] = svTy.getStridesAndOffset();
          if (!typeStrides.empty() && typeStrides[0] != ShapedType::kDynamic) {
            colStride = b.create<arith::ConstantIndexOp>(loc, typeStrides[0]);
          } else {
            continue;
          }
        }
      }
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

  // ── 7c. Convert remaining GM→GM memref.copy to flat-pointer memmove ─────
  // These arise from concat insert_slice ops that bufferize to copies between
  // GM allocs (now promoted to block args) and subviews of the output arg.
  // AscendC has no GM→GM DataCopy primitive; we emit a verbatim memmove.
  //
  // Helper: walk a value up through subviews/casts to find the root
  // BlockArgument and accumulate a flat element offset.
  // Returns {BlockArgument, flat_index_offset} or {null, null} on failure.
  auto resolveGMChain = [&](Value start, OpBuilder &b,
                             Location loc) -> std::pair<BlockArgument, Value> {
    Value cur = start;
    Value acc = b.create<arith::ConstantIndexOp>(loc, 0);
    // For 2D subviews we need col stride to convert (row,col) to flat offset.
    // Accumulate row and col offsets separately when inside a 2D subview.
    while (true) {
      if (auto sv = cur.getDefiningOp<memref::SubViewOp>()) {
        SmallVector<OpFoldResult> offs = sv.getMixedOffsets();
        if (offs.size() == 1) {
          acc = b.create<arith::AddIOp>(loc, acc, materializeOffset(b, loc, offs[0]));
          cur = sv.getSource();
          continue;
        }
        if (offs.size() >= 2) {
          // 2D: flat_offset += row * col_stride + col
          Value src = sv.getSource();
          if (auto castOp = src.getDefiningOp<memref::CastOp>())
            src = castOp.getSource();
          auto ba = dyn_cast<BlockArgument>(src);
          if (!ba)
            return {BlockArgument{}, Value{}};
          // Get col stride.
          Value colStride;
          if (Value ci64 = getDimI64Value(ba.getArgNumber(), 1)) {
            colStride = b.create<arith::IndexCastOp>(loc, indexTy, ci64);
          } else {
            auto dynIt = promotedArgDynSizes.find(ba.getArgNumber());
            if (dynIt != promotedArgDynSizes.end() && dynIt->second.size() >= 2) {
              colStride = dynIt->second[1];
            } else {
              // Fallback: read row stride from the subview's type directly.
              auto svTy = cast<MemRefType>(sv.getResult().getType());
              auto [typeStrides, typeOffset] = svTy.getStridesAndOffset();
              if (!typeStrides.empty() && typeStrides[0] != ShapedType::kDynamic) {
                colStride = b.create<arith::ConstantIndexOp>(loc, typeStrides[0]);
              } else {
                return {BlockArgument{}, Value{}};
              }
            }
          }
          Value row = materializeOffset(b, loc, offs[0]);
          Value col = materializeOffset(b, loc, offs[1]);
          Value rowFlat = b.create<arith::MulIOp>(loc, row, colStride);
          Value flat2d = b.create<arith::AddIOp>(loc, rowFlat, col);
          acc = b.create<arith::AddIOp>(loc, acc, flat2d);
          return {ba, acc};
        }
        return {BlockArgument{}, Value{}}; // 0-D, unexpected
      }
      if (auto castOp = cur.getDefiningOp<memref::CastOp>()) {
        cur = castOp.getSource();
        continue;
      }
      if (auto ba = dyn_cast<BlockArgument>(cur))
        return {ba, acc};
      return {BlockArgument{}, Value{}};
    }
  };

  {
    SmallVector<memref::CopyOp> gmCopies;
    func.walk([&](memref::CopyOp op) {
      auto srcMs = cast<MemRefType>(op.getSource().getType()).getMemorySpaceAsInt();
      auto dstMs = cast<MemRefType>(op.getTarget().getType()).getMemorySpaceAsInt();
      if (srcMs == 0 && dstMs == 0)
        gmCopies.push_back(op);
    });
    for (memref::CopyOp copyOp : gmCopies) {
      OpBuilder b(copyOp);
      Location loc = copyOp.getLoc();
      Value src = copyOp.getSource();
      Value dst = copyOp.getTarget();

      auto [srcArg, srcOff] = resolveGMChain(src, b, loc);
      auto [dstArg, dstOff] = resolveGMChain(dst, b, loc);

      if (!srcArg || !dstArg) {
        LLVM_DEBUG(llvm::dbgs() << "[prepare-emit] unresolved GM copy\n");
        continue;
      }

      // Build flat GM pointer types (memory_space = kGMSpace = 22).
      Type srcElem = cast<MemRefType>(srcArg.getType()).getElementType();
      Type dstElem = cast<MemRefType>(dstArg.getType()).getElementType();
      auto mkFlatTy = [&](Type elem) {
        return MemRefType::get({ShapedType::kDynamic}, elem,
            MemRefLayoutAttrInterface{},
            IntegerAttr::get(IntegerType::get(ctx, 32), kGMSpace));
      };

      Value srcBase = b.create<emitasc::ReinterpretCastOp>(loc, mkFlatTy(srcElem), srcArg);
      Value dstBase = b.create<emitasc::ReinterpretCastOp>(loc, mkFlatTy(dstElem), dstArg);

      Value srcPtr = b.create<emitasc::PtrOffsetOp>(
          loc, mkFlatTy(srcElem), srcBase,
          /*staticOffset=*/IntegerAttr{}, /*dynamicOffset=*/srcOff);
      Value dstPtr = b.create<emitasc::PtrOffsetOp>(
          loc, mkFlatTy(dstElem), dstBase,
          /*staticOffset=*/IntegerAttr{}, /*dynamicOffset=*/dstOff);

      // Compute byte count: product of dynamic sizes from promotedArgDynSizes
      // (src is a promoted alloc arg, so its sizes are known).
      Value byteCount;
      {
        auto dynIt = promotedArgDynSizes.find(srcArg.getArgNumber());
        if (dynIt != promotedArgDynSizes.end() && !dynIt->second.empty()) {
          Value count = b.create<arith::ConstantIndexOp>(loc, 1);
          for (Value sz : dynIt->second)
            count = b.create<arith::MulIOp>(loc, count, sz);
          int64_t elemBytes = srcElem.getIntOrFloatBitWidth() / 8;
          Value elemSize = b.create<arith::ConstantIndexOp>(loc, elemBytes);
          byteCount = b.create<arith::MulIOp>(loc, count, elemSize);
        } else {
          // Fall back: multiply shape dims from type (static only).
          auto mrt = cast<MemRefType>(src.getType());
          int64_t staticElems = 1;
          bool allStatic = true;
          for (int64_t d : mrt.getShape()) {
            if (d == ShapedType::kDynamic) { allStatic = false; break; }
            staticElems *= d;
          }
          if (!allStatic) {
            LLVM_DEBUG(llvm::dbgs() << "[prepare-emit] dynamic GM copy size unknown\n");
            continue;
          }
          int64_t bytes = staticElems * (srcElem.getIntOrFloatBitWidth() / 8);
          byteCount = b.create<arith::ConstantIndexOp>(loc, bytes);
        }
      }

      // Emit: memmove(dst_ptr, src_ptr, byte_count)
      b.create<emitasc::VerbatimOp>(loc,
          b.getStringAttr("memmove($1, $2, $3)"),
          ValueRange{dstPtr, srcPtr, byteCount});
      copyOp.erase();
    }
    // Clean up dead subviews/casts.
    SmallVector<Operation *> dead;
    func.walk([&](Operation *op) {
      if (op->use_empty() && isa<memref::SubViewOp, memref::CastOp>(op))
        dead.push_back(op);
    });
    for (Operation *op : dead)
      op->erase();
  }

  // ── 8. Erase tiling block args (reverse order to keep indices stable) ────
  {
    SmallVector<unsigned> toErase;
    for (BlockArgument arg : tilingArgs)
      toErase.push_back(arg.getArgNumber());
    llvm::sort(toErase, std::greater<unsigned>());
    for (unsigned idx : toErase)
      entry.eraseArgument(idx);
    // entry.eraseArgument does not update FuncOp::arg_attrs; clear it so the
    // attribute count matches the new block arg count after step 9 rebuilds the
    // function type. (TilePlanGen may have set vector_plan.default_tile_size on
    // the now-erased tiling args.)
    if (func->getAttr("arg_attrs"))
      func->removeAttr("arg_attrs");
  }

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
