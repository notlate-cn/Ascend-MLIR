//===- PrepareForEmit.cpp - Prepare Ascend kernels for emission -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This pass transforms a func.func produced by ascend-compute-lower and
// ascend-parallelize into the form expected by ascir-translate:
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

#include "PreEmitInternalPasses.h"

#include "Conversion/Ascend/Common/Attributes.h"
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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#include <functional>
#include <optional>

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#define DEBUG_TYPE "ascend-prepare-for-emit"

using namespace mlir;
using namespace mlir::ascendc;
using namespace mlir::emitasc;

namespace mlir::ascend {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// GM memory_space value used by ascir-translate for __gm__ pointers.
static constexpr int64_t kGMSpace = 22;

static constexpr const char *kTilingDataStructName = "TilingData";
static constexpr llvm::StringLiteral kFuncArgAttrsAttr = "arg_attrs";
static constexpr const char *kDefaultTileNames[] = {"TB_M", "TB_N", "Tb_M",
                                                    "Tb_N", "t_K"};
static constexpr unsigned kDefaultTileNameCount =
    sizeof(kDefaultTileNames) / sizeof(kDefaultTileNames[0]);

/// Represents a memref.dim query on a block argument: (argNumber, dimIndex).
struct DimKey {
  unsigned argNumber;
  int64_t dimIndex;
  bool operator==(const DimKey &o) const {
    return argNumber == o.argNumber && dimIndex == o.dimIndex;
  }
};

static void appendUniqueDimKey(SmallVectorImpl<DimKey> &dimKeys,
                               unsigned argNum, int64_t dimIdx) {
  DimKey key{argNum, dimIdx};
  if (llvm::none_of(dimKeys, [&](const DimKey &k) { return k == key; }))
    dimKeys.push_back(key);
}

static std::optional<int64_t> getConstantIndexValue(Value value) {
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;
  auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
  if (!intAttr)
    return std::nullopt;
  return intAttr.getValue().getSExtValue();
}

static std::optional<int64_t> getConstantIndexValue(OpFoldResult value) {
  if (auto attr = value.dyn_cast<Attribute>()) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr)
      return std::nullopt;
    return intAttr.getValue().getSExtValue();
  }
  return getConstantIndexValue(value.get<Value>());
}

static std::optional<int64_t> getStaticMemRefDim(Value source, int64_t dimIdx) {
  auto memTy = dyn_cast<MemRefType>(source.getType());
  if (!memTy || dimIdx < 0 || dimIdx >= memTy.getRank())
    return std::nullopt;
  if (memTy.isDynamicDim(dimIdx))
    return std::nullopt;
  return memTy.getDimSize(dimIdx);
}

static bool collectDimKeysForValue(Value source, int64_t dimIdx,
                                   SmallVectorImpl<DimKey> &dimKeys) {
  auto memTy = dyn_cast<MemRefType>(source.getType());
  if (!memTy || dimIdx < 0 || dimIdx >= memTy.getRank())
    return false;

  if (!memTy.isDynamicDim(dimIdx))
    return true;

  if (auto arg = dyn_cast<BlockArgument>(source)) {
    appendUniqueDimKey(dimKeys, arg.getArgNumber(), dimIdx);
    return true;
  }

  if (auto castOp = source.getDefiningOp<memref::CastOp>())
    return collectDimKeysForValue(castOp.getSource(), dimIdx, dimKeys);

  if (auto collapseOp = source.getDefiningOp<memref::CollapseShapeOp>()) {
    SmallVector<ReassociationIndices, 4> reassociation =
        collapseOp.getReassociationIndices();
    if (dimIdx >= static_cast<int64_t>(reassociation.size()))
      return false;
    for (int64_t sourceDim : reassociation[dimIdx]) {
      if (!collectDimKeysForValue(collapseOp.getSrc(), sourceDim, dimKeys))
        return false;
    }
    return true;
  }

  if (auto expandOp = source.getDefiningOp<memref::ExpandShapeOp>()) {
    SmallVector<OpFoldResult> outputShape = expandOp.getMixedOutputShape();
    if (dimIdx >= static_cast<int64_t>(outputShape.size()))
      return false;
    if (std::optional<int64_t> staticSize =
            getConstantIndexValue(outputShape[dimIdx]))
      return *staticSize != ShapedType::kDynamic;
    return true;
  }

  if (auto subviewOp = source.getDefiningOp<memref::SubViewOp>()) {
    SmallVector<OpFoldResult> sizes = subviewOp.getMixedSizes();
    if (dimIdx >= static_cast<int64_t>(sizes.size()))
      return false;
    if (std::optional<int64_t> staticSize = getConstantIndexValue(sizes[dimIdx]))
      return *staticSize != ShapedType::kDynamic;
    return true;
  }

  return false;
}

static emitasc::PyStructType getTilingStructTypeFromType(Type type) {
  if (auto pyStruct = dyn_cast<emitasc::PyStructType>(type))
    return pyStruct;
  if (auto memrefType = dyn_cast<MemRefType>(type))
    return dyn_cast<emitasc::PyStructType>(memrefType.getElementType());
  return {};
}

static StringAttr getArgTileName(func::FuncOp func, unsigned argNumber) {
  auto argAttrs = func->getAttrOfType<ArrayAttr>(kFuncArgAttrsAttr);
  if (!argAttrs || argNumber >= argAttrs.size())
    return {};
  auto dict = dyn_cast<DictionaryAttr>(argAttrs[argNumber]);
  if (!dict)
    return {};
  return dyn_cast_or_null<StringAttr>(dict.get(kScheduleTileArgAttr));
}

static void appendTilingTypeNames(emitasc::PyStructType tilingType,
                                  SmallVectorImpl<std::string> &names) {
  for (Attribute nameAttr : tilingType.getNamesAttr().getValue())
    names.push_back(cast<StringAttr>(nameAttr).getValue().str());
}

static SmallVector<std::string>
collectProspectiveTilingNames(func::FuncOp func) {
  SmallVector<std::string> names;

  // Already-prepared functions carry their tiling schema in the function type.
  for (BlockArgument arg : func.getArguments()) {
    if (emitasc::PyStructType tilingType =
            getTilingStructTypeFromType(arg.getType())) {
      appendTilingTypeNames(tilingType, names);
      return names;
    }
  }

  SmallVector<BlockArgument> i64Args;
  for (BlockArgument arg : func.getArguments()) {
    if (arg.getType().isInteger(64))
      i64Args.push_back(arg);
  }
  for (unsigned i = 0; i < i64Args.size(); ++i)
    names.push_back(i < kDefaultTileNameCount ? kDefaultTileNames[i]
                                              : "field");

  SmallVector<DimKey> dimKeys;
  func.walk([&](memref::DimOp dimOp) {
    std::optional<int64_t> dimIdx = getConstantIndexValue(dimOp.getIndex());
    if (!dimIdx)
      return;
    collectDimKeysForValue(dimOp.getSource(), *dimIdx, dimKeys);
  });

  func.walk([&](ascendc::GlobalTensorSetGlobalBufferOp sgbOp) {
    Value buf = sgbOp.getBuffer();
    while (buf) {
      if (auto ba = dyn_cast<BlockArgument>(buf)) {
        if (auto memTy = dyn_cast<MemRefType>(ba.getType())) {
          for (int64_t d = 0; d < memTy.getRank(); ++d)
            appendUniqueDimKey(dimKeys, ba.getArgNumber(), d);
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
      break;
    }
  });

  for (const DimKey &key : dimKeys) {
    names.push_back("dim_arg" + std::to_string(key.argNumber) + "_" +
                    std::to_string(key.dimIndex));
  }
  return names;
}

static bool sameNames(ArrayRef<StringRef> lhs, ArrayRef<std::string> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs)) {
    if (left != right)
      return false;
  }
  return true;
}

static std::string sanitizeCppIdentifier(StringRef value) {
  std::string result;
  result.reserve(value.size() + 1);
  for (char c : value) {
    if (llvm::isAlnum(c) || c == '_')
      result.push_back(c);
    else
      result.push_back('_');
  }
  if (result.empty() || llvm::isDigit(result.front()))
    result.insert(result.begin(), '_');
  return result;
}

static bool requiresKernelSpecificTilingDataName(
    func::FuncOp func, ArrayRef<StringRef> localTilingNames) {
  ModuleOp module = func->getParentOfType<ModuleOp>();
  if (!module || localTilingNames.empty())
    return false;

  for (func::FuncOp other : module.getOps<func::FuncOp>()) {
    if (other == func)
      continue;
    SmallVector<std::string> otherNames = collectProspectiveTilingNames(other);
    if (otherNames.empty())
      continue;
    if (!sameNames(localTilingNames, otherNames))
      return true;
  }
  return false;
}

static std::string getTilingDataStructName(func::FuncOp func,
                                           ArrayRef<StringRef> names) {
  if (!requiresKernelSpecificTilingDataName(func, names))
    return kTilingDataStructName;
  return (Twine(kTilingDataStructName) + "_" +
          sanitizeCppIdentifier(func.getName()))
      .str();
}

/// Build the TilingData PyStruct type with the given field names and i64 types.
static emitasc::PyStructType buildTilingDataType(MLIRContext *ctx,
                                                 StringRef structName,
                                                 ArrayRef<StringRef> names) {
  SmallVector<Attribute> typeAttrs, nameAttrs;
  Type i64 = IntegerType::get(ctx, 64);
  for (auto name : names) {
    typeAttrs.push_back(TypeAttr::get(i64));
    nameAttrs.push_back(StringAttr::get(ctx, name));
  }
  return emitasc::PyStructType::get(ctx, StringAttr::get(ctx, structName),
                                    ArrayAttr::get(ctx, typeAttrs),
                                    ArrayAttr::get(ctx, nameAttrs));
}

//===----------------------------------------------------------------------===//
// Main transformation
//===----------------------------------------------------------------------===//

static LogicalResult prepareFunc(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  Block &entry = func.getBody().front();
  Type indexTy = IndexType::get(ctx);
  Type i64Ty = IntegerType::get(ctx, 64);

  // ── 1. Collect memref.dim uses on block arguments ────────────────────────
  // Scan the whole function for constant-index memref.dim ops that can be
  // resolved to function argument dimensions through view-like ops. Collect
  // unique (argNumber, dimIndex) pairs in stable order and remember the ops.
  SmallVector<DimKey> dimKeys;        // unique keys, insertion order
  SmallVector<memref::DimOp> dimOps; // one entry per op (may repeat key)

  func.walk([&](memref::DimOp dimOp) {
    std::optional<int64_t> dimIdxConst = getConstantIndexValue(dimOp.getIndex());
    if (!dimIdxConst)
      return;
    if (collectDimKeysForValue(dimOp.getSource(), *dimIdxConst, dimKeys))
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
            appendUniqueDimKey(dimKeys, ba.getArgNumber(), d);
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

  // ── 2. Collect i64 tiling args ───────────────────────────────────────────
  SmallVector<BlockArgument> i64Args;
  for (BlockArgument arg : entry.getArguments()) {
    if (arg.getType().isInteger(64))
      i64Args.push_back(arg);
  }

  // ── 3. Build TilingData field names ─────────────────────────────────────
  // Order: i64 tile-size args first, then dim fields.
  // Build all names upfront in a stable vector so StringRefs stay valid.
  SmallVector<std::string> tilingNameStorage;
  tilingNameStorage.reserve(i64Args.size() + dimKeys.size());

  // i64 tile-size args
  for (unsigned i = 0; i < i64Args.size(); ++i) {
    if (StringAttr tileArgName =
            getArgTileName(func, i64Args[i].getArgNumber())) {
      tilingNameStorage.push_back(tileArgName.getValue().str());
      continue;
    }
    tilingNameStorage.push_back(i < kDefaultTileNameCount
                                    ? kDefaultTileNames[i]
                                    : "field");
  }
  // dim fields: "dim_argN_D"
  for (const DimKey &key : dimKeys)
    tilingNameStorage.push_back("dim_arg" + std::to_string(key.argNumber) +
                                "_" + std::to_string(key.dimIndex));

  // StringRefs into the stable storage (no realloc after reserve).
  SmallVector<StringRef> tilingNames;
  for (const std::string &s : tilingNameStorage)
    tilingNames.push_back(s);

  // ── 4. Build TilingData struct type and GM pointer arg type ──────────────
  // CANN ABI requires every global kernel to carry a tiling pointer even when
  // the kernel has no dynamic shape fields. Use an empty TilingData struct for
  // fully static kernels so downstream signature canonicalization stays uniform.
  bool hasTilingData = true;
  emitasc::PyStructType tilingStructTy;
  Type tilingArgTy;
  if (hasTilingData) {
    std::string structName = getTilingDataStructName(func, tilingNames);
    tilingStructTy = buildTilingDataType(ctx, structName, tilingNames);
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

  auto materializeIndexValue = [&](OpBuilder &b, Location loc,
                                   OpFoldResult value) -> Value {
    if (std::optional<int64_t> staticValue = getConstantIndexValue(value)) {
      if (*staticValue == ShapedType::kDynamic)
        return {};
      return b.create<arith::ConstantIndexOp>(loc, *staticValue);
    }
    Value dynamicValue = value.get<Value>();
    if (dynamicValue.getType().isIndex())
      return dynamicValue;
    if (dynamicValue.getType().isInteger(64))
      return b.create<arith::IndexCastOp>(loc, indexTy, dynamicValue);
    return {};
  };

  std::function<Value(Value, int64_t, OpBuilder &, Location)> materializeDim;
  materializeDim = [&](Value source, int64_t dimIdx, OpBuilder &b,
                       Location loc) -> Value {
    auto memTy = dyn_cast<MemRefType>(source.getType());
    if (!memTy || dimIdx < 0 || dimIdx >= memTy.getRank())
      return {};

    if (std::optional<int64_t> staticDim = getStaticMemRefDim(source, dimIdx))
      return b.create<arith::ConstantIndexOp>(loc, *staticDim);

    if (auto arg = dyn_cast<BlockArgument>(source)) {
      Value i64Val = getDimI64Value(arg.getArgNumber(), dimIdx);
      if (!i64Val)
        return {};
      return b.create<arith::IndexCastOp>(loc, indexTy, i64Val);
    }

    if (auto castOp = source.getDefiningOp<memref::CastOp>())
      return materializeDim(castOp.getSource(), dimIdx, b, loc);

    if (auto collapseOp = source.getDefiningOp<memref::CollapseShapeOp>()) {
      SmallVector<ReassociationIndices, 4> reassociation =
          collapseOp.getReassociationIndices();
      if (dimIdx >= static_cast<int64_t>(reassociation.size()))
        return {};
      Value product = b.create<arith::ConstantIndexOp>(loc, 1);
      for (int64_t sourceDim : reassociation[dimIdx]) {
        Value factor = materializeDim(collapseOp.getSrc(), sourceDim, b, loc);
        if (!factor)
          return {};
        product = b.create<arith::MulIOp>(loc, product, factor);
      }
      return product;
    }

    if (auto expandOp = source.getDefiningOp<memref::ExpandShapeOp>()) {
      SmallVector<OpFoldResult> outputShape = expandOp.getMixedOutputShape();
      if (dimIdx >= static_cast<int64_t>(outputShape.size()))
        return {};
      return materializeIndexValue(b, loc, outputShape[dimIdx]);
    }

    if (auto subviewOp = source.getDefiningOp<memref::SubViewOp>()) {
      SmallVector<OpFoldResult> sizes = subviewOp.getMixedSizes();
      if (dimIdx >= static_cast<int64_t>(sizes.size()))
        return {};
      return materializeIndexValue(b, loc, sizes[dimIdx]);
    }

    return {};
  };

  for (memref::DimOp dimOp : dimOps) {
    std::optional<int64_t> dimIdxVal = getConstantIndexValue(dimOp.getIndex());
    if (!dimIdxVal)
      continue;
    OpBuilder b(dimOp);
    Value idxVal = materializeDim(dimOp.getSource(), *dimIdxVal, b,
                                  dimOp.getLoc());
    if (!idxVal)
      continue;
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

  // Helper: walk up a chain of subview ops to find the root GM value,
  // accumulating a 1D flat offset along the way.  Returns null if the chain
  // does not terminate at a BlockArgument or memref.get_global, or if any
  // subview has rank > 1 and cannot be reduced to a 1D offset here (2D handled
  // separately below).
  auto resolveSubviewChain1D =
      [&](memref::SubViewOp leaf, OpBuilder &b,
          Location loc) -> std::pair<Value, Value> {
    Value accOffset = b.create<arith::ConstantIndexOp>(loc, 0);
    Value cur = leaf.getResult();
    while (true) {
      auto sv = cur.getDefiningOp<memref::SubViewOp>();
      if (!sv)
        return {Value{}, Value{}};
      SmallVector<OpFoldResult> offs = sv.getMixedOffsets();
      if (offs.size() != 1)
        return {Value{}, Value{}}; // not 1D, give up
      Value off = materializeOffset(b, loc, offs[0]);
      accOffset = b.create<arith::AddIOp>(loc, accOffset, off);
      Value src = sv.getSource();
      // See through memref.cast to the underlying value.
      if (auto castOp = src.getDefiningOp<memref::CastOp>())
        src = castOp.getSource();
      if (auto ba = dyn_cast<BlockArgument>(src))
        return {ba, accOffset};
      if (src.getDefiningOp<memref::GetGlobalOp>())
        return {src, accOffset};
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

    Value baseMemref;
    Value flatOffset;

    SmallVector<OpFoldResult> mixedOffsets = subview.getMixedOffsets();

    if (mixedOffsets.size() >= 2) {
      // ── 2D case: walk up a chain of 2D subviews to find the root BlockArg.
      // Accumulate row/col offsets at each level; the base stride is taken
      // from the root BlockArgument's dim-1 (column stride).
      Value accRow = b.create<arith::ConstantIndexOp>(loc, 0);
      Value accCol = b.create<arith::ConstantIndexOp>(loc, 0);
      Value cur = subview.getResult();
      BlockArgument baseArg;
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
      baseMemref = baseArg;
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
      auto [base, acc] = resolveSubviewChain1D(subview, b, loc);
      if (!base)
        continue;
      baseMemref = base;
      flatOffset = acc;
    }

    Value flatOffsetI32 = b.create<arith::IndexCastOp>(loc, i32Ty, flatOffset);

    // Cast base memref to flat GM pointer: memref<?xElem, 22>.
    auto baseMemRefTy = cast<MemRefType>(baseMemref.getType());
    Type elemTy = baseMemRefTy.getElementType();
    auto flatTy = MemRefType::get(
        {ShapedType::kDynamic}, elemTy, MemRefLayoutAttrInterface{},
        IntegerAttr::get(IntegerType::get(ctx, 32), kGMSpace));
    Value flatBase = b.create<emitasc::ReinterpretCastOp>(loc, flatTy, baseMemref);

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

  // ── 8. Erase old i64 block args (reverse order) ─────────────────────────
  SmallVector<unsigned> toErase;
  for (BlockArgument arg : i64Args)
    toErase.push_back(arg.getArgNumber());

  SmallVector<bool> eraseArg(entry.getNumArguments(), false);
  for (unsigned idx : toErase)
    if (idx < eraseArg.size())
      eraseArg[idx] = true;

  auto oldArgAttrs = func->getAttrOfType<ArrayAttr>(kFuncArgAttrsAttr);
  SmallVector<Attribute> preservedArgAttrs;
  bool hasPreservedArgAttrs = false;
  preservedArgAttrs.reserve(entry.getNumArguments() - toErase.size());
  for (unsigned i = 0, e = entry.getNumArguments(); i < e; ++i) {
    if (eraseArg[i])
      continue;
    SmallVector<NamedAttribute> attrs;
    if (oldArgAttrs && i < oldArgAttrs.size()) {
      if (auto dict = dyn_cast<DictionaryAttr>(oldArgAttrs[i])) {
        for (NamedAttribute attr : dict) {
          if (attr.getName().getValue() == kScheduleTileArgAttr)
            continue;
          attrs.push_back(attr);
        }
      }
    }
    if (!attrs.empty())
      hasPreservedArgAttrs = true;
    preservedArgAttrs.push_back(DictionaryAttr::get(ctx, attrs));
  }

  llvm::sort(toErase, std::greater<unsigned>());
  for (unsigned idx : toErase)
    entry.eraseArgument(idx);

  // ── 9. Update function type ──────────────────────────────────────────────
  SmallVector<Type> newArgTypes;
  for (BlockArgument arg : entry.getArguments())
    newArgTypes.push_back(arg.getType());
  func.setFunctionType(FunctionType::get(ctx, newArgTypes, /*results=*/{}));
  if (hasPreservedArgAttrs)
    func->setAttr(kFuncArgAttrsAttr, ArrayAttr::get(ctx, preservedArgAttrs));
  else
    func->removeAttr(kFuncArgAttrsAttr);

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

struct AscendPreEmitPrepareForEmitPass
    : public PassWrapper<AscendPreEmitPrepareForEmitPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AscendPreEmitPrepareForEmitPass)

  StringRef getArgument() const final {
    return "ascend-preemit-prepare-for-emit-internal";
  }

  StringRef getDescription() const final {
    return "Run internal Ascend prepare-for-emit lowering";
  }

  void runOnOperation() override {
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      if (failed(prepareFunc(func))) {
        signalPassFailure();
        return;
      }
    }
  }
};

std::unique_ptr<Pass> createAscendPreEmitPrepareForEmitPass() {
  return std::make_unique<AscendPreEmitPrepareForEmitPass>();
}

} // namespace mlir::ascend
