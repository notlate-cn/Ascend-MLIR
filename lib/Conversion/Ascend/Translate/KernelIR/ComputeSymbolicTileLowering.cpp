#include "Compute/ComputeLoweringInternal.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <optional>

using namespace mlir;

namespace mlir {
namespace ascend {
namespace {

static constexpr llvm::StringLiteral kFuncArgAttrsAttr = "arg_attrs";

bool isSupportedRank2Reduction(linalg::GenericOp op) {
  if (op.getNumDpsInits() != 1)
    return false;
  auto iterTypes = op.getIteratorTypesArray();
  return iterTypes.size() == 2 &&
         iterTypes[0] == utils::IteratorType::parallel &&
         iterTypes[1] == utils::IteratorType::reduction;
}

bool isSupportedRank2AllParallel(linalg::GenericOp op) {
  if (op.getNumDpsInits() != 1)
    return false;
  auto iterTypes = op.getIteratorTypesArray();
  return iterTypes.size() == 2 &&
         iterTypes[0] == utils::IteratorType::parallel &&
         iterTypes[1] == utils::IteratorType::parallel;
}

bool hasSymbolicTileBinding(Operation *op) {
  auto binding =
      op->getAttrOfType<StringAttr>(ascend::kScheduleTileBindingAttr);
  return binding && binding.getValue() == ascend::kScheduleTileBindingSymbolic;
}

std::optional<std::string> getRuntimeTileParamName(Operation *op,
                                                   int64_t logicalAxis) {
  if (!hasSymbolicTileBinding(op))
    return std::nullopt;

  auto tileParams = op->getAttrOfType<ArrayAttr>(
      ascend::kScheduleTileParamsAttr);
  if (!tileParams)
    return std::nullopt;

  for (Attribute rawEntry : tileParams) {
    auto entry = dyn_cast<DictionaryAttr>(rawEntry);
    if (!entry)
      continue;
    auto axis = dyn_cast_or_null<IntegerAttr>(entry.get("axis"));
    if (!axis || axis.getInt() != logicalAxis)
      continue;
    auto binding = dyn_cast_or_null<StringAttr>(entry.get("binding"));
    if (!binding || binding.getValue() != "runtime")
      continue;
    auto name = dyn_cast_or_null<StringAttr>(entry.get("name"));
    if (!name)
      continue;
    return name.getValue().str();
  }

  return std::nullopt;
}

std::optional<int64_t> getTileParamI64(Operation *op, int64_t logicalAxis,
                                       StringRef fieldName) {
  if (!hasSymbolicTileBinding(op))
    return std::nullopt;

  auto tileParams = op->getAttrOfType<ArrayAttr>(
      ascend::kScheduleTileParamsAttr);
  if (!tileParams)
    return std::nullopt;

  for (Attribute rawEntry : tileParams) {
    auto entry = dyn_cast<DictionaryAttr>(rawEntry);
    if (!entry)
      continue;
    auto axis = dyn_cast_or_null<IntegerAttr>(entry.get("axis"));
    if (!axis || axis.getInt() != logicalAxis)
      continue;
    auto value = dyn_cast_or_null<IntegerAttr>(entry.get(fieldName));
    if (!value || !value.getType().isInteger(64))
      return std::nullopt;
    return value.getInt();
  }

  return std::nullopt;
}

std::optional<int64_t> getTileParamDefault(Operation *op,
                                           int64_t logicalAxis) {
  return getTileParamI64(op, logicalAxis, "default");
}

StringAttr getArgTileName(func::FuncOp funcOp, unsigned argNumber) {
  auto argAttrs = funcOp->getAttrOfType<ArrayAttr>(kFuncArgAttrsAttr);
  if (!argAttrs || argNumber >= argAttrs.size())
    return {};
  auto dict = dyn_cast<DictionaryAttr>(argAttrs[argNumber]);
  if (!dict)
    return {};
  return dyn_cast_or_null<StringAttr>(
      dict.get(ascend::kScheduleTileArgAttr));
}

void refreshFunctionType(func::FuncOp funcOp) {
  Block &entry = funcOp.getBody().front();
  SmallVector<Type> inputs;
  inputs.reserve(entry.getNumArguments());
  for (BlockArgument arg : entry.getArguments())
    inputs.push_back(arg.getType());
  funcOp.setFunctionType(FunctionType::get(
      funcOp.getContext(), inputs, funcOp.getFunctionType().getResults()));
}

void setArgTileName(func::FuncOp funcOp, unsigned argNumber, StringRef name) {
  MLIRContext *ctx = funcOp.getContext();
  auto existingArgAttrs =
      funcOp->getAttrOfType<ArrayAttr>(kFuncArgAttrsAttr);
  SmallVector<Attribute> newArgAttrs;
  newArgAttrs.reserve(funcOp.getNumArguments());
  for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
    SmallVector<NamedAttribute> attrs;
    if (existingArgAttrs && i < existingArgAttrs.size()) {
      if (auto dict = dyn_cast<DictionaryAttr>(existingArgAttrs[i])) {
        for (NamedAttribute attr : dict)
          attrs.push_back(attr);
      }
    }
    if (i == argNumber) {
      llvm::erase_if(attrs, [](NamedAttribute attr) {
        return attr.getName().getValue() == ascend::kScheduleTileArgAttr;
      });
      attrs.push_back(NamedAttribute(
          StringAttr::get(ctx, ascend::kScheduleTileArgAttr),
          StringAttr::get(ctx, name)));
    }
    newArgAttrs.push_back(DictionaryAttr::get(ctx, attrs));
  }
  funcOp->setAttr(kFuncArgAttrsAttr, ArrayAttr::get(ctx, newArgAttrs));
}

BlockArgument getOrCreateRuntimeTileArg(func::FuncOp funcOp, Location loc,
                                        StringRef name) {
  Block &entry = funcOp.getBody().front();
  for (BlockArgument arg : entry.getArguments()) {
    if (!arg.getType().isInteger(64))
      continue;
    StringAttr argName = getArgTileName(funcOp, arg.getArgNumber());
    if (argName && argName.getValue() == name)
      return arg;
  }

  auto arg = entry.addArgument(IntegerType::get(funcOp.getContext(), 64), loc);
  refreshFunctionType(funcOp);
  setArgTileName(funcOp, arg.getArgNumber(), name);
  return arg;
}

Value buildTileStep(OpBuilder &builder, Location loc, func::FuncOp funcOp,
                    Operation *op, int64_t logicalAxis,
                    int64_t fallbackTile) {
  if (std::optional<std::string> name =
          getRuntimeTileParamName(op, logicalAxis)) {
    BlockArgument tileArg = getOrCreateRuntimeTileArg(funcOp, loc, *name);
    return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                              tileArg);
  }
  return builder.create<arith::ConstantIndexOp>(loc, fallbackTile);
}

} // namespace

namespace {

FailureOr<unsigned> getSingleDimProjection(AffineMap map) {
  if (map.getNumDims() != 2 || map.getNumResults() != 1)
    return failure();
  auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(0));
  if (!dimExpr)
    return failure();
  unsigned position = dimExpr.getPosition();
  if (position >= 2)
    return failure();
  return position;
}

bool isRank2IdentityMap(AffineMap map) {
  return map.getNumDims() == 2 && map.getNumResults() == 2 &&
         map.isIdentity();
}

bool isRank2SwapPermutation(ArrayRef<int64_t> permutation) {
  return permutation.size() == 2 && permutation[0] == 1 &&
         permutation[1] == 0;
}

bool isRank2BroadcastTransposeMap(AffineMap map) {
  if (map.getNumDims() != 2 || map.getNumResults() != 2)
    return false;

  auto first = dyn_cast<AffineDimExpr>(map.getResult(0));
  auto second = dyn_cast<AffineConstantExpr>(map.getResult(1));
  return first && first.getPosition() == 1 && second &&
         second.getValue() == 0;
}

bool isSupportedSymbolicTileMap(Value operand, AffineMap map) {
  auto memrefType = dyn_cast<MemRefType>(operand.getType());
  if (!memrefType)
    return false;
  if (memrefType.getRank() == 1)
    return succeeded(getSingleDimProjection(map));
  if (memrefType.getRank() == 2)
    return isRank2IdentityMap(map);
  return false;
}

bool isSupportedSymbolicAllParallelTileMap(Value operand, AffineMap map) {
  auto memrefType = dyn_cast<MemRefType>(operand.getType());
  if (!memrefType)
    return false;
  if (memrefType.getRank() == 1)
    return succeeded(getSingleDimProjection(map));
  if (memrefType.getRank() == 2)
    return isRank2IdentityMap(map) || isRank2BroadcastTransposeMap(map);
  return false;
}

bool hasSupportedSelectedAllParallelTileMaps(linalg::GenericOp op,
                                             ArrayRef<AffineMap> maps,
                                             Value writebackTarget) {
  if (maps.size() !=
      static_cast<size_t>(op.getNumDpsInputs() + op.getNumDpsInits()))
    return false;

  for (unsigned i = 0, e = op.getNumDpsInputs(); i < e; ++i)
    if (!isSupportedSymbolicAllParallelTileMap(
            op.getDpsInputOperand(i)->get(), maps[i]))
      return false;

  return isSupportedSymbolicAllParallelTileMap(writebackTarget, maps.back());
}

FailureOr<int64_t> getStaticReductionExtent(linalg::GenericOp op,
                                            ArrayRef<AffineMap> maps) {
  for (unsigned i = 0, e = op.getNumDpsInputs(); i < e; ++i) {
    Value input = op.getDpsInputOperand(i)->get();
    auto inputType = dyn_cast<MemRefType>(input.getType());
    if (!inputType || inputType.getRank() != 2 ||
        !isRank2IdentityMap(maps[i]))
      continue;
    return inputType.getShape()[1];
  }
  return failure();
}

LogicalResult validateSymbolicReductionTile(linalg::GenericOp op,
                                            ArrayRef<AffineMap> maps,
                                            int64_t reductionTile) {
  if (ShapedType::isDynamic(reductionTile))
    return success();

  FailureOr<int64_t> reductionExtent = getStaticReductionExtent(op, maps);
  if (failed(reductionExtent) || ShapedType::isDynamic(*reductionExtent))
    return op.emitError("symbolic reduction tile requires full reduction axis");
  if (reductionTile != *reductionExtent)
    return op.emitError("symbolic reduction tile requires full reduction axis");

  return success();
}

LogicalResult validateSymbolicAllParallelTile(linalg::GenericOp op,
                                              Value outMemref,
                                              int64_t innerTile) {
  if (ShapedType::isDynamic(innerTile) || innerTile <= 0)
    return op.emitError("symbolic all-parallel tile requires a static "
                        "positive inner tile");

  auto outType = dyn_cast<MemRefType>(outMemref.getType());
  if (!outType || outType.getRank() != 2)
    return op.emitError("symbolic all-parallel tile requires a rank-2 output");

  return success();
}

LogicalResult validateSymbolicTileMaps(linalg::GenericOp op,
                                       ArrayRef<AffineMap> maps,
                                       Value writebackTarget) {
  if (maps.size() !=
      static_cast<size_t>(op.getNumDpsInputs() + op.getNumDpsInits()))
    return op.emitError("unsupported symbolic-tile indexing map");

  for (unsigned i = 0, e = op.getNumDpsInputs(); i < e; ++i)
    if (!isSupportedSymbolicTileMap(op.getDpsInputOperand(i)->get(), maps[i]))
      return op.emitError("unsupported symbolic-tile indexing map");

  if (!isSupportedSymbolicTileMap(writebackTarget, maps.back()))
    return op.emitError("unsupported symbolic-tile indexing map");

  return success();
}

Value createRank2TileAlloc(OpBuilder &builder, Location loc,
                           MemRefType sourceType, Value tileRows,
                           Value tileCols) {
  SmallVector<int64_t> tileShape{ShapedType::kDynamic,
                                 ShapedType::kDynamic};
  SmallVector<Value> dynamicSizes{tileRows, tileCols};

  auto tileType = MemRefType::get(tileShape, sourceType.getElementType(),
                                  MemRefLayoutAttrInterface{},
                                  sourceType.getMemorySpace());
  return builder.create<memref::AllocOp>(loc, tileType, dynamicSizes);
}

FailureOr<Value> buildTiledOperandSubview(OpBuilder &builder, Location loc,
                                          Value operand, AffineMap map,
                                          Value rowOffset, Value tileRows,
                                          Value reductionExtent) {
  auto memrefType = dyn_cast<MemRefType>(operand.getType());
  if (!memrefType)
    return failure();

  auto one = builder.getIndexAttr(1);
  if (memrefType.getRank() == 1) {
    FailureOr<unsigned> projection = getSingleDimProjection(map);
    if (failed(projection))
      return failure();

    SmallVector<OpFoldResult> offsets{
        *projection == 0 ? OpFoldResult(rowOffset)
                         : OpFoldResult(builder.getIndexAttr(0))};
    SmallVector<OpFoldResult> sizes{
        *projection == 0 ? OpFoldResult(tileRows)
                         : OpFoldResult(reductionExtent)};
    SmallVector<OpFoldResult> strides{one};
    return builder
        .create<memref::SubViewOp>(loc, operand, offsets, sizes, strides)
        .getResult();
  }

  if (memrefType.getRank() == 2 && isRank2IdentityMap(map)) {
    SmallVector<OpFoldResult> offsets{rowOffset, builder.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{tileRows, reductionExtent};
    SmallVector<OpFoldResult> strides{one, one};
    return builder
        .create<memref::SubViewOp>(loc, operand, offsets, sizes, strides)
        .getResult();
  }

  return failure();
}

FailureOr<Value> buildTiledAllParallelOperandSubview(
    OpBuilder &builder, Location loc, Value operand, AffineMap map,
    Value rowOffset, Value colOffset, Value tileRows, Value tileCols) {
  auto memrefType = dyn_cast<MemRefType>(operand.getType());
  if (!memrefType)
    return failure();

  auto one = builder.getIndexAttr(1);
  if (memrefType.getRank() == 1) {
    FailureOr<unsigned> projection = getSingleDimProjection(map);
    if (failed(projection))
      return failure();

    SmallVector<OpFoldResult> offsets{*projection == 0
                                          ? OpFoldResult(rowOffset)
                                          : OpFoldResult(colOffset)};
    SmallVector<OpFoldResult> sizes{*projection == 0
                                        ? OpFoldResult(tileRows)
                                        : OpFoldResult(tileCols)};
    SmallVector<OpFoldResult> strides{one};
    return builder
        .create<memref::SubViewOp>(loc, operand, offsets, sizes, strides)
        .getResult();
  }

  if (memrefType.getRank() == 2 && isRank2IdentityMap(map)) {
    SmallVector<OpFoldResult> offsets{rowOffset, colOffset};
    SmallVector<OpFoldResult> sizes{tileRows, tileCols};
    SmallVector<OpFoldResult> strides{one, one};
    return builder
        .create<memref::SubViewOp>(loc, operand, offsets, sizes, strides)
        .getResult();
  }

  if (memrefType.getRank() == 2 && isRank2BroadcastTransposeMap(map)) {
    SmallVector<OpFoldResult> offsets{colOffset, builder.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{tileCols, builder.getIndexAttr(1)};
    SmallVector<OpFoldResult> strides{one, one};
    return builder
        .create<memref::SubViewOp>(loc, operand, offsets, sizes, strides)
        .getResult();
  }

  return failure();
}

memref::CopyOp findSingleWritebackCopy(Value source) {
  memref::CopyOp result;
  for (Operation *user : llvm::make_early_inc_range(source.getUsers())) {
    auto copyOp = dyn_cast<memref::CopyOp>(user);
    if (!copyOp || copyOp.getSource() != source ||
        getMemorySpace(copyOp.getTarget().getType()) != 0)
      continue;
    if (result)
      return {};
    result = copyOp;
  }
  return result;
}

Value rootMemref(Value value) {
  while (true) {
    if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
      value = subview.getSource();
      continue;
    }
    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    return value;
  }
}

SmallVector<Value, 4> inputMemrefRoots(linalg::GenericOp op) {
  SmallVector<Value, 4> roots;
  auto appendRoot = [&](Value value) {
    if (!isa<MemRefType>(value.getType()))
      return;
    Value root = rootMemref(value);
    if (!llvm::is_contained(roots, root))
      roots.push_back(root);
  };

  for (OpOperand *operand : op.getDpsInputOperands())
    appendRoot(operand->get());
  return roots;
}

bool rootIntersects(Value value, ArrayRef<Value> roots) {
  return isa<MemRefType>(value.getType()) &&
         llvm::is_contained(roots, rootMemref(value));
}

bool rootEquals(Value value, Value root) {
  return isa<MemRefType>(value.getType()) && rootMemref(value) == root;
}

bool isBenignShapeOrViewOp(Operation *op) {
  return isa<arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
             arith::MinSIOp, arith::MaxSIOp, arith::IndexCastOp,
             affine::AffineApplyOp, memref::AllocOp, memref::DimOp,
             memref::SubViewOp, memref::CastOp>(op);
}

bool mayWriteAnyRoot(Operation *op, ArrayRef<Value> roots) {
  if (auto copyOp = dyn_cast<memref::CopyOp>(op))
    return rootIntersects(copyOp.getTarget(), roots);
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return rootIntersects(storeOp.getMemref(), roots);
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
    return llvm::any_of(linalgOp.getDpsInits(), [&](Value init) {
      return rootIntersects(init, roots);
    });
  return false;
}

bool mayReadRoot(Operation *op, Value root) {
  if (auto copyOp = dyn_cast<memref::CopyOp>(op))
    return rootEquals(copyOp.getSource(), root);
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return rootEquals(loadOp.getMemref(), root);
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
    return llvm::any_of(linalgOp.getDpsInputs(), [&](Value input) {
      return rootEquals(input, root);
    });
  return false;
}

bool touchesAnyRoot(Operation *op, ArrayRef<Value> roots) {
  return llvm::any_of(op->getOperands(), [&](Value operand) {
    return rootIntersects(operand, roots);
  });
}

bool canMoveSymbolicTileLoopBeforeWriteback(linalg::GenericOp op,
                                            memref::CopyOp writeback,
                                            Value outMemref) {
  SmallVector<Value, 4> inputRoots = inputMemrefRoots(op);
  Value outputRoot = rootMemref(outMemref);
  SmallVector<Value, 1> outputRoots{outputRoot};
  for (Operation *it = op->getNextNode(); it && it != writeback.getOperation();
       it = it->getNextNode()) {
    if (isBenignShapeOrViewOp(it))
      continue;

    if (mayWriteAnyRoot(it, inputRoots) || mayWriteAnyRoot(it, outputRoots) ||
        mayReadRoot(it, outputRoot))
      return false;

    if (!isa<linalg::LinalgOp, memref::CopyOp, memref::LoadOp,
             memref::StoreOp>(it) &&
        (touchesAnyRoot(it, inputRoots) || touchesAnyRoot(it, outputRoots)))
      return false;
  }
  return true;
}

Operation *symbolicTileInsertionPoint(linalg::GenericOp op,
                                      memref::CopyOp writeback,
                                      Value outMemref) {
  if (!canMoveSymbolicTileLoopBeforeWriteback(op, writeback, outMemref))
    return nullptr;

  Operation *targetDef = writeback.getTarget().getDefiningOp();
  if (targetDef && targetDef->getBlock() == op->getBlock() &&
      op->isBeforeInBlock(targetDef))
    return writeback.getOperation();
  return op.getOperation();
}

bool isZeroScalarConstant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;

  Attribute attr = constant.getValue();
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValue().isZero();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().isZero();
  return false;
}

linalg::FillOp findRedundantZeroFill(Value target, Operation *anchor) {
  if (!target || !anchor || anchor->getBlock() == nullptr)
    return {};

  linalg::FillOp latestFill;
  for (Operation *user : target.getUsers()) {
    auto fillOp = dyn_cast<linalg::FillOp>(user);
    if (!fillOp || fillOp.getOutputs()[0] != target ||
        fillOp->getBlock() != anchor->getBlock() ||
        !fillOp->isBeforeInBlock(anchor) ||
        !isZeroScalarConstant(fillOp.getInputs()[0]))
      continue;
    if (!latestFill || latestFill->isBeforeInBlock(fillOp))
      latestFill = fillOp;
  }
  if (!latestFill)
    return {};

  Value targetRoot = rootMemref(target);
  SmallVector<Value, 1> targetRoots{targetRoot};
  for (Operation *it = latestFill->getNextNode(); it && it != anchor;
       it = it->getNextNode()) {
    if (isBenignShapeOrViewOp(it))
      continue;
    if (mayReadRoot(it, targetRoot) || mayWriteAnyRoot(it, targetRoots))
      return {};
    if (!isa<linalg::LinalgOp, memref::CopyOp, memref::LoadOp,
             memref::StoreOp>(it) &&
        touchesAnyRoot(it, targetRoots))
      return {};
  }

  return latestFill;
}

} // namespace


LogicalResult materializeSymbolicReductionTiles(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  SmallVector<linalg::GenericOp> candidates;
  funcOp.walk([&](linalg::GenericOp op) {
    if (op->getParentOfType<scf::ForOp>())
      return;
    if (hasSymbolicTileBinding(op) &&
        op->getAttrOfType<ArrayAttr>(ascend::kScheduleTileParamsAttr))
      candidates.push_back(op);
  });

  for (linalg::GenericOp genOp : candidates) {
    if (!isSupportedRank2Reduction(genOp))
      continue;

    std::optional<int64_t> tileRowsDefault =
        getTileParamDefault(genOp.getOperation(), /*logicalAxis=*/0);
    std::optional<int64_t> reductionTileDefault =
        getTileParamDefault(genOp.getOperation(), /*logicalAxis=*/1);
    if (!tileRowsDefault || !reductionTileDefault)
      return genOp.emitError(
          "symbolic rank-2 reduction tile requires two tile params");
    int64_t tileRows = *tileRowsDefault;
    if (ShapedType::isDynamic(tileRows) || tileRows <= 0)
      return genOp.emitError(
          "symbolic reduction tile requires a positive parallel default");

    Value outMemref = genOp.getDpsInitOperand(0)->get();
    auto outType = dyn_cast<MemRefType>(outMemref.getType());
    if (!outType || outType.getRank() != 1)
      continue;

    memref::CopyOp writeback = findSingleWritebackCopy(outMemref);
    if (!writeback)
      continue;

    auto maps = genOp.getIndexingMapsArray();
    if (failed(validateSymbolicTileMaps(genOp, maps, writeback.getTarget())))
      return failure();
    if (failed(validateSymbolicReductionTile(
            genOp, maps, *reductionTileDefault)))
      return failure();

    Location loc = genOp.getLoc();
    Operation *insertionPoint =
        symbolicTileInsertionPoint(genOp, writeback, outMemref);
    if (!insertionPoint)
      continue;

    Value dstMemref = writeback.getTarget();
    linalg::FillOp redundantZeroFill =
        findRedundantZeroFill(dstMemref, writeback.getOperation());
    builder.setInsertionPoint(insertionPoint);
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value step = buildTileStep(builder, loc, funcOp, genOp.getOperation(),
                               /*logicalAxis=*/0, tileRows);
    Value rows = getDimValue(builder, loc, dstMemref, 0);

    auto forOp = builder.create<scf::ForOp>(loc, zero, rows, step);
    forOp->setAttr("ascendc.parallel", builder.getBoolAttr(true));

    OpBuilder bodyBuilder(funcOp.getContext());
    bodyBuilder.setInsertionPointToStart(forOp.getBody());
    Value remaining =
        bodyBuilder.create<arith::SubIOp>(loc, rows, forOp.getInductionVar());
    Value tileRowsValue =
        bodyBuilder.create<arith::MinSIOp>(loc, step, remaining);

    Value reductionExtent;
    for (unsigned i = 0, e = genOp.getNumDpsInputs(); i < e; ++i) {
      Value input = genOp.getDpsInputOperand(i)->get();
      auto inputType = dyn_cast<MemRefType>(input.getType());
      if (!inputType || inputType.getRank() != 2 ||
          !isRank2IdentityMap(maps[i]))
        continue;
      reductionExtent = getDimValue(bodyBuilder, loc, input, 1);
      break;
    }
    if (!reductionExtent)
      return genOp.emitError("symbolic reduction tile requires a rank-2 input");

    IRMapping mapper;
    for (unsigned i = 0, e = genOp.getNumDpsInputs(); i < e; ++i) {
      Value input = genOp.getDpsInputOperand(i)->get();
      FailureOr<Value> tiledInput = buildTiledOperandSubview(
          bodyBuilder, loc, input, maps[i], forOp.getInductionVar(),
          tileRowsValue, reductionExtent);
      if (failed(tiledInput))
        return genOp.emitError("unsupported symbolic-tile indexing map");
      mapper.map(input, *tiledInput);
    }

    auto tileOutType =
        MemRefType::get({ShapedType::kDynamic}, outType.getElementType(),
                        MemRefLayoutAttrInterface{}, outType.getMemorySpace());
    Value tiledOut = bodyBuilder.create<memref::AllocOp>(
        loc, tileOutType, ValueRange{tileRowsValue});
    mapper.map(outMemref, tiledOut);

    bodyBuilder.clone(*genOp, mapper);

    FailureOr<Value> tiledDst = buildTiledOperandSubview(
        bodyBuilder, loc, dstMemref, maps.back(), forOp.getInductionVar(),
        tileRowsValue, reductionExtent);
    if (failed(tiledDst))
      return writeback.emitError("unsupported symbolic-tile indexing map");
    bodyBuilder.create<memref::CopyOp>(loc, tiledOut, *tiledDst);

    genOp.erase();
    writeback.erase();
    if (redundantZeroFill)
      redundantZeroFill.erase();
    if (auto allocOp = outMemref.getDefiningOp<memref::AllocOp>())
      if (allocOp->use_empty())
        allocOp.erase();
  }

  return success();
}

LogicalResult materializeSymbolicTransposeTiles(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  SmallVector<linalg::TransposeOp> candidates;
  funcOp.walk([&](linalg::TransposeOp op) {
    if (op->getParentOfType<scf::ForOp>())
      return;
    candidates.push_back(op);
  });

  for (linalg::TransposeOp transposeOp : candidates) {
    FailureOr<TransposeLoweringSpec> spec =
        buildTransposeLoweringSpec(transposeOp);
    if (failed(spec) || !isRank2SwapPermutation(spec->permutation))
      continue;

    Value inMemref = transposeOp.getDpsInputOperand(0)->get();
    Value outMemref = transposeOp.getDpsInitOperand(0)->get();
    auto inType = dyn_cast<MemRefType>(inMemref.getType());
    auto outType = dyn_cast<MemRefType>(outMemref.getType());
    if (!inType || !outType || inType.getRank() != 2 ||
        outType.getRank() != 2)
      continue;
    if (getMemorySpace(inType) != 0 || getMemorySpace(outType) != 0)
      continue;
    if (inType.getElementType() != outType.getElementType())
      continue;

    int64_t tileRows = ShapedType::kDynamic;
    int64_t tileCols = ShapedType::kDynamic;
    ArrayRef<int64_t> outShape = outType.getShape();
    auto deriveStaticFullInnerTile = [&]() -> bool {
      if (ShapedType::isDynamic(outShape[0]) ||
          ShapedType::isDynamic(outShape[1]))
        return false;
      unsigned elemBits = outType.getElementTypeBitWidth();
      if (elemBits == 0 || elemBits % 8 != 0)
        return false;
      int64_t elemBytes = static_cast<int64_t>(elemBits / 8);
      constexpr int64_t kTransposeBufferBudgetBytes = 48 * 1024;
      int64_t fullBytes = outShape[0] * outShape[1] * elemBytes;
      if (fullBytes <= kTransposeBufferBudgetBytes)
        return false;
      tileCols = outShape[1];
      tileRows =
          std::max<int64_t>(1, kTransposeBufferBudgetBytes /
                                   std::max<int64_t>(1, tileCols * elemBytes));
      tileRows = std::min(tileRows, outShape[0]);
      return true;
    };

    if (!deriveStaticFullInnerTile())
      continue;

    if (ShapedType::isDynamic(outType.getShape()[1]) ||
        tileCols != outType.getShape()[1])
      continue;

    Location loc = transposeOp.getLoc();
    builder.setInsertionPoint(transposeOp);
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value step = builder.create<arith::ConstantIndexOp>(loc, tileRows);
    Value rows = getDimValue(builder, loc, outMemref, 0);
    auto forOp = builder.create<scf::ForOp>(loc, zero, rows, step);
    forOp->setAttr("ascendc.parallel", builder.getBoolAttr(true));

    OpBuilder bodyBuilder(funcOp.getContext());
    bodyBuilder.setInsertionPointToStart(forOp.getBody());
    Value remaining =
        bodyBuilder.create<arith::SubIOp>(loc, rows, forOp.getInductionVar());
    Value tileRowsValue =
        bodyBuilder.create<arith::MinSIOp>(loc, step, remaining);

    OpFoldResult zeroAttr = bodyBuilder.getIndexAttr(0);
    OpFoldResult oneAttr = bodyBuilder.getIndexAttr(1);
    OpFoldResult fullInnerAttr = bodyBuilder.getIndexAttr(tileCols);
    OpFoldResult rowOffset = forOp.getInductionVar();
    OpFoldResult tileRowsSize = tileRowsValue;

    Value inputTile =
        bodyBuilder
            .create<memref::SubViewOp>(
                loc, inMemref,
                SmallVector<OpFoldResult>{zeroAttr, rowOffset},
                SmallVector<OpFoldResult>{fullInnerAttr, tileRowsSize},
                SmallVector<OpFoldResult>{oneAttr, oneAttr})
            .getResult();
    Value outputTile =
        bodyBuilder
            .create<memref::SubViewOp>(
                loc, outMemref,
                SmallVector<OpFoldResult>{rowOffset, zeroAttr},
                SmallVector<OpFoldResult>{tileRowsSize, fullInnerAttr},
                SmallVector<OpFoldResult>{oneAttr, oneAttr})
            .getResult();

    IRMapping mapper;
    mapper.map(inMemref, inputTile);
    mapper.map(outMemref, outputTile);
    bodyBuilder.clone(*transposeOp, mapper);
    transposeOp.erase();
  }

  return success();
}

LogicalResult materializeSymbolicAllParallelTiles(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  SmallVector<linalg::GenericOp> candidates;
  funcOp.walk([&](linalg::GenericOp op) {
    if (op->getParentOfType<scf::ForOp>())
      return;
    if (hasSymbolicTileBinding(op) &&
        op->getAttrOfType<ArrayAttr>(ascend::kScheduleTileParamsAttr))
      candidates.push_back(op);
  });

  for (linalg::GenericOp genOp : candidates) {
    if (!isSupportedRank2AllParallel(genOp))
      continue;

    std::optional<int64_t> tileRowsDefault =
        getTileParamDefault(genOp.getOperation(), /*logicalAxis=*/0);
    std::optional<int64_t> tileColsDefault =
        getTileParamDefault(genOp.getOperation(), /*logicalAxis=*/1);
    if (!tileRowsDefault || !tileColsDefault)
      return genOp.emitError(
          "symbolic rank-2 all-parallel tile requires two tile params");
    int64_t tileRows = *tileRowsDefault;
    if (ShapedType::isDynamic(tileRows) || tileRows <= 0)
      return genOp.emitError("symbolic all-parallel tile requires a positive "
                             "outer default");
    int64_t tileCols = *tileColsDefault;
    if (ShapedType::isDynamic(tileCols) || tileCols <= 0)
      return genOp.emitError("symbolic all-parallel tile requires a positive "
                             "inner default");

    Value outMemref = genOp.getDpsInitOperand(0)->get();
    auto outType = dyn_cast<MemRefType>(outMemref.getType());
    if (!outType || outType.getRank() != 2)
      continue;

    memref::CopyOp writeback = findSingleWritebackCopy(outMemref);
    if (!writeback)
      continue;

    auto maps = genOp.getIndexingMapsArray();
    if (!hasSupportedSelectedAllParallelTileMaps(genOp, maps,
                                                 writeback.getTarget()))
      continue;
    if (failed(validateSymbolicAllParallelTile(
            genOp, outMemref, tileCols)))
      return failure();

    Location loc = genOp.getLoc();
    Operation *insertionPoint =
        symbolicTileInsertionPoint(genOp, writeback, outMemref);
    if (!insertionPoint)
      continue;

    Value dstMemref = writeback.getTarget();
    builder.setInsertionPoint(insertionPoint);
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value step = buildTileStep(builder, loc, funcOp, genOp.getOperation(),
                               /*logicalAxis=*/0, tileRows);
    Value rows = getDimValue(builder, loc, dstMemref, 0);
    Value innerExtent = getDimValue(builder, loc, dstMemref, 1);
    bool dynamicInner = ShapedType::isDynamic(outType.getShape()[1]);
    bool hasRuntimeInnerTile =
        getRuntimeTileParamName(genOp.getOperation(),
                                /*logicalAxis=*/1)
            .has_value();
    bool hasRank2SubviewInput = false;
    for (unsigned i = 0, e = genOp.getNumDpsInputs(); i < e; ++i) {
      auto inputType =
          dyn_cast<MemRefType>(genOp.getDpsInputOperand(i)->get().getType());
      if (inputType && inputType.getRank() == 2 &&
          genOp.getDpsInputOperand(i)->get().getDefiningOp<memref::SubViewOp>()) {
        hasRank2SubviewInput = true;
        break;
      }
    }
    bool needsInnerLoop =
        hasRuntimeInnerTile ||
        (!dynamicInner && tileCols < outType.getShape()[1]) ||
        (dynamicInner && hasRank2SubviewInput);

    auto forOp = builder.create<scf::ForOp>(loc, zero, rows, step);
    forOp->setAttr("ascendc.parallel", builder.getBoolAttr(true));

    OpBuilder bodyBuilder(funcOp.getContext());
    bodyBuilder.setInsertionPointToStart(forOp.getBody());
    Value remaining =
        bodyBuilder.create<arith::SubIOp>(loc, rows, forOp.getInductionVar());
    Value tileRowsValue =
        bodyBuilder.create<arith::MinSIOp>(loc, step, remaining);

    auto emitTileBody = [&](OpBuilder &tileBuilder, Value colOffset,
                            Value tileColsValue) -> LogicalResult {
      IRMapping mapper;
      for (unsigned i = 0, e = genOp.getNumDpsInputs(); i < e; ++i) {
        Value input = genOp.getDpsInputOperand(i)->get();
        FailureOr<Value> tiledInput = buildTiledAllParallelOperandSubview(
            tileBuilder, loc, input, maps[i], forOp.getInductionVar(),
            colOffset, tileRowsValue, tileColsValue);
        if (failed(tiledInput))
          return genOp.emitError("unsupported symbolic-tile indexing map");
        mapper.map(input, *tiledInput);
      }

      Value tiledOut = createRank2TileAlloc(tileBuilder, loc, outType,
                                            tileRowsValue, tileColsValue);
      mapper.map(outMemref, tiledOut);
      tileBuilder.clone(*genOp, mapper);

      FailureOr<Value> tiledDst = buildTiledAllParallelOperandSubview(
          tileBuilder, loc, dstMemref, maps.back(), forOp.getInductionVar(),
          colOffset, tileRowsValue, tileColsValue);
      if (failed(tiledDst))
        return writeback.emitError("unsupported symbolic-tile indexing map");
      tileBuilder.create<memref::CopyOp>(loc, tiledOut, *tiledDst);
      return success();
    };

    if (needsInnerLoop) {
      Value colStep = buildTileStep(bodyBuilder, loc, funcOp,
                                    genOp.getOperation(),
                                    /*logicalAxis=*/1, tileCols);
      auto colFor = bodyBuilder.create<scf::ForOp>(loc, zero, innerExtent,
                                                   colStep);
      OpBuilder innerBuilder(funcOp.getContext());
      innerBuilder.setInsertionPointToStart(colFor.getBody());
      Value remainingCols = innerBuilder.create<arith::SubIOp>(
          loc, innerExtent, colFor.getInductionVar());
      Value tileColsValue =
          innerBuilder.create<arith::MinSIOp>(loc, colStep, remainingCols);
      if (failed(emitTileBody(innerBuilder, colFor.getInductionVar(),
                              tileColsValue)))
        return failure();
    } else if (failed(emitTileBody(bodyBuilder, zero, innerExtent))) {
      return failure();
    }

    genOp.erase();
    writeback.erase();
    if (auto allocOp = outMemref.getDefiningOp<memref::AllocOp>())
      if (allocOp->use_empty())
        allocOp.erase();
  }

  return success();
}

} // namespace ascend
} // namespace mlir
