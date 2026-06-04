//===- SymbolAxisSpace.cpp - Ascend symbolic axis space ------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/Analysis/SymbolAxisSpace.h"

#include "Conversion/Ascend/Common/SymbolConstraints.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <optional>

using namespace mlir;

namespace mlir::ascend::kernelize {
namespace {

IteratorKind convertIteratorType(utils::IteratorType iteratorType) {
  if (iteratorType == utils::IteratorType::parallel)
    return IteratorKind::Parallel;
  if (iteratorType == utils::IteratorType::reduction)
    return IteratorKind::Reduction;
  return IteratorKind::Unknown;
}

SmallVector<IteratorKind, 4> getIteratorKinds(linalg::LinalgOp linalgOp) {
  SmallVector<IteratorKind, 4> kinds;
  for (utils::IteratorType iteratorType : linalgOp.getIteratorTypesArray())
    kinds.push_back(convertIteratorType(iteratorType));
  return kinds;
}

std::optional<unsigned>
lookupAxisId(const symbol::SymbolConstraintTable &symbols, Value value,
             int64_t dim) {
  symbol::DimRef ref{value, dim};
  for (auto [classIndex, klass] : llvm::enumerate(symbols.classes)) {
    if (llvm::is_contained(klass.members, ref))
      return static_cast<unsigned>(classIndex);
  }
  return std::nullopt;
}

void refineAxisKind(FunctionAxisSpace &space, unsigned axisId,
                    IteratorKind iteratorKind) {
  if (axisId >= space.axes.size() || iteratorKind == IteratorKind::Unknown)
    return;

  LogicalAxis &axis = space.axes[axisId];
  if (axis.kind == IteratorKind::Unknown) {
    axis.kind = iteratorKind;
    return;
  }
  if (axis.kind != iteratorKind)
    axis.hasMixedIteratorKinds = true;
}

LogicalResult mergeIteratorAxis(SmallVectorImpl<OpAxisRef> &opAxes,
                                unsigned iteratorIdx, unsigned axisId,
                                StringRef symbolName,
                                IteratorKind iteratorKind, Operation *op) {
  if (iteratorIdx >= opAxes.size())
    return success();

  OpAxisRef &axis = opAxes[iteratorIdx];
  if (!axis.hasAxis()) {
    axis.axisId = axisId;
    axis.symbolName = symbolName.str();
    axis.iteratorKind = iteratorKind;
    return success();
  }

  if (axis.axisId == static_cast<int64_t>(axisId))
    return success();

  return op->emitError()
         << "conflicting symbol axes for iterator " << iteratorIdx << ": "
         << axis.symbolName << " vs " << symbolName;
}

LogicalResult mapValueDimsToIterators(
    Value value, AffineMap map, const symbol::SymbolConstraintTable &symbols,
    ArrayRef<IteratorKind> iteratorTypes, FunctionAxisSpace &space,
    SmallVectorImpl<OpAxisRef> &opAxes, Operation *op) {
  auto shapedType = dyn_cast<ShapedType>(value.getType());
  if (!shapedType || !shapedType.hasRank())
    return success();
  if (map.getNumResults() != static_cast<unsigned>(shapedType.getRank()))
    return success();

  for (auto [dim, expr] : llvm::enumerate(map.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      continue;

    unsigned iteratorIdx = dimExpr.getPosition();
    if (iteratorIdx >= opAxes.size())
      continue;

    std::optional<unsigned> axisId =
        lookupAxisId(symbols, value, static_cast<int64_t>(dim));
    if (!axisId)
      continue;

    IteratorKind iteratorKind = iteratorIdx < iteratorTypes.size()
                                    ? iteratorTypes[iteratorIdx]
                                    : IteratorKind::Unknown;
    StringRef symbolName = space.axes[*axisId].symbolName;
    if (failed(mergeIteratorAxis(opAxes, iteratorIdx, *axisId, symbolName,
                                 iteratorKind, op)))
      return failure();
    refineAxisKind(space, *axisId, iteratorKind);
  }

  return success();
}

LogicalResult buildLinalgOpAxisMap(
    linalg::LinalgOp linalgOp, const symbol::SymbolConstraintTable &symbols,
    FunctionAxisSpace &space,
    DenseMap<Operation *, SmallVector<OpAxisRef, 4>> &opAxisMap) {
  Operation *op = linalgOp.getOperation();
  SmallVector<IteratorKind, 4> iteratorTypes = getIteratorKinds(linalgOp);
  if (iteratorTypes.empty())
    return success();

  SmallVector<OpAxisRef, 4> axes(iteratorTypes.size());
  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();

  unsigned operandMapCount =
      std::min<unsigned>(indexingMaps.size(), op->getNumOperands());
  for (unsigned operandIndex = 0; operandIndex < operandMapCount;
       ++operandIndex) {
    if (failed(mapValueDimsToIterators(
            op->getOperand(operandIndex), indexingMaps[operandIndex], symbols,
            iteratorTypes, space, axes, op)))
      return failure();
  }

  unsigned outputMapBase = linalgOp.getNumDpsInputs();
  unsigned resultMapCount = std::min<unsigned>(
      op->getNumResults(),
      outputMapBase < indexingMaps.size() ? indexingMaps.size() - outputMapBase
                                          : 0);
  for (unsigned resultIndex = 0; resultIndex < resultMapCount; ++resultIndex) {
    if (failed(mapValueDimsToIterators(
            op->getResult(resultIndex),
            indexingMaps[outputMapBase + resultIndex], symbols, iteratorTypes,
            space, axes, op)))
      return failure();
  }

  if (llvm::any_of(axes, [](const OpAxisRef &axis) { return axis.hasAxis(); }))
    opAxisMap.try_emplace(op, std::move(axes));
  return success();
}

LogicalResult populateSymbolAxisSpace(
    func::FuncOp func, ArrayRef<Operation *> ops,
    const symbol::SymbolConstraintTable &symbols, SymbolAxisSpace &space) {
  space.function.func = func.getOperation();
  for (auto [classIndex, klass] : llvm::enumerate(symbols.classes)) {
    LogicalAxis axis;
    axis.func = func.getOperation();
    axis.axisId = static_cast<unsigned>(classIndex);
    axis.symbolName = klass.symName.getValue().str();
    axis.memberCount = static_cast<unsigned>(klass.members.size());
    space.function.axes.push_back(std::move(axis));
  }

  for (Operation *op : ops) {
    auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op);
    if (!linalgOp)
      continue;
    if (failed(buildLinalgOpAxisMap(linalgOp, symbols, space.function,
                                    space.opAxisMap)))
      return failure();
  }

  return success();
}

} // namespace

FailureOr<SymbolAxisSpace>
buildSymbolAxisSpace(func::FuncOp func, ArrayRef<Operation *> ops) {
  SymbolAxisSpace space;
  FailureOr<symbol::SymbolConstraintTable> symbols =
      symbol::parseSymbolConstraintAttr(func);
  if (failed(symbols))
    return failure();
  if (symbols->classes.empty())
    return space;

  if (failed(populateSymbolAxisSpace(func, ops, *symbols, space)))
    return failure();
  return space;
}

FailureOr<SymbolAxisSpace> buildSymbolAxisSpace(func::FuncOp func) {
  SmallVector<Operation *, 16> ops;
  func.walk([&](linalg::LinalgOp op) { ops.push_back(op.getOperation()); });
  return buildSymbolAxisSpace(func, ops);
}

} // namespace mlir::ascend::kernelize
