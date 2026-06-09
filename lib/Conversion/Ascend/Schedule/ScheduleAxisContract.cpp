//===- ScheduleAxisContract.cpp - Ascend schedule axis contract --------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "ScheduleAxisContract.h"

#include "Conversion/Ascend/Kernelize/Analysis/SymbolAxisSpace.h"
#include "Conversion/Ascend/Kernelize/KernelizeTypes.h"
#include "ScheduleSymbolAxisSpaceCache.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <optional>
#include <utility>

using namespace mlir;

namespace mlir::ascend::schedule {
namespace {

std::string getStableAxisName(const LogicalAxisInfo &axis) {
  if (!axis.symbolName.empty())
    return axis.symbolName;
  return (llvm::Twine("axis") + llvm::Twine(axis.logicalAxisId)).str();
}

SymbolicAxisRef makeAxisRef(const LogicalAxisInfo &axis) {
  SymbolicAxisRef ref;
  ref.logicalAxisId = axis.logicalAxisId;
  ref.symbolName = getStableAxisName(axis);
  ref.kind = axis.kind;
  return ref;
}

const LogicalAxisInfo *lookupLogicalAxis(const CoalescedAxisInfo &axes,
                                         unsigned logicalAxisId) {
  for (const LogicalAxisInfo &axis : axes.logicalAxes)
    if (axis.logicalAxisId == logicalAxisId)
      return &axis;
  return nullptr;
}

bool hasAxisSymbol(ArrayRef<SymbolicAxisRef> refs, StringRef symbolName) {
  return llvm::any_of(refs, [&](const SymbolicAxisRef &ref) {
    return ref.symbolName == symbolName;
  });
}

void appendUniqueAxis(SmallVectorImpl<SymbolicAxisRef> &refs,
                      const LogicalAxisInfo &axis) {
  SymbolicAxisRef ref = makeAxisRef(axis);
  if (hasAxisSymbol(refs, ref.symbolName))
    return;
  refs.push_back(std::move(ref));
}

void removeTileableReductionSymbols(ScheduleAxisContract &contract) {
  llvm::StringSet<> reductionSymbols;
  for (const SymbolicAxisRef &axis : contract.requiredReductionAxes)
    reductionSymbols.insert(axis.symbolName);

  llvm::erase_if(contract.tileableAxes, [&](const SymbolicAxisRef &axis) {
    return reductionSymbols.contains(axis.symbolName);
  });
}

void appendRequiredReductionSymbol(ScheduleAxisContract &contract,
                                   const CoalescedAxisInfo &axes,
                                   StringRef symbolName) {
  for (const LogicalAxisInfo &axis : axes.logicalAxes) {
    if (getStableAxisName(axis) != symbolName)
      continue;
    appendUniqueAxis(contract.requiredReductionAxes, axis);
    return;
  }
}

void removeTileableAxis(ScheduleAxisContract &contract, unsigned logicalAxisId) {
  llvm::erase_if(contract.tileableAxes, [&](const SymbolicAxisRef &axis) {
    return axis.logicalAxisId == logicalAxisId;
  });
}

void appendUniqueConstraint(SmallVectorImpl<std::string> &constraints,
                            StringRef constraint) {
  if (!llvm::is_contained(constraints, constraint))
    constraints.push_back(constraint.str());
}

bool hasAttr(Operation *op, llvm::StringLiteral attrName) {
  return op && op->hasAttr(attrName);
}

bool hasBranchOrMergeMarker(Operation *op) {
  return hasAttr(op, ::mlir::ascend::kernelize::kBranchGroupAttr) ||
         hasAttr(op, ::mlir::ascend::kernelize::kMergeGroupAttr);
}

bool isStructuralPropagationOp(Operation *op) {
  return isa_and_nonnull<tensor::ConcatOp, tensor::CollapseShapeOp,
                         tensor::ExpandShapeOp, tensor::ExtractSliceOp>(op);
}

void collectStructuralDefChainOps(Operation *op,
                                  llvm::SmallPtrSetImpl<Operation *> &seen,
                                  SmallVectorImpl<Operation *> &ops) {
  if (!op)
    return;
  for (Value operand : op->getOperands()) {
    Operation *def = operand.getDefiningOp();
    if (!def || !seen.insert(def).second)
      continue;
    if (isStructuralPropagationOp(def))
      ops.push_back(def);
    collectStructuralDefChainOps(def, seen, ops);
  }
}

SmallVector<Operation *, 8>
collectPropagationOps(const KernelPatternView &pattern) {
  SmallVector<Operation *, 8> ops;
  llvm::SmallPtrSet<Operation *, 8> seen;
  for (const PatternOpView &opView : pattern.ops) {
    if (!seen.insert(opView.op).second)
      continue;
    ops.push_back(opView.op);
    collectStructuralDefChainOps(opView.op, seen, ops);
  }
  return ops;
}

bool hasReductionIterator(linalg::LinalgOp op) {
  return llvm::is_contained(op.getIteratorTypesArray(),
                            utils::IteratorType::reduction);
}

bool isRoleReduction(linalg::LinalgOp op) {
  return hasReductionIterator(op) &&
         deriveOpRole(op.getOperation()) == OpRole::Reduction;
}

bool isTraversableSoftmaxOp(Operation *op) {
  if (!op)
    return false;
  if (isStructuralPropagationOp(op))
    return true;

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;

  OpRole role = deriveOpRole(op);
  return role == OpRole::Reduction || role == OpRole::Vector;
}

bool isTensorTyped(Value value) {
  return isa<RankedTensorType, UnrankedTensorType>(value.getType());
}

void enqueueSoftmaxNeighbor(
    Operation *op, unsigned depth,
    SmallVectorImpl<std::pair<Operation *, unsigned>> &worklist,
    llvm::SmallPtrSetImpl<Operation *> &seen) {
  if (!isTraversableSoftmaxOp(op) || !seen.insert(op).second)
    return;
  worklist.push_back({op, depth});
}

SmallVector<linalg::LinalgOp, 4>
collectConnectedReductionOps(const KernelPatternView &pattern) {
  SmallVector<linalg::LinalgOp, 4> reductions;
  SmallVector<std::pair<Operation *, unsigned>, 16> worklist;
  llvm::SmallPtrSet<Operation *, 16> seen;
  llvm::SmallPtrSet<Operation *, 4> seenReductions;

  bool patternHasReduction = false;
  for (const PatternOpView &opView : pattern.ops) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(opView.op);
    patternHasReduction |= linalgOp && isRoleReduction(linalgOp);
    enqueueSoftmaxNeighbor(opView.op, /*depth=*/0, worklist, seen);
  }
  if (!patternHasReduction)
    return reductions;

  constexpr unsigned kMaxSoftmaxDataflowDepth = 8;
  while (!worklist.empty()) {
    auto [op, depth] = worklist.pop_back_val();
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      if (isRoleReduction(linalgOp) && seenReductions.insert(op).second)
        reductions.push_back(linalgOp);
    }

    if (depth >= kMaxSoftmaxDataflowDepth)
      continue;

    for (Value operand : op->getOperands()) {
      if (!isTensorTyped(operand))
        continue;
      enqueueSoftmaxNeighbor(operand.getDefiningOp(), depth + 1, worklist,
                             seen);
    }

    for (Value result : op->getResults()) {
      if (!isTensorTyped(result))
        continue;
      for (Operation *user : result.getUsers())
        enqueueSoftmaxNeighbor(user, depth + 1, worklist, seen);
    }
  }

  return reductions;
}

void sortAndUnique(SmallVectorImpl<std::string> &symbols) {
  llvm::sort(symbols);
  symbols.erase(std::unique(symbols.begin(), symbols.end()), symbols.end());
}

SmallVector<std::string, 2>
collectReductionAxisSymbols(
    linalg::LinalgOp op, const CoalescedAxisInfo &axes,
    const ::mlir::ascend::kernelize::SymbolAxisSpace *symbolAxes) {
  SmallVector<std::string, 2> symbols;
  Operation *operation = op.getOperation();
  SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();

  if (symbolAxes) {
    auto axisIt = symbolAxes->opAxisMap.find(operation);
    if (axisIt != symbolAxes->opAxisMap.end()) {
      for (auto [iteratorIdx, iteratorType] : llvm::enumerate(iteratorTypes)) {
        if (iteratorType != utils::IteratorType::reduction ||
            iteratorIdx >= axisIt->second.size())
          continue;
        const ::mlir::ascend::kernelize::OpAxisRef &axis =
            axisIt->second[iteratorIdx];
        if (axis.hasAxis() && !axis.symbolName.empty())
          symbols.push_back(axis.symbolName);
      }
      sortAndUnique(symbols);
      if (!symbols.empty())
        return symbols;
    }
  }

  for (const LogicalAxisInfo &axis : axes.logicalAxes) {
    bool hasReductionUse = false;
    for (auto [rawOp, rawAxis] : axis.rawAxes) {
      if (rawOp != operation || rawAxis >= iteratorTypes.size())
        continue;
      if (iteratorTypes[rawAxis] == utils::IteratorType::reduction) {
        hasReductionUse = true;
        break;
      }
    }
    if (!hasReductionUse)
      continue;
    symbols.push_back(getStableAxisName(axis));
  }
  sortAndUnique(symbols);
  return symbols;
}

LogicalResult appendMultiReductionConstraint(const KernelPatternView &pattern,
                                             const CoalescedAxisInfo &axes,
                                             ScheduleAxisContract &contract,
                                             const ::mlir::ascend::kernelize::
                                                 SymbolAxisSpace *symbolAxes) {
  SmallVector<SmallVector<std::string, 2>, 2> reductionSymbolSets;
  SmallVector<linalg::LinalgOp, 4> reductionOps =
      collectConnectedReductionOps(pattern);
  for (linalg::LinalgOp linalgOp : reductionOps) {
    SmallVector<std::string, 2> symbols =
        collectReductionAxisSymbols(linalgOp, axes, symbolAxes);
    if (!symbols.empty())
      reductionSymbolSets.push_back(std::move(symbols));
  }

  if (reductionSymbolSets.size() < 2)
    return success();

  ArrayRef<std::string> expected = reductionSymbolSets.front();
  for (ArrayRef<std::string> candidate : llvm::drop_begin(reductionSymbolSets)) {
    if (candidate == expected)
      continue;
    appendUniqueConstraint(contract.propagationConstraints,
                           "softmax_reduction_axis_mismatch");
    return failure();
  }

  appendUniqueConstraint(contract.propagationConstraints,
                         "multi_reduction_consistent");
  for (StringRef symbolName : expected)
    appendRequiredReductionSymbol(contract, axes, symbolName);
  return success();
}

LogicalResult appendPropagationConstraints(const KernelPatternView &pattern,
                                           const CoalescedAxisInfo &axes,
                                           ScheduleAxisContract &contract,
                                           const ::mlir::ascend::kernelize::
                                               SymbolAxisSpace *symbolAxes) {
  bool hasBranchMerge = false;
  SmallVector<Operation *, 8> propagationOps = collectPropagationOps(pattern);
  for (Operation *op : propagationOps) {
    if (auto concatOp = dyn_cast<tensor::ConcatOp>(op)) {
      appendUniqueConstraint(contract.propagationConstraints,
                             "concat_axis_barrier");
      removeTileableAxis(contract, static_cast<unsigned>(concatOp.getDim()));
      continue;
    }

    if (isa<tensor::ExtractSliceOp>(op)) {
      appendUniqueConstraint(contract.propagationConstraints,
                             "split_axis_barrier");
      continue;
    }

    if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op)) {
      appendUniqueConstraint(contract.propagationConstraints,
                             "reshape_static_bridge");
      continue;
    }

    hasBranchMerge |= hasBranchOrMergeMarker(op);
  }

  if (hasBranchMerge)
    appendUniqueConstraint(contract.propagationConstraints,
                           "branch_merge_axes_consistent");

  return appendMultiReductionConstraint(pattern, axes, contract, symbolAxes);
}

FailureOr<const ::mlir::ascend::kernelize::SymbolAxisSpace *>
getFunctionSymbolAxisSpace(const KernelPatternView &pattern,
                           ScheduleSymbolAxisSpaceCache *symbolAxisCache,
                           std::optional<::mlir::ascend::kernelize::
                                             SymbolAxisSpace> &localStorage) {
  if (pattern.ops.empty())
    return static_cast<const ::mlir::ascend::kernelize::SymbolAxisSpace *>(
        nullptr);

  auto func = pattern.ops.front().op->getParentOfType<func::FuncOp>();
  if (!func)
    return static_cast<const ::mlir::ascend::kernelize::SymbolAxisSpace *>(
        nullptr);

  if (symbolAxisCache)
    return symbolAxisCache->get(func);

  FailureOr<::mlir::ascend::kernelize::SymbolAxisSpace> symbolAxes =
      ::mlir::ascend::kernelize::buildSymbolAxisSpace(func);
  if (failed(symbolAxes))
    return failure();

  localStorage = std::move(*symbolAxes);
  return &*localStorage;
}

} // namespace

FailureOr<ScheduleAxisContract>
buildScheduleAxisContract(const KernelPatternView &pattern,
                          const CoalescedAxisInfo &axes,
                          ScheduleSymbolAxisSpaceCache *symbolAxisCache) {
  ScheduleAxisContract contract;
  llvm::SmallSet<unsigned, 4> requiredReductionIds;
  for (unsigned logicalAxisId : axes.reductionAxes) {
    const LogicalAxisInfo *axis = lookupLogicalAxis(axes, logicalAxisId);
    if (!axis)
      continue;
    appendUniqueAxis(contract.requiredReductionAxes, *axis);
    requiredReductionIds.insert(logicalAxisId);
  }

  for (unsigned logicalAxisId : axes.parallelAxes) {
    if (requiredReductionIds.contains(logicalAxisId))
      continue;
    const LogicalAxisInfo *axis = lookupLogicalAxis(axes, logicalAxisId);
    if (!axis)
      continue;
    appendUniqueAxis(contract.tileableAxes, *axis);
  }

  removeTileableReductionSymbols(contract);
  std::optional<::mlir::ascend::kernelize::SymbolAxisSpace>
      localSymbolAxisSpace;
  FailureOr<const ::mlir::ascend::kernelize::SymbolAxisSpace *> symbolAxes =
      getFunctionSymbolAxisSpace(pattern, symbolAxisCache,
                                 localSymbolAxisSpace);
  if (failed(symbolAxes))
    return failure();
  if (failed(appendPropagationConstraints(pattern, axes, contract,
                                          *symbolAxes)))
    return failure();
  removeTileableReductionSymbols(contract);
  return contract;
}

void printScheduleAxisList(ArrayRef<SymbolicAxisRef> axes,
                           llvm::raw_ostream &os) {
  os << "[";
  llvm::interleaveComma(axes, os,
                        [&](const SymbolicAxisRef &axis) {
                          os << axis.symbolName;
                        });
  os << "]";
}

} // namespace mlir::ascend::schedule
