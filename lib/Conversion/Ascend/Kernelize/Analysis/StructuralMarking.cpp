//===- StructuralMarking.cpp - Ascend structural marking --------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/Analysis/StructuralMarking.h"

#include "Conversion/Ascend/Kernelize/KernelizeTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::ascend::kernelize {
namespace {

bool hasAtLeastTwoEntries(
    Operation *op,
    const DenseMap<Operation *, SmallVector<Operation *>> &adjacency) {
  auto it = adjacency.find(op);
  return it != adjacency.end() && it->second.size() >= 2;
}

bool hasBranchRootMark(Operation *op) {
  auto attr = op->getAttrOfType<BoolAttr>(kBranchRootAttr);
  return attr && attr.getValue();
}

bool hasMergeRootMark(Operation *op) {
  auto attr = op->getAttrOfType<BoolAttr>(kMergeRootAttr);
  return attr && attr.getValue();
}

void clearStructuralMarks(Operation *op) {
  op->removeAttr(kBranchRootAttr);
  op->removeAttr(kBranchGroupAttr);
  op->removeAttr(kMergeRootAttr);
  op->removeAttr(kMergeGroupAttr);
}

bool isPropagationEligible(Operation *op,
                           const DependencyAnalysisResult &deps) {
  auto it = deps.summaries.find(op);
  if (it == deps.summaries.end())
    return false;

  switch (it->second.accessPattern) {
  case AccessPatternKind::Elementwise:
  case AccessPatternKind::Broadcast:
  case AccessPatternKind::Gather:
  case AccessPatternKind::LayoutTransform:
    return true;
  case AccessPatternKind::Contraction:
  case AccessPatternKind::Reduction:
  case AccessPatternKind::Scatter:
  case AccessPatternKind::NotApplicable:
  case AccessPatternKind::Unknown:
    return false;
  }
  return false;
}

ArrayRef<Operation *>
lookupAdjacent(Operation *op,
               const DenseMap<Operation *, SmallVector<Operation *>> &edges) {
  auto it = edges.find(op);
  if (it == edges.end())
    return {};
  return it->second;
}

void setGroupMark(Operation *op, StringRef attrName, int64_t group,
                  MLIRContext *context) {
  op->setAttr(attrName,
              IntegerAttr::get(IntegerType::get(context, 64), group));
}

void propagateBranchGroup(Operation *root, int64_t group,
                          MLIRContext *context,
                          const DependencyAnalysisResult &deps) {
  SmallVector<Operation *> worklist(lookupAdjacent(root, deps.index.consumers));
  DenseSet<Operation *> visited;

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!visited.insert(op).second)
      continue;
    if (hasMergeRootMark(op) ||
        hasAtLeastTwoEntries(op, deps.index.producers) ||
        !isPropagationEligible(op, deps))
      continue;

    setGroupMark(op, kBranchGroupAttr, group, context);
    if (hasAtLeastTwoEntries(op, deps.index.consumers))
      continue;
    llvm::append_range(worklist, lookupAdjacent(op, deps.index.consumers));
  }
}

void propagateMergeGroup(Operation *root, int64_t group,
                         MLIRContext *context,
                         const DependencyAnalysisResult &deps) {
  SmallVector<Operation *> worklist(lookupAdjacent(root, deps.index.producers));
  DenseSet<Operation *> visited;

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!visited.insert(op).second)
      continue;
    if (hasBranchRootMark(op) ||
        hasAtLeastTwoEntries(op, deps.index.consumers) ||
        !isPropagationEligible(op, deps))
      continue;

    setGroupMark(op, kMergeGroupAttr, group, context);
    if (hasAtLeastTwoEntries(op, deps.index.producers))
      continue;
    llvm::append_range(worklist, lookupAdjacent(op, deps.index.producers));
  }
}

} // namespace

LogicalResult StructuralMarker::mark(
    ModuleOp module, const DependencyAnalysisResult &deps) const {
  MLIRContext *context = module.getContext();

  for (Operation *op : deps.index.orderedOps)
    clearStructuralMarks(op);

  int64_t nextBranchGroup = 0;
  for (Operation *op : deps.index.orderedOps) {
    if (!hasAtLeastTwoEntries(op, deps.index.consumers))
      continue;

    op->setAttr(kBranchRootAttr, BoolAttr::get(context, true));
    int64_t group = nextBranchGroup++;
    setGroupMark(op, kBranchGroupAttr, group, context);
    propagateBranchGroup(op, group, context, deps);
  }

  int64_t nextMergeGroup = 0;
  for (Operation *op : deps.index.orderedOps) {
    if (!hasAtLeastTwoEntries(op, deps.index.producers))
      continue;

    op->setAttr(kMergeRootAttr, BoolAttr::get(context, true));
    int64_t group = nextMergeGroup++;
    setGroupMark(op, kMergeGroupAttr, group, context);
    propagateMergeGroup(op, group, context, deps);
  }

  return success();
}

void emitStructuralMarkingReport(raw_ostream &os,
                                 const DependencyAnalysisResult &deps) {
  os << "StructuralMarking\n";
  for (Operation *op : deps.index.orderedOps) {
    bool hasBranch = hasBranchRootMark(op);
    bool hasMerge = hasMergeRootMark(op);
    auto branchGroup = op->getAttrOfType<IntegerAttr>(kBranchGroupAttr);
    auto mergeGroup = op->getAttrOfType<IntegerAttr>(kMergeGroupAttr);
    if (!hasBranch && !branchGroup && !hasMerge && !mergeGroup)
      continue;

    OperationId opId = deps.index.opIds.lookup(op);
    os << "  op_id = " << opId.value;
    if (hasBranch) {
      os << " branch_root = true";
    }
    if (branchGroup)
      os << " branch_group = " << branchGroup.getInt();
    if (hasMerge) {
      os << " merge_root = true";
    }
    if (mergeGroup)
      os << " merge_group = " << mergeGroup.getInt();
    os << "\n";
  }
}

} // namespace mlir::ascend::kernelize
