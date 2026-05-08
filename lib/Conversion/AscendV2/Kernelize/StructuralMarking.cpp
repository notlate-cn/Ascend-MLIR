//===- StructuralMarking.cpp - Ascend V2 structural marking --------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Kernelize/StructuralMarking.h"

#include "Conversion/AscendV2/Kernelize/KernelizeTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::afir::ascend::v2::kernelize {
namespace {

bool hasAtLeastTwoEntries(
    Operation *op,
    const DenseMap<Operation *, SmallVector<Operation *>> &adjacency) {
  auto it = adjacency.find(op);
  return it != adjacency.end() && it->second.size() >= 2;
}

bool hasBranchRootMark(Operation *op) {
  return op->hasAttr(kBranchRootAttr);
}

bool hasMergeRootMark(Operation *op) {
  return op->hasAttr(kMergeRootAttr);
}

} // namespace

LogicalResult StructuralMarker::mark(
    ModuleOp module, const DependencyAnalysisResult &deps) const {
  MLIRContext *context = module.getContext();

  int64_t nextBranchGroup = 0;
  for (Operation *op : deps.index.orderedOps) {
    if (!hasAtLeastTwoEntries(op, deps.index.consumers))
      continue;

    op->setAttr(kBranchRootAttr, BoolAttr::get(context, true));
    op->setAttr(kBranchGroupAttr,
                IntegerAttr::get(IntegerType::get(context, 64),
                                 nextBranchGroup++));
  }

  int64_t nextMergeGroup = 0;
  for (Operation *op : deps.index.orderedOps) {
    if (!hasAtLeastTwoEntries(op, deps.index.producers))
      continue;

    op->setAttr(kMergeRootAttr, BoolAttr::get(context, true));
    op->setAttr(kMergeGroupAttr,
                IntegerAttr::get(IntegerType::get(context, 64),
                                 nextMergeGroup++));
  }

  return success();
}

void emitStructuralMarkingReport(raw_ostream &os,
                                 const DependencyAnalysisResult &deps) {
  os << "StructuralMarking\n";
  for (Operation *op : deps.index.orderedOps) {
    bool hasBranch = hasBranchRootMark(op);
    bool hasMerge = hasMergeRootMark(op);
    if (!hasBranch && !hasMerge)
      continue;

    OperationId opId = deps.index.opIds.lookup(op);
    os << "  op_id = " << opId.value;
    if (hasBranch) {
      auto branchGroup = op->getAttrOfType<IntegerAttr>(kBranchGroupAttr);
      os << " branch_root = true branch_group = "
         << branchGroup.getInt();
    }
    if (hasMerge) {
      auto mergeGroup = op->getAttrOfType<IntegerAttr>(kMergeGroupAttr);
      os << " merge_root = true merge_group = " << mergeGroup.getInt();
    }
    os << "\n";
  }
}

} // namespace mlir::afir::ascend::v2::kernelize
