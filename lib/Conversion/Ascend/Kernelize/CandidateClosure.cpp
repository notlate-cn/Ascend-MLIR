//===- CandidateClosure.cpp - Ascend candidate closure model ----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/CandidateClosure.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

using namespace mlir;

namespace mlir::afir::ascend::kernelize {
namespace {

void appendUniqueValue(SmallVectorImpl<Value> &values, Value value) {
  if (!llvm::is_contained(values, value))
    values.push_back(value);
}

void appendUniqueOp(SmallVectorImpl<Operation *> &ops, Operation *op) {
  if (!llvm::is_contained(ops, op))
    ops.push_back(op);
}

} // namespace

CandidateClosure computeCandidateClosure(ArrayRef<Operation *> internalOps,
                                          const ProducerConsumerIndex &index) {
  CandidateClosure closure;

  bool hasUnknownInternalOp = false;
  for (Operation *op : internalOps) {
    if (!index.opIds.contains(op)) {
      hasUnknownInternalOp = true;
      continue;
    }
    appendUniqueOp(closure.internalOps, op);
  }

  llvm::sort(closure.internalOps, [&](Operation *lhs, Operation *rhs) {
    return index.opIds.lookup(lhs).value < index.opIds.lookup(rhs).value;
  });

  DenseSet<Operation *> internalSet;
  for (Operation *op : closure.internalOps)
    internalSet.insert(op);

  for (Operation *op : closure.internalOps) {
    for (Value operand : op->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      if (!producer || !internalSet.contains(producer))
        appendUniqueValue(closure.externalInputs, operand);
    }

    for (Value result : op->getResults()) {
      bool hasInternalUse = false;
      bool hasExternalUse = false;
      for (Operation *user : result.getUsers()) {
        if (internalSet.contains(user))
          hasInternalUse = true;
        else
          hasExternalUse = true;
      }
      if (hasInternalUse && hasExternalUse)
        appendUniqueValue(closure.escapingValues, result);
      else if (hasExternalUse)
        appendUniqueValue(closure.externalOutputs, result);
    }
  }

  closure.isClosed = !hasUnknownInternalOp && closure.escapingValues.empty();
  if (hasUnknownInternalOp)
    closure.failureReason = "UnknownInternalOp";
  else if (!closure.escapingValues.empty())
    closure.failureReason = "ClosureEscape";
  return closure;
}

} // namespace mlir::afir::ascend::kernelize
