//===- CandidateClosure.cpp - Ascend candidate closure model ----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/Candidate/CandidateClosure.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

using namespace mlir;

namespace mlir::afir::ascend::kernelize {
namespace {

void appendUniqueValue(SmallVectorImpl<Value> &values,
                       DenseSet<void *> &seen, Value value) {
  if (seen.insert(value.getAsOpaquePointer()).second)
    values.push_back(value);
}

void appendUniqueOp(SmallVectorImpl<Operation *> &ops,
                    DenseSet<Operation *> &seen, Operation *op) {
  if (seen.insert(op).second)
    ops.push_back(op);
}

} // namespace

CandidateClosure computeCandidateClosure(ArrayRef<Operation *> internalOps,
                                          const ProducerConsumerIndex &index) {
  CandidateClosure closure;

  DenseSet<Operation *> seenInternalOps;
  bool hasUnknownInternalOp = false;
  for (Operation *op : internalOps) {
    if (!index.opIds.contains(op)) {
      hasUnknownInternalOp = true;
      continue;
    }
    appendUniqueOp(closure.internalOps, seenInternalOps, op);
  }

  llvm::sort(closure.internalOps, [&](Operation *lhs, Operation *rhs) {
    return index.opIds.lookup(lhs).value < index.opIds.lookup(rhs).value;
  });

  DenseSet<Operation *> internalSet(closure.internalOps.begin(),
                                    closure.internalOps.end());

  DenseSet<void *> seenExternalInputs;
  DenseSet<void *> seenEscapingValues;
  DenseSet<void *> seenExternalOutputs;
  for (Operation *op : closure.internalOps) {
    for (Value operand : op->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      if (!producer || !internalSet.contains(producer))
        appendUniqueValue(closure.externalInputs, seenExternalInputs, operand);
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
        appendUniqueValue(closure.escapingValues, seenEscapingValues, result);
      else if (hasExternalUse)
        appendUniqueValue(closure.externalOutputs, seenExternalOutputs, result);
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
