//===- reduce.cpp - AFIR reduce operation implementations -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::afir;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

namespace {

static LogicalResult verifyUnaryOp(Operation *op) {
  if (op->getNumOperands() != 1) {
    return op->emitOpError("expected 1 operand");
  }
  if (op->getNumResults() != 1) {
    return op->emitOpError("expected 1 result");
  }

  auto inputType = mlir::dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());

  if (!inputType || !resultType) {
    return op->emitOpError("expected tensor operand and result");
  }

  if (inputType.hasRank() && resultType.hasRank()) {
    if (resultType.getRank() > inputType.getRank()) {
      return op->emitOpError("result rank must be <= input rank for reduce");
    }
  }

  return success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// Reduce Operations
//===----------------------------------------------------------------------===//

LogicalResult MaxOp::verify() {
  return verifyUnaryOp(getOperation());
}
LogicalResult MinOp::verify() {
  return verifyUnaryOp(getOperation());
}
LogicalResult SumOp::verify() {
  return verifyUnaryOp(getOperation());
}
LogicalResult MeanOp::verify() {
  return verifyUnaryOp(getOperation());
}
LogicalResult ProdOp::verify() {
  return verifyUnaryOp(getOperation());
}
LogicalResult AnyOp::verify() {
  return verifyUnaryOp(getOperation());
}
LogicalResult AllOp::verify() {
  return verifyUnaryOp(getOperation());
}
