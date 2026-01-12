//===- data.cpp - AFIR data operation implementations -*- C++ -*-===//
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

static LogicalResult verifyNullaryOp(Operation *op) {
  if (op->getNumOperands() != 0) {
    return op->emitOpError("expected 0 operands");
  }
  if (op->getNumResults() != 1) {
    return op->emitOpError("expected 1 result");
  }

  auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());
  if (!resultType) {
    return op->emitOpError("expected tensor result");
  }

  return success();
}

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
    if (inputType.getRank() != resultType.getRank()) {
      return op->emitOpError("operand and result must have the same rank");
    }

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (inputShape[i] != ShapedType::kDynamic && resultShape[i] != ShapedType::kDynamic &&
          inputShape[i] != resultShape[i]) {
        return op->emitOpError("operand and result must have compatible shapes");
      }
    }
  }

  return success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// Data Operations
//===----------------------------------------------------------------------===//

LogicalResult ScalarOp::verify() {
  return verifyNullaryOp(getOperation());
}
LogicalResult IndexExprOp::verify() {
  return verifyNullaryOp(getOperation());
}
LogicalResult DataOp::verify() {
  return verifyNullaryOp(getOperation());
}
LogicalResult OutputOp::verify() {
  return verifyUnaryOp(getOperation());
}
LogicalResult LoadOp::verify() {
  return verifyUnaryOp(getOperation());
}
LogicalResult StoreOp::verify() {
  return verifyUnaryOp(getOperation());
}
