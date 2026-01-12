//===- math.cpp - AFIR math operation implementations -------*- C++ -*-===//
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

static LogicalResult verifyUnaryElementwiseOp(Operation *op) {
  auto inputType = mlir::dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());

  if (!inputType || !resultType) return op->emitOpError("expected tensor operand and result");

  if (inputType.hasRank() && resultType.hasRank()) {
    if (inputType.getRank() != resultType.getRank())
      return op->emitOpError("operand and result must have the same rank");

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

static LogicalResult verifyBinaryElementwiseOp(Operation *op) {
  auto lhsType = mlir::dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto rhsType = mlir::dyn_cast<ShapedType>(op->getOperand(1).getType());
  auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());

  if (!lhsType || !rhsType || !resultType) return op->emitOpError("expected tensor operands and results");

  if (lhsType.hasRank() && rhsType.hasRank()) {
    if (lhsType.getRank() != rhsType.getRank()) return op->emitOpError("operands must have the same rank");

    ArrayRef<int64_t> lhsShape = lhsType.getShape();
    ArrayRef<int64_t> rhsShape = rhsType.getShape();

    for (size_t i = 0; i < lhsShape.size(); ++i) {
      if (lhsShape[i] != ShapedType::kDynamic && rhsShape[i] != ShapedType::kDynamic && lhsShape[i] != rhsShape[i]) {
        return op->emitOpError("operands must have compatible shapes");
      }
    }
  }

  return success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

LogicalResult AbsOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult ExpOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult LnOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult SqrtOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult RsqrtOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult ReciprocalOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult ErfOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult TanhOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult ReluOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult NegOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult LogicalNotOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult SigmoidOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult IsnanOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult IsFiniteOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult SubOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult MulOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult DivOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult MinimumOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult MaximumOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult TrueDivOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult PowOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult LeakyReluOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult BitwiseAndOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult FloorDivOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult GeluOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult SignOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult LogicalOrOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult LogicalAndOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}

//===----------------------------------------------------------------------===//
// Compare Operations
//===----------------------------------------------------------------------===//

LogicalResult GeOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult EqOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult NeOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult GtOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult LeOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult LtOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}

//===----------------------------------------------------------------------===//
// Ternary Operations
//===----------------------------------------------------------------------===//

LogicalResult ClipByValue::verify() {
  auto inputType = mlir::dyn_cast<ShapedType>(getOperand(0).getType());
  auto minType = mlir::dyn_cast<ShapedType>(getOperand(1).getType());
  auto maxType = mlir::dyn_cast<ShapedType>(getOperand(2).getType());
  auto resultType = mlir::dyn_cast<ShapedType>(getResult().getType());

  if (!inputType || !minType || !maxType || !resultType) {
    return emitOpError("expected tensor operands and result");
  }

  if (inputType.hasRank() && minType.hasRank()) {
    if (inputType.getRank() != minType.getRank()) {
      return emitOpError("input and min must have the same rank");
    }
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> minShape = minType.getShape();
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (inputShape[i] != ShapedType::kDynamic && minShape[i] != ShapedType::kDynamic &&
          inputShape[i] != minShape[i]) {
        return emitOpError("input and min must have compatible shapes");
      }
    }
  }

  if (inputType.hasRank() && maxType.hasRank()) {
    if (inputType.getRank() != maxType.getRank()) {
      return emitOpError("input and max must have the same rank");
    }
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> maxShape = maxType.getShape();
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (inputShape[i] != ShapedType::kDynamic && maxShape[i] != ShapedType::kDynamic &&
          inputShape[i] != maxShape[i]) {
        return emitOpError("input and max must have compatible shapes");
      }
    }
  }

  if (inputType.hasRank() && resultType.hasRank()) {
    if (inputType.getRank() != resultType.getRank()) {
      return emitOpError("input and result must have the same rank");
    }
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (inputShape[i] != ShapedType::kDynamic && resultShape[i] != ShapedType::kDynamic &&
          inputShape[i] != resultShape[i]) {
        return emitOpError("input and result must have compatible shapes");
      }
    }
  }

  return success();
}