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
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

LogicalResult SubOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

LogicalResult MulOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

LogicalResult DivOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}