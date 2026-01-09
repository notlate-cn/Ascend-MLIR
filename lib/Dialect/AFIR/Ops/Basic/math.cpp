//===- math.cpp - AFIR math operation implementations -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "Dialect/AFIR/ShapeHelper.h"
#include "Interface/ShapeHelperOpInterface.h"
#include "Interface/ShapeInferenceOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::afir;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

namespace {

static bool hasShapeAndRank(Value val) {
  Type valType = val.getType();
  ShapedType shapedType = mlir::dyn_cast<ShapedType>(valType);
  return shapedType && shapedType.hasRank();
}

static bool hasShapeAndRank(Operation *op) {
  int num = op->getNumOperands();
  for (int i = 0; i < num; ++i)
    if (!hasShapeAndRank(op->getOperand(i))) return false;
  return true;
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

template <typename OpType>
static LogicalResult inferShapesForBinaryElementwiseOp(OpType &op, Type elementType = nullptr) {
  if (!hasShapeAndRank(op.getOperation())) return success();

  if (!elementType) elementType = mlir::cast<ShapedType>(op.getLhs().getType()).getElementType();
  AFIRBroadcastOpShapeHelper shapeHelper(op.getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

}  // namespace

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}

LogicalResult AddOp::inferShapes(std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapesForBinaryElementwiseOp<AddOp>(*this);
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

LogicalResult SubOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}

LogicalResult SubOp::inferShapes(std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapesForBinaryElementwiseOp<SubOp>(*this);
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

LogicalResult MulOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}

LogicalResult MulOp::inferShapes(std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapesForBinaryElementwiseOp<MulOp>(*this);
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

LogicalResult DivOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}

LogicalResult DivOp::inferShapes(std::function<void(mlir::Region &)> doShapeInference) {
  return inferShapesForBinaryElementwiseOp<DivOp>(*this);
}