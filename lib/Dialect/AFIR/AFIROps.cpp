//===- AFIROps.cpp - AFIR operation implementations -------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIROps.h"
#include "Dialect/AFIR/AFIRDialect.h"
#include "Interface/ShapeHelperOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::afir;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

static LogicalResult verifyBinaryElementwiseOp(Operation *op) {
  auto lhsType = mlir::dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto rhsType = mlir::dyn_cast<ShapedType>(op->getOperand(1).getType());
  auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());

  if (!lhsType || !rhsType || !resultType)
    return op->emitOpError("expected tensor operands and results");

  if (lhsType.hasRank() && rhsType.hasRank()) {
    if (lhsType.getShape() != rhsType.getShape())
      return op->emitOpError("operands must have the same shape");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify() { return verifyBinaryElementwiseOp(*this); }

LogicalResult AddOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  BinaryElementwiseShapeHelper<AddOp> shapeHelper(*this);
  if (failed(shapeHelper.computeShape()))
    return failure();

  auto outputShapes = shapeHelper.getOutputShapes();
  if (outputShapes.empty())
    return failure();

  auto inputType = mlir::cast<ShapedType>(getLhs().getType());
  auto newType = RankedTensorType::get(outputShapes[0], inputType.getElementType());
  // getResult().setType(newType);
  return success();
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

LogicalResult SubOp::verify() { return verifyBinaryElementwiseOp(*this); }

LogicalResult SubOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  BinaryElementwiseShapeHelper<SubOp> shapeHelper(*this);
  if (failed(shapeHelper.computeShape()))
    return failure();

  auto outputShapes = shapeHelper.getOutputShapes();
  if (outputShapes.empty())
    return failure();

  auto inputType = mlir::cast<ShapedType>(getLhs().getType());
  auto newType = RankedTensorType::get(outputShapes[0], inputType.getElementType());
  // getResult().setType(newType);
  return success();
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

LogicalResult MulOp::verify() { return verifyBinaryElementwiseOp(*this); }

LogicalResult MulOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  BinaryElementwiseShapeHelper<MulOp> shapeHelper(*this);
  if (failed(shapeHelper.computeShape()))
    return failure();

  auto outputShapes = shapeHelper.getOutputShapes();
  if (outputShapes.empty())
    return failure();

  auto inputType = mlir::cast<ShapedType>(getLhs().getType());
  auto newType = RankedTensorType::get(outputShapes[0], inputType.getElementType());
  // getResult().setType(newType);
  return success();
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

LogicalResult DivOp::verify() { return verifyBinaryElementwiseOp(*this); }

LogicalResult DivOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  BinaryElementwiseShapeHelper<DivOp> shapeHelper(*this);
  if (failed(shapeHelper.computeShape()))
    return failure();

  auto outputShapes = shapeHelper.getOutputShapes();
  if (outputShapes.empty())
    return failure();

  auto inputType = mlir::cast<ShapedType>(getLhs().getType());
  auto newType = RankedTensorType::get(outputShapes[0], inputType.getElementType());
  // getResult().setType(newType);
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/AFIR/AFIROps.cpp.inc"
