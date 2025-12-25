//===- ShapeHelperOpInterface.h - Shape helper interface --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
// This file is inspired by onnx-mlir's ShapeHelperOpInterface design.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACE_SHAPEHELPEROPINTERFACE_H
#define MLIR_INTERFACE_SHAPEHELPEROPINTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include <functional>

namespace mlir {
namespace afir {

//===----------------------------------------------------------------------===//
// ShapeHelper Base Class
//===----------------------------------------------------------------------===//

/// ShapeHelper is a base class for implementing shape inference logic.
/// Each operation can have its own ShapeHelper subclass that computes
/// output shapes based on input shapes.
template <typename OpType>
class ShapeHelper {
public:
  ShapeHelper(OpType op) : op(op) {}
  virtual ~ShapeHelper() = default;

  /// Compute the output shapes. Returns failure if shapes cannot be inferred.
  virtual LogicalResult computeShape() = 0;

  /// Get the inferred output shapes.
  ArrayRef<SmallVector<int64_t>> getOutputShapes() const {
    return outputShapes;
  }

protected:
  OpType op;
  SmallVector<SmallVector<int64_t>> outputShapes;
};

//===----------------------------------------------------------------------===//
// BinaryElementwiseShapeHelper
//===----------------------------------------------------------------------===//

/// ShapeHelper for binary elementwise operations (Add, Sub, Mul, Div).
/// These operations require inputs to have the same shape and produce
/// an output with the same shape.
template <typename OpType>
class BinaryElementwiseShapeHelper : public ShapeHelper<OpType> {
public:
  using ShapeHelper<OpType>::ShapeHelper;
  using ShapeHelper<OpType>::op;
  using ShapeHelper<OpType>::outputShapes;

  LogicalResult computeShape() override {
    auto lhsType = mlir::dyn_cast<ShapedType>(op.getLhs().getType());
    auto rhsType = mlir::dyn_cast<ShapedType>(op.getRhs().getType());

    if (!lhsType || !rhsType)
      return failure();

    if (!lhsType.hasRank() || !rhsType.hasRank())
      return failure();

    // For same operands and result type, output shape equals input shape
    SmallVector<int64_t> outputShape;
    for (int64_t dim : lhsType.getShape()) {
      outputShape.push_back(dim);
    }

    outputShapes.clear();
    outputShapes.push_back(std::move(outputShape));
    return success();
  }
};

} // namespace afir
} // namespace mlir

#include "mlir/Interface/ShapeHelperOpInterface.h.inc"

#endif // MLIR_INTERFACE_SHAPEHELPEROPINTERFACE_H
