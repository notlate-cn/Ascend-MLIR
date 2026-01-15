//===- concat.cpp - AFIR concat operation implementations -*- C++ -*-===//
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
// Concat Operations
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::verify() {
  if (getInputs().empty()) {
    return emitOpError("requires at least one input tensor");
  }

  int64_t axis = getConcatAxisAttr().getInt();

  auto firstTensorType = llvm::dyn_cast<RankedTensorType>(getInputs()[0].getType());
  if (!firstTensorType) {
    return emitOpError("requires ranked tensor inputs");
  }

  int64_t rank = firstTensorType.getRank();

  if (axis < 0 || axis >= rank) {
    return emitOpError("axis ") << axis << " is out of bounds [0, " << rank << ")";
  }

  ArrayRef<int64_t> referenceShape = firstTensorType.getShape();
  Type referenceElementType = firstTensorType.getElementType();

  int64_t concatDimSize = referenceShape[axis];

  for (auto input : llvm::drop_begin(getInputs(), 1)) {
    auto tensorType = llvm::dyn_cast<RankedTensorType>(input.getType());
    if (!tensorType) {
      return emitOpError("all inputs must be ranked tensors");
    }

    if (tensorType.getRank() != rank) {
      return emitOpError("all inputs must have the same rank");
    }

    if (tensorType.getElementType() != referenceElementType) {
      return emitOpError("all inputs must have the same element type");
    }

    ArrayRef<int64_t> shape = tensorType.getShape();
    for (int64_t i = 0; i < rank; ++i) {
      if (i == axis) {
        if (shape[i] != ShapedType::kDynamic && concatDimSize != ShapedType::kDynamic) {
          concatDimSize += shape[i];
        } else {
          concatDimSize = ShapedType::kDynamic;
        }
      } else {
        if (shape[i] != referenceShape[i] && shape[i] != ShapedType::kDynamic &&
            referenceShape[i] != ShapedType::kDynamic) {
          return emitOpError("all inputs must have the same shape except on axis ") << axis;
        }
      }
    }
  }

  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  SmallVector<int64_t> expectedShape(referenceShape.begin(), referenceShape.end());
  expectedShape[axis] = concatDimSize;

  if (resultType.getShape() != ArrayRef<int64_t>(expectedShape)) {
    return emitOpError("result shape mismatch");
  }

  return success();
}
