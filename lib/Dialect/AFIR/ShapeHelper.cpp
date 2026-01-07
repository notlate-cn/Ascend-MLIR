//===----------------- ShapeHelper.cpp - help for shapes ---------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file has the computations to compute the shapes using the index expr
// approach.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "Dialect/AFIR/ShapeHelper.h"

#include <algorithm>

#define DEBUG_TYPE "shape-helper"

using namespace mlir;

namespace mlir {
namespace afir {

//===----------------------------------------------------------------------===//
// Support functions
//===----------------------------------------------------------------------===//

/// Refine `inferredDims` using the output's shape if possible. For example,
/// replacing a dynamic dim in `inferredDims` by a static dim in the output's
/// shape.
static void refineDims(Operation *op, DimsExpr &inferredDims, Value output) {
  // Nothing to do if the output is unranked.
  auto outputType = dyn_cast<ShapedType>(output.getType());
  if (!outputType || !outputType.hasRank()) return;

  llvm::ArrayRef<int64_t> existingDims = outputType.getShape();
  // Do not handle the case of scalar tensor whose type can be tensor<f32>
  // or tensor<1xf32>. Just use the inferredShape in this case.
  if (existingDims.size() < 1 || inferredDims.size() < 1) return;

  assert((existingDims.size() == inferredDims.size()) &&
         "Inferred shape and existing shape are inconsistent in the number "
         "of elements");

  // Try to update inferredDim if existingDim is static.
  for (unsigned i = 0; i < existingDims.size(); ++i) {
    // ExistingDim is dynamic, nothing to learn from.
    if (existingDims[i] == ShapedType::kDynamic) continue;

    // InferredDim is unknown at shape inference: update it.
    if (inferredDims[i] == ShapedType::kDynamic) {
      inferredDims[i] = existingDims[i];
      continue;
    }
    // inferredDim is different from existingDim. Believe in existingDim.
    if (existingDims[i] != inferredDims[i]) {
      if (op) {
        llvm::outs() << "\nWarning for operation " << op->getName() << ": [Shape inference, dim " << i
                     << "] the inferred dim (" << inferredDims[i] << ") is different from the existing dim ("
                     << existingDims[i] << "). Use the existing dim instead.\n\n";
      } else {
        llvm::outs() << "\nWarning: [Shape inference, dim " << i << "] the inferred dim (" << inferredDims[i]
                     << ") is different from the existing dim (" << existingDims[i]
                     << "). Use the existing dim instead.\n\n";
      }
      inferredDims[i] = existingDims[i];
    }
  }
}

void updateType(mlir::Operation *op, mlir::Value val, llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                mlir::Attribute encoding, bool refineShape) {
  // If elementType is not specified, use the existing element type.
  if (!elementType) {
    auto valType = dyn_cast<ShapedType>(val.getType());
    if (!valType) return;
    elementType = valType.getElementType();
  }

  // If encoding is not specified, use the existing encoding.
  if (!encoding) {
    auto valType = dyn_cast<RankedTensorType>(val.getType());
    if (valType) encoding = valType.getEncoding();
  }

  // Create the new type.
  RankedTensorType newType;
  if (encoding)
    newType = RankedTensorType::get(shape, elementType, encoding);
  else
    newType = RankedTensorType::get(shape, elementType);

  // Update the value type.
  val.setType(newType);
}

//===----------------------------------------------------------------------===//
// AFIR Op Shape Helper
//===----------------------------------------------------------------------===//

void AFIROpShapeHelper::computeShapeAndAssertOnFailure() {
  // Invoke virtual compute shape.
  LogicalResult res = computeShape();
  assert(succeeded(res) && "Failed to compute shape");
}

void AFIROpShapeHelper::setOutputDims(const DimsExpr &inferredDims, int n, bool refineShape) {
  privateOutputsDims[n] = inferredDims;
  if (refineShape) {
    Value output = getOutput(n);
    refineDims(op, privateOutputsDims[n], output);
  }
}

LogicalResult AFIROpShapeHelper::setOutputDimsFromOperand(Value operand, int n, bool refineShape) {
  // Output and operand have the same shape. Just pass the operand shape to the
  // output.
  auto operandType = dyn_cast<ShapedType>(operand.getType());
  if (!operandType || !operandType.hasRank()) return failure();

  DimsExpr outputDims;
  for (int64_t dim : operandType.getShape()) {
    outputDims.push_back(dim);
  }
  setOutputDims(outputDims, n, refineShape);
  return success();
}

LogicalResult AFIROpShapeHelper::setOutputDimsFromLiterals(SmallVector<int64_t, 4> shape, int n, bool refineShape) {
  // Output has the shape given by the vector of integer numbers. Number
  // ShapedType::kDynamic is transformed into a questionmark.
  DimsExpr outputDims;
  for (int64_t dim : shape) {
    outputDims.push_back(dim);
  }
  setOutputDims(outputDims, n, refineShape);
  return success();
}

LogicalResult AFIROpShapeHelper::setOutputDimsFromTypeWithConstantShape(Type type, int n, bool refineShape) {
  auto rankedType = dyn_cast<RankedTensorType>(type);
  if (!rankedType) return failure();

  DimsExpr outputDims;
  for (int64_t dim : rankedType.getShape()) {
    if (dim == ShapedType::kDynamic) return failure();
    outputDims.push_back(dim);
  }
  setOutputDims(outputDims, n, refineShape);
  return success();
}

// Reuse the same type for each of the outputs.
LogicalResult AFIROpShapeHelper::computeShapeAndUpdateType(Type elementType, Attribute encoding) {
  // Invoke virtual compute shape.
  if (failed(computeShape())) return op->emitError("Failed to scan parameters successfully");
  assert((mlir::isa<VectorType>(elementType) || !mlir::isa<ShapedType>(elementType)) &&
         "element type cannot be a shaped type other than vector type");
  uint64_t resNum = op->getNumResults();
  for (uint64_t i = 0; i < resNum; ++i) {
    // If we have an optional type, leave it as is.
    if (mlir::isa<NoneType>(op->getResults()[i].getType())) continue;
    llvm::SmallVector<int64_t, 4> shapeVect(getOutputDims(i).begin(), getOutputDims(i).end());
    // Set refineShape to false here because we refine it (or not) when setting
    // the output shape. So there is no need to perform this again here.
    updateType(op, op->getResults()[i], shapeVect, elementType, encoding,
               /*refineShape*/ false);
  }
  return success();
}

// Use a distinct type for each of the output.
LogicalResult AFIROpShapeHelper::computeShapeAndUpdateTypes(TypeRange elementTypeRange,
                                                            ArrayRef<Attribute> encodingList) {
  uint64_t resNum = op->getNumResults();
  assert((elementTypeRange.size() == resNum) && "Incorrect number of elementTypes");
  bool hasEncoding = encodingList.size() > 0;
  assert((!hasEncoding || encodingList.size() == resNum) && "Incorrect number of encoding");
  // Invoke virtual compute.
  if (failed(computeShape()))
    return op->emitError("Failed to scan " + op->getName().getStringRef() + " parameters successfully");
  for (uint64_t i = 0; i < resNum; ++i) {
    // If we have an optional type, leave it as is.
    if (mlir::isa<NoneType>(op->getResults()[i].getType())) continue;
    llvm::SmallVector<int64_t, 4> shapeVect(getOutputDims(i).begin(), getOutputDims(i).end());
    Type currElementType = elementTypeRange[i];
    // Set refineShape to false here because we refine it (or not) when setting
    // the output shape. So there is no need to perform this again here.
    updateType(op, op->getResults()[i], shapeVect, currElementType, hasEncoding ? encodingList[i] : nullptr,
               /*refineShape*/ false);
  }
  return success();
}

void AFIROpShapeHelper::setOperands(ValueRange inputs) {
  // Note: do not use operands until it is re-assigned
  privateOperandsCache = llvm::SmallVector<Value, 4>(inputs.begin(), inputs.end());
  operands = ValueRange(privateOperandsCache);
}

//===----------------------------------------------------------------------===//
// AFIR Broadcast Op Shape Helper
//===----------------------------------------------------------------------===//

LogicalResult AFIRBroadcastOpShapeHelper::computeShape() {
  DimsExpr dimsExpr;
  uint64_t numOfInputs = operands.size();

  if (numOfInputs == 0) return failure();

  for (Value operand : operands) {
    auto shapedType = dyn_cast<ShapedType>(operand.getType());
    if (!shapedType || !shapedType.hasRank()) return failure();
  }

  // Compute rank of the output. Rank of the output is the maximum rank of all
  // inputs.
  outputRank = 0;
  for (Value operand : operands) {
    auto shapedType = cast<ShapedType>(operand.getType());
    outputRank = std::max(outputRank, (uint64_t)shapedType.getRank());
  }
  dimsExpr.resize(outputRank);

  // Prepare dims for every input. Prepend 1s if the input's shape has smaller
  // rank, so that all the shapes have the same rank.
  int64_t one = 1;
  for (uint64_t i = 0; i < numOfInputs; ++i) {
    auto shapedType = cast<ShapedType>(operands[i].getType());
    uint64_t r = shapedType.getRank();

    DimsExpr dims(outputRank - r, one);
    for (uint64_t k = 0; k < r; ++k) {
      dims.push_back(shapedType.getShape()[k]);
    }
    inputsDims.emplace_back(dims);
  }

  // Initialize the output with the first operand.
  dimsExpr = inputsDims[0];

  // Now compute each broadcasted dimension for the output. Folding over the
  // other operands along the current dimension index.
  for (uint64_t i = 1; i < numOfInputs; ++i) {
    for (uint64_t j = 0; j < outputRank; ++j) {
      // Set the output dimension based on the two dimension values.
      // Dimension value can be one of 1, Dynamic, LiteralNot1.
      int64_t currentDim = dimsExpr[j];
      int64_t nextDim = inputsDims[i][j];

      // Case: 1 - *.
      if (currentDim == 1) {
        if (!hasUniBroadcasting) {
          dimsExpr[j] = nextDim;
        }
        continue;
      }

      // Case: LiteralNot1 - *.
      if (currentDim != 1 && currentDim != ShapedType::kDynamic) {
        // LiteralNot1 - LiteralNot1 => keep unchanged with verifying.
        if (nextDim != 1 && nextDim != ShapedType::kDynamic && currentDim != nextDim)
          return op->emitOpError("Incompatible broadcast matching " + std::to_string(currentDim) + " with " +
                                 std::to_string(nextDim));
        // Case: LiteralNot1 - (Dynamic or 1) => Keep unchanged without
        // verifying.
        continue;
      }

      // Case: Dynamic - 1 => keep unchanged.
      if (currentDim == ShapedType::kDynamic && nextDim == 1) {
        continue;
      }

      // Case Dynamic - LiteralNot1 => set to LiteralNot1 without verifying.
      if (currentDim == ShapedType::kDynamic && nextDim != 1 && nextDim != ShapedType::kDynamic) {
        dimsExpr[j] = nextDim;
        continue;
      }

      // Case: Dynamic - Dynamic
      if (!hasUniBroadcasting) {
        dimsExpr[j] = ShapedType::kDynamic;
      }
    }
  }

  // Set the final output.
  setOutputDims(dimsExpr);
  return success();
}

//===----------------------------------------------------------------------===//
// AFIR Unary Op Shape Helper
//===----------------------------------------------------------------------===//

LogicalResult AFIRUnaryOpShapeHelper::computeShape() {
  if (operands.size() == 0) return failure();

  auto shapedType = dyn_cast<ShapedType>(operands[0].getType());
  if (!shapedType || !shapedType.hasRank()) return failure();

  return setOutputDimsFromOperand(operands[0]);
}

}  // namespace afir
}  // namespace mlir