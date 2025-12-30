//===----------------- ShapeHelper.hpp - help for shapes ---------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file has the computations to compute the shapes using the index expr
// approach.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_SHAPE_HELPER_H
#define AFIR_SHAPE_HELPER_H

#include <utility>

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "Interface/ShapeHelperOpInterface.h"

namespace mlir {
namespace afir {

//===----------------------------------------------------------------------===//
// Support functions.
//===----------------------------------------------------------------------===//

// Update a tensor type by using the given shape, elementType and encoding.
void updateType(mlir::Operation *op, mlir::Value val, llvm::ArrayRef<int64_t> shape, mlir::Type elementType = nullptr,
                mlir::Attribute encoding = nullptr, bool refineShape = true);

//===----------------------------------------------------------------------===//
// Broadcast Ops
//===----------------------------------------------------------------------===//

// Compute a broadcasted shape from the shapes of given operands. Operands must
// be ranked in advance.
struct AFIRBroadcastOpShapeHelper : public AFIROpShapeHelper {
  AFIRBroadcastOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands, IndexExprBuilder *ieBuilder = nullptr,
                             IndexExprScope *scope = nullptr, bool hasUniBroadcasting = false)
      : AFIROpShapeHelper(op, operands, ieBuilder, scope),
        inputsDims(),
        outputRank(0),
        hasUniBroadcasting(hasUniBroadcasting) {}
  virtual ~AFIRBroadcastOpShapeHelper() {}

  // Default shape compute (every operands of the operation and no additional
  // parameters).
  mlir::LogicalResult computeShape() override;

  // A vector of input shapes where dimensions are padded with 1 if necessary,
  // so that all inputs have the same rank. Instantiated during ComputeShape.
  llvm::SmallVector<DimsExpr, 4> inputsDims;
  // Rank of the output shape.
  uint64_t outputRank;

 protected:
  // When unidirectional broadcasting is true, the other operands are always
  // unidirectional broadcastable to the first operand.
  bool hasUniBroadcasting;
};

//===----------------------------------------------------------------------===//
// Unary Ops
//===----------------------------------------------------------------------===//

/// Compute an output shape for a unary element-wise operation. The output and
/// input of an unary element-wise operation have the same shape.
struct AFIRUnaryOpShapeHelper : public AFIRBroadcastOpShapeHelper {
  AFIRUnaryOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands, IndexExprBuilder *ieBuilder = nullptr,
                         IndexExprScope *scope = nullptr)
      : AFIRBroadcastOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~AFIRUnaryOpShapeHelper() {}

  mlir::LogicalResult computeShape() override;
};

//===----------------------------------------------------------------------===//
// Type aliases for common ops
//===----------------------------------------------------------------------===//

using AFIRAddOpShapeHelper = AFIRBroadcastOpShapeHelper;
using AFIRSubOpShapeHelper = AFIRBroadcastOpShapeHelper;
using AFIRMulOpShapeHelper = AFIRBroadcastOpShapeHelper;
using AFIRDivOpShapeHelper = AFIRBroadcastOpShapeHelper;

}  // namespace afir
}  // namespace mlir

#endif  // AFIR_SHAPE_HELPER_H