//===- ShapeHelperOpInterface.h - Shape helper interface --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
// This file is inspired by onnx-mlir's ShapeHelperOpInterface design.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACE_SHAPEHELPEROPINTERFACE_H
#define MLIR_INTERFACE_SHAPEHELPEROPINTERFACE_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace afir {

//===----------------------------------------------------------------------===//
// Forward declarations
//===----------------------------------------------------------------------===//

class IndexExprBuilder;
class IndexExprScope;

//===----------------------------------------------------------------------===//
// Type aliases
//===----------------------------------------------------------------------===//

using DimsExpr = llvm::SmallVector<int64_t>;

//===----------------------------------------------------------------------===//
// AFIR Op Shape Helper
//===----------------------------------------------------------------------===//

/// Base class for shape inference.
///
/// This class and its specialized subclasses are used to compute shapes
/// during shape inference.
///
/// @param op Operation to be analyzed.
///
/// @param operands Operands of the operation to be analyzed. When passing an
/// empty list, the operands are taken from the operations. When passing an
/// explicit, non-empty list, these operands are used instead of the ones from
/// the operation.
///
/// @param ieBuilder Class that scans the operands to gather shape information.
/// Currently unused placeholder for future extension.
///
/// @param scope Index expression scope to be used. Currently unused placeholder
/// for future extension.
struct AFIROpShapeHelper {
  AFIROpShapeHelper(mlir::Operation *op, mlir::ValueRange operands, IndexExprBuilder *ieBuilder = nullptr,
                    IndexExprScope *scope = nullptr)
      : op(op),
        operands(operands),
        createIE(ieBuilder),
        scope(scope),
        ownScope(scope == nullptr),
        ownBuilder(ieBuilder == nullptr) {
    assert(op && "Expecting a valid operation pointer");
    if (operands.size() == 0) {
      privateOperandsCache = llvm::SmallVector<mlir::Value, 4>(op->getOperands().begin(), op->getOperands().end());
      this->operands = mlir::ValueRange(privateOperandsCache);
    }
    privateOutputsDims.resize(op->getNumResults());
  }

  virtual ~AFIROpShapeHelper() {}

  /// Return true if implemented.
  virtual bool isImplemented() {
    return true;
  }

  /// Every leaf class is expected to create a computeShape with the following
  /// signature. This method is responsible to compute at a minimum the output
  /// dims.
  /// Unimplemented operations return success, as these operations may be
  /// transformed later in a sequence of operations with implemented shape
  /// inference. To ensure an implementation, check the `isImplemented` function.
  virtual mlir::LogicalResult computeShape() = 0;

  /// Compute shape and assert on failure.
  void computeShapeAndAssertOnFailure();

  /// Invoke the virtual computeShape, and on success, update the types of the
  /// original operation. First call is used for operations where all the results
  /// share the same output type, second for operations where all results have
  /// their own output types.
  mlir::LogicalResult computeShapeAndUpdateType(mlir::Type elementType, mlir::Attribute encoding = nullptr);
  mlir::LogicalResult computeShapeAndUpdateTypes(mlir::TypeRange elementTypeRange,
                                                 mlir::ArrayRef<mlir::Attribute> encodingList = {});

  /// Get output dims for the N-th output dimension.
  /// Scalar may have a DimsExpr that is empty. Requires an implementation.
  DimsExpr &getOutputDims(int n = 0) {
    if (!isImplemented()) {
      llvm::errs() << "Implementation of shape helper for op " << op->getName()
                   << " is not currently available; please open an issue.\n";
      llvm_unreachable("missing implementation for shape inference");
    }
    return privateOutputsDims[n];
  }

  /// Set output dims, merging the dims associated with the current type with
  /// inferred dims provided here, as appropriate.
  void setOutputDims(const DimsExpr &inferredDims, int n = 0, bool refineShape = true);

  /// Obtain the n-th output result as value.
  mlir::Value getOutput(int n = 0) {
    return op->getResult(n);
  }

  /// Get index expression scope and operation.
  IndexExprScope *getScope() {
    return scope;
  }
  mlir::Operation *getOp() {
    return op;
  }

  /// Set the operands with a vector of Value
  void setOperands(mlir::ValueRange);

 protected:
  /// Helper for ops for which the output (n'th) is the same as the type of a
  /// given input operand's type.
  mlir::LogicalResult setOutputDimsFromOperand(mlir::Value operand, int n = 0, bool refineShape = true);

  /// Helper for ops for which the output (n'th) is a constant shape. Value
  /// ShapedType::kDynamic indicates runtime dim.
  mlir::LogicalResult setOutputDimsFromLiterals(llvm::SmallVector<int64_t, 4> shape, int n = 0,
                                                bool refineShape = true);

  /// Helper for ops for which the output (n'th) is defined by the shape of
  /// another type. Type must have constant shape (all values !=
  /// ShapedType::kDynamic).
  mlir::LogicalResult setOutputDimsFromTypeWithConstantShape(mlir::Type type, int n = 0, bool refineShape = true);

  /// Data that must be present for every ShapeHelper operation. Op and scope
  /// are initialized in the constructor.
  mlir::Operation *op;
  mlir::ValueRange operands;
  IndexExprBuilder *createIE;
  IndexExprScope *scope;

 private:
  /// OutputsDims is computed by the child's struct `computeShape` function. It
  /// can be set using setOutputDims and retrieved using getOutputDims.
  llvm::SmallVector<DimsExpr, 1> privateOutputsDims;
  /// Used to cache the operation's operands (shape inference only).
  llvm::SmallVector<mlir::Value> privateOperandsCache;
  bool ownScope, ownBuilder;
};

//===----------------------------------------------------------------------===//
// Unimplemented Ops (to be used sparingly)
//===----------------------------------------------------------------------===//

struct AFIRUnimplementedOpShapeHelper : public AFIROpShapeHelper {
  AFIRUnimplementedOpShapeHelper(mlir::Operation *op, mlir::ValueRange operands, IndexExprBuilder *ieBuilder = nullptr,
                                 IndexExprScope *scope = nullptr)
      : AFIROpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~AFIRUnimplementedOpShapeHelper() {}

  bool isImplemented() override {
    return false;
  }
  mlir::LogicalResult computeShape() final {
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Type aliases for common ops
//===----------------------------------------------------------------------===//

// These are now defined in ShapeHelper.h

}  // namespace afir
}  // namespace mlir

#include "Interface/ShapeHelperOpInterface.h.inc"

#endif  // MLIR_INTERFACE_SHAPEHELPEROPINTERFACE_H
