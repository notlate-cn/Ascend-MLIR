//===- math.td - AFIR math operations -------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the math operations for the AFIR dialect, organized into
// three categories: unary, binary, and compare operations.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_MATH_OPS
#define AFIR_MATH_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "Interface/ShapeHelperOpInterface.td"
include "Interface/ShapeInferenceOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

def AFIR_AddOp : AFIR_BinaryOp<"add", [
  Pure,
  DeclareOpInterfaceMethods<ShapeHelperOpInterface>,
  DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
], [F16, F32, I16, I32]> {
  let summary = "Element-wise addition operation";
  let description = [{
    Performs element-wise addition of two tensors.

    Example:
    ```mlir
    %result = afir.add %lhs, %rhs : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    ```
  }];

  let extraClassDefinition = [{
    mlir::afir::AFIROpShapeHelper * $cppClass::getShapeHelper(mlir::Operation *op, 
        llvm::ArrayRef<mlir::Value> operands, mlir::afir::IndexExprBuilder *ieBuilder, 
        mlir::afir::IndexExprScope *scope) {
      mlir::afir::AFIROpShapeHelper *sh = new mlir::afir::AFIRAddOpShapeHelper(op, operands, ieBuilder, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def AFIR_SubOp : AFIR_BinaryOp<"sub", [
  Pure,
  DeclareOpInterfaceMethods<ShapeHelperOpInterface>,
  DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
], [F16, F32, I16, I32]> {
  let summary = "Element-wise subtraction operation";
  let description = [{
    Performs element-wise subtraction of two tensors.

    Example:
    ```mlir
    %result = afir.sub %lhs, %rhs : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    ```
  }];

  let extraClassDefinition = [{
    mlir::afir::AFIROpShapeHelper * $cppClass::getShapeHelper(mlir::Operation *op, 
        llvm::ArrayRef<mlir::Value> operands, mlir::afir::IndexExprBuilder *ieBuilder, 
        mlir::afir::IndexExprScope *scope) {
      mlir::afir::AFIROpShapeHelper *sh = new mlir::afir::AFIRSubOpShapeHelper(op, operands, ieBuilder, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def AFIR_MulOp : AFIR_BinaryOp<"mul", [
  Pure,
  DeclareOpInterfaceMethods<ShapeHelperOpInterface>,
  DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
], [F16, F32, I16, I32]> {
  let summary = "Element-wise multiplication operation";
  let description = [{
    Performs element-wise multiplication of two tensors.

    Example:
    ```mlir
    %result = afir.mul %lhs, %rhs : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    ```
  }];

  let extraClassDefinition = [{
    mlir::afir::AFIROpShapeHelper * $cppClass::getShapeHelper(mlir::Operation *op, 
        llvm::ArrayRef<mlir::Value> operands, mlir::afir::IndexExprBuilder *ieBuilder, 
        mlir::afir::IndexExprScope *scope) {
      mlir::afir::AFIROpShapeHelper *sh = new mlir::afir::AFIRMulOpShapeHelper(op, operands, ieBuilder, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def AFIR_DivOp : AFIR_BinaryOp<"div", [
  Pure,
  DeclareOpInterfaceMethods<ShapeHelperOpInterface>,
  DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
], [F16, F32]> {
  let summary = "Element-wise division operation";
  let description = [{
    Performs element-wise division of two tensors.

    Example:
    ```mlir
    %result = afir.div %lhs, %rhs : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    ```
  }];

  let extraClassDefinition = [{
    mlir::afir::AFIROpShapeHelper * $cppClass::getShapeHelper(mlir::Operation *op, 
        llvm::ArrayRef<mlir::Value> operands, mlir::afir::IndexExprBuilder *ieBuilder, 
        mlir::afir::IndexExprScope *scope) {
      mlir::afir::AFIROpShapeHelper *sh = new mlir::afir::AFIRDivOpShapeHelper(op, operands, ieBuilder, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Compare Operations
//===----------------------------------------------------------------------===//

#endif // AFIR_MATH_OPS
