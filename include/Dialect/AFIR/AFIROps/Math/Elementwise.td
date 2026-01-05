//===- Elementwise.td - AFIR elementwise operations --------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the elementwise operations for the AFIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_ELEMENTWISE_OPS
#define AFIR_ELEMENTWISE_OPS

include "mlir/IR/CommonAttrConstraints.td"
include "Dialect/AFIR/AFIRDialect.td"
include "Dialect/AFIR/AFIRBase.td"
include "Interface/ShapeHelperOpInterface.td"
include "Interface/ShapeInferenceOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Core/Types.td"

//===----------------------------------------------------------------------===//
// Add Operation
//===----------------------------------------------------------------------===//

def AFIR_AddOp : AFIR_Op<"add", [
  Pure,
  DeclareOpInterfaceMethods<ShapeHelperOpInterface>,
  DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
]> {
  let summary = "Element-wise addition operation";
  let description = [{
    Performs element-wise addition of two tensors.

    Example:
    ```mlir
    %result = afir.add %lhs, %rhs : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    ```
  }];

  let arguments = (ins
    ShapedTypeOf<[F16, F32, I16, I32]>:$lhs,
    ShapedTypeOf<[F16, F32, I16, I32]>:$rhs,
    AffineMapArrayAttr:$indexing_maps
  );
  let results = (outs ShapedTypeOf<[F16, F32, I16, I32]>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `indexing_maps` `=` $indexing_maps attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 2;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static bool isSupportVectorFunction() {
      return true;
    }
    static ComputeType getComputeType() {
      return ComputeType::Elewise;
    }
    static ComputeUnit getComputeUnit() {
      return ComputeUnit::Vector;
    }
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

//===----------------------------------------------------------------------===//
// Sub Operation
//===----------------------------------------------------------------------===//

def AFIR_SubOp : AFIR_Op<"sub", [
  Pure,
  DeclareOpInterfaceMethods<ShapeHelperOpInterface>,
  DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
]> {
  let summary = "Element-wise subtraction operation";
  let description = [{
    Performs element-wise subtraction of two tensors.

    Example:
    ```mlir
    %result = afir.sub %lhs, %rhs : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    ```
  }];

  let arguments = (ins
    ShapedTypeOf<[F16, F32, I16, I32]>:$lhs,
    ShapedTypeOf<[F16, F32, I16, I32]>:$rhs,
    AffineMapArrayAttr:$indexing_maps
  );
  let results = (outs ShapedTypeOf<[F16, F32, I16, I32]>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `indexing_maps` `=` $indexing_maps attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 2;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static bool isSupportVectorFunction() {
      return true;
    }
    static ComputeType getComputeType() {
      return ComputeType::Elewise;
    }
    static ComputeUnit getComputeUnit() {
      return ComputeUnit::Vector;
    }
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

//===----------------------------------------------------------------------//
// Mul Operation
//===----------------------------------------------------------------------//

def AFIR_MulOp : AFIR_Op<"mul", [
  Pure,
  DeclareOpInterfaceMethods<ShapeHelperOpInterface>,
  DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
]> {
  let summary = "Element-wise multiplication operation";
  let description = [{
    Performs element-wise multiplication of two tensors.

    Example:
    ```mlir
    %result = afir.mul %lhs, %rhs : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    ```
  }];

  let arguments = (ins
    ShapedTypeOf<[F16, F32, I16, I32]>:$lhs,
    ShapedTypeOf<[F16, F32, I16, I32]>:$rhs,
    AffineMapArrayAttr:$indexing_maps
  );
  let results = (outs ShapedTypeOf<[F16, F32, I16, I32]>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `indexing_maps` `=` $indexing_maps attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 2;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static bool isSupportVectorFunction() {
      return true;
    }
    static ComputeType getComputeType() {
      return ComputeType::Elewise;
    }
    static ComputeUnit getComputeUnit() {
      return ComputeUnit::Vector;
    }
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

//===----------------------------------------------------------------------===//
// Div Operation
//===----------------------------------------------------------------------===//

def AFIR_DivOp : AFIR_Op<"div", [
  Pure,
  DeclareOpInterfaceMethods<ShapeHelperOpInterface>,
  DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
]> {
  let summary = "Element-wise division operation";
  let description = [{
    Performs element-wise division of two tensors.

    Example:
    ```mlir
    %result = afir.div %lhs, %rhs : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    ```
  }];

  let arguments = (ins
    ShapedTypeOf<[F16, F32, I16, I32]>:$lhs,
    ShapedTypeOf<[F16, F32, I16, I32]>:$rhs,
    AffineMapArrayAttr:$indexing_maps
  );
  let results = (outs ShapedTypeOf<[F16, F32, I16, I32]>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `indexing_maps` `=` $indexing_maps attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 2;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static bool isSupportVectorFunction() {
      return false;
    }
    static ComputeType getComputeType() {
      return ComputeType::Elewise;
    }
    static ComputeUnit getComputeUnit() {
      return ComputeUnit::Vector;
    }
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

#endif // AFIR_ELEMENTWISE_OPS
