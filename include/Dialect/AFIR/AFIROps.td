//===- AFIROps.td - AFIR operation definitions -------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations for the AFIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_OPS
#define AFIR_OPS

include "Dialect/AFIR/AFIRDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base AFIR Binary Operation
//===----------------------------------------------------------------------===//

class AFIR_BinaryOp<string mnemonic, list<Trait> traits = []> :
    AFIR_Op<mnemonic, !listconcat(traits, [
      Pure,
      SameOperandsAndResultType,
      Elementwise
    ])> {
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// AFIR Arithmetic Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AFIR_AddOp : AFIR_BinaryOp<"add", [Commutative]> {
  let summary = "element-wise addition";
  let description = [{
    Performs element-wise addition of two tensors with broadcasting support.

    The operation follows standard broadcasting semantics: when operating on
    tensors of different shapes, the smaller tensor is broadcast to match
    the shape of the larger tensor.

    Example:
    ```mlir
    %result = afir.add %lhs, %rhs : tensor<4x4xf32>
    ```

    Broadcasting example:
    ```mlir
    // tensor<4x1xf32> + tensor<4x4xf32> -> tensor<4x4xf32>
    %result = afir.add %small, %large : tensor<4x4xf32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def AFIR_SubOp : AFIR_BinaryOp<"sub"> {
  let summary = "element-wise subtraction";
  let description = [{
    Performs element-wise subtraction of two tensors with broadcasting support.

    Computes: result = lhs - rhs

    The operation follows standard broadcasting semantics: when operating on
    tensors of different shapes, the smaller tensor is broadcast to match
    the shape of the larger tensor.

    Example:
    ```mlir
    %result = afir.sub %lhs, %rhs : tensor<4x4xf32>
    ```

    Broadcasting example:
    ```mlir
    // tensor<4x4xf32> - tensor<1x4xf32> -> tensor<4x4xf32>
    %result = afir.sub %large, %small : tensor<4x4xf32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def AFIR_MulOp : AFIR_BinaryOp<"mul", [Commutative]> {
  let summary = "element-wise multiplication";
  let description = [{
    Performs element-wise multiplication of two tensors with broadcasting support.

    This is element-wise (Hadamard) multiplication, not matrix multiplication.

    The operation follows standard broadcasting semantics: when operating on
    tensors of different shapes, the smaller tensor is broadcast to match
    the shape of the larger tensor.

    Example:
    ```mlir
    %result = afir.mul %lhs, %rhs : tensor<4x4xf32>
    ```

    Broadcasting example:
    ```mlir
    // Scalar multiplication
    // tensor<1xf32> * tensor<4x4xf32> -> tensor<4x4xf32>
    %result = afir.mul %scalar, %tensor : tensor<4x4xf32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

def AFIR_DivOp : AFIR_BinaryOp<"div"> {
  let summary = "element-wise division";
  let description = [{
    Performs element-wise division of two tensors with broadcasting support.

    Computes: result = lhs / rhs

    The operation follows standard broadcasting semantics: when operating on
    tensors of different shapes, the smaller tensor is broadcast to match
    the shape of the larger tensor.

    Note: Division by zero behavior is undefined and may result in NaN or Inf
    depending on the backend implementation.

    Example:
    ```mlir
    %result = afir.div %lhs, %rhs : tensor<4x4xf32>
    ```

    Broadcasting example:
    ```mlir
    // Divide by scalar
    // tensor<4x4xf32> / tensor<1xf32> -> tensor<4x4xf32>
    %result = afir.div %tensor, %scalar : tensor<4x4xf32>
    ```
  }];
}

#endif // AFIR_OPS
