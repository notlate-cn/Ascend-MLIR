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
include "Interface/ShapeHelperOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// AFIR Binary Elementwise Operations
//===----------------------------------------------------------------------===//

class AFIR_BinaryElementwiseOp<string mnemonic, list<Trait> traits = []> :
    AFIR_Op<mnemonic, !listconcat(traits, [
      Pure,
      SameOperandsAndResultType,
      DeclareOpInterfaceMethods<ShapeHelperOpInterface>
    ])> {
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Add Operation
//===----------------------------------------------------------------------===//

def AFIR_AddOp : AFIR_BinaryElementwiseOp<"add"> {
  let summary = "Element-wise addition operation";
  let description = [{
    Performs element-wise addition of two tensors.

    Example:
    ```mlir
    %result = afir.add %lhs, %rhs : tensor<4x4xf32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Sub Operation
//===----------------------------------------------------------------------===//

def AFIR_SubOp : AFIR_BinaryElementwiseOp<"sub"> {
  let summary = "Element-wise subtraction operation";
  let description = [{
    Performs element-wise subtraction of two tensors.

    Example:
    ```mlir
    %result = afir.sub %lhs, %rhs : tensor<4x4xf32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Mul Operation
//===----------------------------------------------------------------------===//

def AFIR_MulOp : AFIR_BinaryElementwiseOp<"mul"> {
  let summary = "Element-wise multiplication operation";
  let description = [{
    Performs element-wise multiplication of two tensors.

    Example:
    ```mlir
    %result = afir.mul %lhs, %rhs : tensor<4x4xf32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Div Operation
//===----------------------------------------------------------------------===//

def AFIR_DivOp : AFIR_BinaryElementwiseOp<"div"> {
  let summary = "Element-wise division operation";
  let description = [{
    Performs element-wise division of two tensors.

    Example:
    ```mlir
    %result = afir.div %lhs, %rhs : tensor<4x4xf32>
    ```
  }];
}

#endif // AFIR_OPS