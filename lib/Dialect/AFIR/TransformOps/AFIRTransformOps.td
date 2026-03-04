//===- AFIRTransformOps.td - AFIR transformation ops -*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_TRANSFORM_OPS
#define AFIR_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/OpBase.td"

def AddIndexArgsOp : Op<Transform_Dialect,
    "func.add_index_args",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Add index arguments to a function";
  let description = [{
    This transform takes a function operation and adds a specified number
    of index type arguments to its argument list. The new arguments are
    appended to the end of the existing argument list.

    The transform returns:
    - The modified function operation
    - Handles to each of the newly added index arguments

    This transform will emit a silenceable failure if:
     - The handle does not contain exactly one function operation
     - The function operation is not a func.func operation

    This transform will emit a definite failure if:
     - Failed to modify the function signature
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$target,
    I32Attr:$num_args);
  let results = (outs
    TransformHandleTypeInterface:$transformed,
    Variadic<TransformHandleTypeInterface>:$new_args);

  let assemblyFormat = [{
    $target `,` $num_args attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

#endif // AFIR_TRANSFORM_OPS
