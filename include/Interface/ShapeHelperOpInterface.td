//===- ShapeHelperOpInterface.td - Shape helper interface --*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
// This file is inspired by onnx-mlir's ShapeHelperOpInterface design.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACE_SHAPEHELPEROPINTERFACE
#define MLIR_INTERFACE_SHAPEHELPEROPINTERFACE

include "mlir/IR/OpBase.td"

def ShapeHelperOpInterface : OpInterface<"ShapeHelperOpInterface"> {
  let description = [{
    Interface to access a registered method to infer the return shapes for an
    operation that can be used during type inference. The shape inference
    interface method takes an operation and its operands as input, augmented
    by an index expression builder and an index expression scope as optional
    inputs. It returns a specialized shape helper object for that operation,
    casted as a generic AFIROpShapeHelper.

    For operations that do not support shape helpers at this stage, a
    `AFIRUnimplementedOpShapeHelper` object is returned. This object does not
    compute shapes, and simply return success when `computeShape` is called
    on it. Users may verify if an operation has an actual implementation by
    calling `isImplemented()` on the shape helper object. An implementation
    is required when attempting to read the outputs of a shape helper object
    via the `getOutputDims` method.

    The new object is allocated on the heap and it is the responsability
    of the object user to free the memory after last use.
  }];

  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Provide a shape helper object that can be used to compute
        the output shape for the current operation.
      }],
      /*retTy=*/"::mlir::afir::AFIROpShapeHelper *",
      /*methodName=*/"getShapeHelper",
      /*args=*/(ins "mlir::Operation *":$op,
                   "llvm::ArrayRef<mlir::Value>":$operands,
                   "::mlir::afir::IndexExprBuilder *":$ieBuilder,
                   "::mlir::afir::IndexExprScope *":$scope)
    >
  ];
}

#endif // MLIR_INTERFACE_SHAPEHELPEROPINTERFACE
