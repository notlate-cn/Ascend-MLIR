//===- ShapeInferenceOpInterface.td - Shape inference interface -*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
// This file is inspired by onnx-mlir's ShapeInferenceOpInterface design.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACE_SHAPEINFERENCEOPINTERFACE
#define MLIR_INTERFACE_SHAPEINFERENCEOPINTERFACE

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInferenceOpInterface"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference. The shape inference
    interface method takes a shape inference function pointer as input, this
    allows shape inference to proceed recursively.

    Operations implicitly associated with other (sub) computation graphs can
    run shape inference on their dependent computation graphs; this is
    necessary because the shape inference procedure of such operations (e.g.,
    Loop operation) depends on the shape inference outcome of their associated
    (sub) computation graphs.
  }];

  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Infer and set the output shape for the current operation.
      }],
      /*retTy=*/"mlir::LogicalResult",
      /*methodName=*/"inferShapes",
      /*args=*/(ins "std::function<void(mlir::Region &)>":$shapeInferenceFunc)
    >
  ];
}

#endif // MLIR_INTERFACE_SHAPEINFERENCEOPINTERFACE
