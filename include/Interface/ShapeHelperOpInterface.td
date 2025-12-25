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
    Interface for operations that can infer their output shapes from their
    input shapes. This interface is inspired by onnx-mlir's ShapeHelper design.
  }];

  let cppNamespace = "::mlir::afir";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Infer the output shapes of the operation based on the input shapes.
        Returns success if shape inference succeeds, failure otherwise.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"inferShapes",
      /*args=*/(ins "std::function<void(mlir::Region &)>":$doShapeInference)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Check if the operation has static shapes for all inputs and outputs.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasStaticShape",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImpl=*/[{
        for (auto operand : $_op->getOperands()) {
          if (auto shapedType = mlir::dyn_cast<ShapedType>(operand.getType())) {
            if (!shapedType.hasStaticShape())
              return false;
          }
        }
        for (auto result : $_op->getResults()) {
          if (auto shapedType = mlir::dyn_cast<ShapedType>(result.getType())) {
            if (!shapedType.hasStaticShape())
              return false;
          }
        }
        return true;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the rank of the output tensor at the given index.
        Returns -1 if the rank is unknown.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getOutputRank",
      /*args=*/(ins "unsigned":$outputIndex),
      /*methodBody=*/"",
      /*defaultImpl=*/[{
        if (outputIndex >= $_op->getNumResults())
          return -1;
        auto resultType = $_op->getResult(outputIndex).getType();
        if (auto shapedType = mlir::dyn_cast<ShapedType>(resultType))
          return shapedType.hasRank() ? shapedType.getRank() : -1;
        return -1;
      }]
    >
  ];
}

#endif // MLIR_INTERFACE_SHAPEHELPEROPINTERFACE
