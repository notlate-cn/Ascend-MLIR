//===- concat.td - AFIR concat operations -------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the concat operations for the AFIR dialect
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CONCAT_OPS
#define AFIR_CONCAT_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def AFIR_ConcatOp : AFIR_VariadicOp<"concat", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, F16, F32], "Concat", "Vector", 1> {
  let arguments = !con(
    (ins 
      Variadic<TensorOf<[I8, UI8, I16, UI16, I32, UI32, I64, F16, F32]>>:$inputs,
      I32Attr:$concat_axis
    ),
    commonArgs
  );
}

#endif // AFIR_CONCAT_OPS