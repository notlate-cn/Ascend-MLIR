//===- reduce.td - AFIR reduce operations -------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the reduce operations for the AFIR dialect
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_REDUCE_OPS
#define AFIR_REDUCE_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

class AFIR_UnaryReduceOp<string mnemonic, list<Trait> traits = [], list<Type> types> :
    AFIR_UnaryOpBase<mnemonic, traits, types, types, "Reduce", "Vector", 1> {
  let arguments = !con(
    (ins 
      TensorOf<types>:$input,
      I32Attr:$axis
    ),
    commonArgs
  );
}

def AFIR_MaxOp : AFIR_UnaryReduceOp<"max", [Pure], [F16, F32]>;
def AFIR_MinOp : AFIR_UnaryReduceOp<"min", [Pure], [F16, F32]>;
def AFIR_SumOp : AFIR_UnaryReduceOp<"sum", [Pure], [I32, F32]>;
def AFIR_MeanOp : AFIR_UnaryReduceOp<"mean", [Pure], [F32]>;
def AFIR_ProdOp : AFIR_UnaryReduceOp<"prod", [Pure], [F32]>;
def AFIR_AnyOp : AFIR_UnaryReduceOp<"any", [Pure], [F32]>;
def AFIR_AllOp : AFIR_UnaryReduceOp<"all", [Pure], [F32]>;

#endif // AFIR_REDUCE_OPS