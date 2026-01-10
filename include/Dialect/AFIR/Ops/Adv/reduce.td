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
    AFIR_UnaryOp<mnemonic, traits, types, "Reduce", "Vector">;

def AFIR_MaxOp : AFIR_UnaryMathOp<"max", [Pure], [F16, F32]>;
def AFIR_MinOp : AFIR_UnaryMathOp<"min", [Pure], [F16, F32]>;
def AFIR_SumOp : AFIR_UnaryMathOp<"sum", [Pure], [I32, F32]>;
def AFIR_MeanOp : AFIR_UnaryMathOp<"mean", [Pure], [F32]>;
def AFIR_ProdOp : AFIR_UnaryMathOp<"prod", [Pure], [F32]>;
def AFIR_AnyOp : AFIR_UnaryMathOp<"any", [Pure], [F32]>;
def AFIR_AllOp : AFIR_UnaryMathOp<"all", [Pure], [F32]>;

#endif // AFIR_REDUCE_OPS