//===- data.td - AFIR data operations -------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the data operations for the AFIR dialect
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_DATA_OPS
#define AFIR_DATA_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def AFIR_ScalarOp : AFIR_NullaryOpBase<"scalar", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32], "Invalid", "Scalar", 1>;
def AFIR_IndexExprOp : AFIR_NullaryOpBase<"index_expr", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32], "Invalid", "Vector", 1>;
def AFIR_DataOp : AFIR_NullaryOpBase<"data", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32], "Invalid", "None", 1>;
def AFIR_OutputOp : AFIR_UnaryOp<"output", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32], "Invalid", "None", 1>;
def AFIR_LoadOp : AFIR_UnaryOp<"load", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, F16, F32], "Load", "MTE2", 1>;
def AFIR_StoreOp : AFIR_UnaryOp<"store", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, F16, F32], "Store", "MTE3", 1>;

#endif // AFIR_DATA_OPS