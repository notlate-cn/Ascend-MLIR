//===- math.td - AFIR math operations -------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the math operations for the AFIR dialect, organized into
// three categories: unary, binary, and compare operations.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_MATH_OPS
#define AFIR_MATH_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

def AFIR_AddOp : AFIR_BinaryOp<"add", [Pure], [F16, F32, I16, I32]>;
def AFIR_SubOp : AFIR_BinaryOp<"sub", [Pure], [F16, F32, I16, I32]>;
def AFIR_MulOp : AFIR_BinaryOp<"mul", [Pure], [F16, F32, I16, I32]>;
def AFIR_DivOp : AFIR_BinaryOp<"div", [Pure], [F16, F32]>;

//===----------------------------------------------------------------------===//
// Compare Operations
//===----------------------------------------------------------------------===//

#endif // AFIR_MATH_OPS
