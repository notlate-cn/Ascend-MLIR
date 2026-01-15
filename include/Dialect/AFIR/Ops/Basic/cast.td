//===- cast.td - AFIR cast operations -------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the cast operations for the AFIR dialect
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CAST_OPS
#define AFIR_CAST_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

class AFIR_UnaryCASTOp<string mnemonic, list<Trait> traits = [], list<Type> types> :
    AFIR_UnaryOp<mnemonic, traits, types, "CAST", "Vector", 1>;

def AFIR_CastOp : AFIR_UnaryOpBase<"cast", 
    [Pure], [I4, I8, UI8, I16, I32, UI32, I64, UI64, F16, F32, BF16], 
    [I4, I8, UI8, I16, UI16, I32, UI32, I64, F16, F32, BF16], 
    "Elementwise", "Vector", 1>;

#endif // AFIR_CAST_OPS