//===- condition.td - AFIR condition operations -------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the condition operations for the AFIR dialect
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CONDITION_OPS
#define AFIR_CONDITION_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

class AFIR_TernaryConditionOp<string mnemonic, list<Trait> traits = [], list<Type> conditionTypes, list<Type> types> :
    AFIR_TernaryOpBase<mnemonic, traits, conditionTypes, types, types, types, "Elementwise", "Vector", 1>;

def AFIR_SelectOp : AFIR_TernaryConditionOp<"select", [Pure], [UI8], [I16, I32, I64, F16, F32]>;
def AFIR_WhereOp : AFIR_TernaryConditionOp<"where", [Pure], [UI8], [I16, I32, I64, F16, F32]>;

#endif // AFIR_CONDITION_OPS