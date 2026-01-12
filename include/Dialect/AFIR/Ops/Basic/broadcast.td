//===- broadcast.td - AFIR broadcast operations -------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the broadcast operations for the AFIR dialect
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_BROADCAST_OPS
#define AFIR_BROADCAST_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def AFIR_BroadcastOp : AFIR_UnaryOp<"broadcast", [Pure], [I8, UI8, I16, UI16, I32, UI32, UI64, F16, F32], "Broadcast", "Vector", 1>;

#endif // AFIR_BROADCAST_OPS