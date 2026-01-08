//===- AFIROps.td - AFIR operation definitions -------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations for the AFIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_OPS
#define AFIR_OPS

include "Dialect/AFIR/Dialect.td"
include "Interface/ShapeHelperOpInterface.td"
include "Interface/ShapeInferenceOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Math Operations
//===----------------------------------------------------------------------===//

include "Dialect/AFIR/Ops/Math/Elementwise.td"

#endif // AFIR_OPS