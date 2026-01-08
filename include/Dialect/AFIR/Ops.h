//===- AFIROps.h - AFIR operation declarations ------------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFIR_AFIROPS_H
#define MLIR_DIALECT_AFIR_AFIROPS_H

#include "Dialect/AFIR/Dialect.h"
#include "Dialect/AFIR/ShapeHelper.h"
#include "Interface/ShapeHelperOpInterface.h"
#include "Interface/ShapeInferenceOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "ascir/Dialect/Asc/IR/Asc.h"

// Include operation definitions
#define GET_OP_CLASSES
#include "Dialect/AFIR/Ops.h.inc"

#endif  // MLIR_DIALECT_AFIR_AFIROPS_H
