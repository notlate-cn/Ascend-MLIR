//===- AFIR.h - AFIR dialect declaration -----------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFIR_AFIR_H
#define MLIR_DIALECT_AFIR_AFIR_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

// Include dialect definition
#include "Dialect/AFIR/Dialect.h.inc"

// Include enums (must be before attributes)
#include "Dialect/AFIR/Enums.h.inc"

// Include attribute definitions
#define GET_ATTRDEF_CLASSES
#include "Dialect/AFIR/Attrs.h.inc"

// Include operation definitions
#define GET_OP_CLASSES
#include "Dialect/AFIR/Ops.h.inc"

#endif  // MLIR_DIALECT_AFIR_AFIR_H