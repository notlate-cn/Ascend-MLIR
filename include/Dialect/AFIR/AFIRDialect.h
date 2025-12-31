//===- AFIRDialect.h - AFIR dialect declaration -----------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFIR_AFIRDIALECT_H
#define MLIR_DIALECT_AFIR_AFIRDIALECT_H

#include "mlir/IR/Dialect.h"

// Include dialect definition
#include "Dialect/AFIR/AFIRDialect.h.inc"

// Include enums (must be before attributes)
#include "Dialect/AFIR/AFIREnums.h.inc"

// Include attribute definitions
#define GET_ATTRDEF_CLASSES
#include "Dialect/AFIR/AFIRAttrs.h.inc"

#endif  // MLIR_DIALECT_AFIR_AFIRDIALECT_H
