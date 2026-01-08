//===- AFIRDialect.h - AFIR dialect declaration -----------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFIR_AFIRDIALECT_H
#define MLIR_DIALECT_AFIR_AFIRDIALECT_H

#include "mlir/IR/Dialect.h"

// Include dialect definition
#include "Dialect/AFIR/Dialect.h.inc"

// Include enums (must be before attributes)
#include "Dialect/AFIR/Enums.h.inc"

// Include attribute definitions
#define GET_ATTRDEF_CLASSES
#include "Dialect/AFIR/Attrs.h.inc"

#endif  // MLIR_DIALECT_AFIR_AFIRDIALECT_H
