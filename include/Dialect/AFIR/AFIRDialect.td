//===- AFIRDialect.td - AFIR dialect definition ------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file is the main entry point for the AFIR dialect TableGen definitions.
// It includes all dialect components: base definitions, types, and attributes.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_DIALECT
#define AFIR_DIALECT

// Include base definitions (dialect, common classes, and fundamental enums)
include "Dialect/AFIR/AFIRBase.td"

// Include attribute definitions (both operation-level and graph-level)
include "Dialect/AFIR/AFIRAttrs.td"

#endif // AFIR_DIALECT
