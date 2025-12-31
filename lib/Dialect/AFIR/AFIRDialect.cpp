//===- AFIRDialect.cpp - AFIR dialect implementation ------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIRDialect.h"
#include "Dialect/AFIR/AFIROps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::afir;

//===----------------------------------------------------------------------===//
// AFIR dialect initialization
//===----------------------------------------------------------------------===//

// Include generated dialect implementation
#include "Dialect/AFIR/AFIRDialect.cpp.inc"

// Include generated enum definitions
#include "Dialect/AFIR/AFIREnums.cpp.inc"

// Include generated attribute definitions
#define GET_ATTRDEF_CLASSES
#include "Dialect/AFIR/AFIRAttrs.cpp.inc"

// Include generated type definitions
#define GET_TYPEDEF_CLASSES
#include "Dialect/AFIR/AFIRTypes.cpp.inc"

void AFIRDialect::initialize() {
  // Register types
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/AFIR/AFIRTypes.cpp.inc"
      >();

  // Register attributes
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/AFIR/AFIRAttrs.cpp.inc"
      >();

  // Register operations
  addOperations<
#define GET_OP_LIST
#include "Dialect/AFIR/AFIROps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type Parsing and Printing
//===----------------------------------------------------------------------===//

Type AFIRDialect::parseType(DialectAsmParser &parser) const {
  // Currently AFIR uses standard MLIR types, no custom types yet
  // When custom types are added, use generated parser:
  // return generatedTypeParser(parser);
  return Type();
}

void AFIRDialect::printType(Type type, DialectAsmPrinter &printer) const {
  // Currently AFIR uses standard MLIR types, no custom types yet
  // When custom types are added, use generated printer:
  // (void)generatedTypePrinter(type, printer);
}

//===----------------------------------------------------------------------===//
// Attribute Parsing and Printing
//===----------------------------------------------------------------------===//

Attribute AFIRDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  // Currently AFIR only has enum attributes which are parsed automatically
  // When custom attributes are added, implement parsing logic here
  return Attribute();
}

void AFIRDialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const {
  // Currently AFIR only has enum attributes which are printed automatically
  // When custom attributes are added, implement printing logic here
}