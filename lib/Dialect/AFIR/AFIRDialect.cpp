//===- AFIRDialect.cpp - AFIR dialect implementation ------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIRDialect.h"
#include "Dialect/AFIR/AFIROps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::afir;

//===----------------------------------------------------------------------===//
// AFIR dialect initialization
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIRDialect.cpp.inc"

void AFIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/AFIR/AFIROps.cpp.inc"
      >();
}

Attribute AFIRDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  return mlir::Attribute();
}

void AFIRDialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const {
  // 空实现
}

Type AFIRDialect::parseType(DialectAsmParser &parser) const {
  return mlir::Type();
}

void AFIRDialect::printType(Type type, DialectAsmPrinter &printer) const {
  // 空实现
}