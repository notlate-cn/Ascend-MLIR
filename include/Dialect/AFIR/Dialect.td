//===- AFIRDialect.td - AFIR dialect definition ------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the AFIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_DIALECT
#define AFIR_DIALECT

include "mlir/IR/OpBase.td"

def AFIR_Dialect : Dialect {
  let name = "afir";
  let summary = "Ascend Frontend IR Dialect";
  let description = [{
    AFIR (Ascend Frontend IR) is a high-level dialect designed for
    representing computational operations targeting Ascend hardware.
    It provides a bridge between high-level ML frameworks and lower-level
    hardware-specific dialects like ASC-IR.

    Features:
    - High-level tensor operations
    - Type-safe representations
    - Hardware-agnostic abstractions
    - Extensible attribute system
    - 1:1 mapping with AscGraph protobuf definitions
  }];

  let cppNamespace = "::mlir::afir";

  // We provide custom type and attribute parsers/printers
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;

  // Enable properties for operations
  let usePropertiesForAttributes = 0;

  // Extra class declarations for the dialect
  let extraClassDeclaration = [{
    // Type parsing and printing
    Type parseType(DialectAsmParser &parser) const override;
    void printType(Type type, DialectAsmPrinter &printer) const override;

    // Attribute parsing and printing
    Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
    void printAttribute(Attribute attr, DialectAsmPrinter &printer) const override;
  }];
}

#endif // AFIR_DIALECT