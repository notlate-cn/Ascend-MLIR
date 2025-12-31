//===- AFIRBase.td - AFIR dialect base definitions --------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the base structures for the AFIR dialect, including
// the dialect definition, common base classes, and fundamental enumerations.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_BASE
#define AFIR_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// AFIR Dialect Definition
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Base AFIR Operation Definition
//===----------------------------------------------------------------------===//

class AFIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<AFIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Base AFIR Type Definition
//===----------------------------------------------------------------------===//

class AFIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AFIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Base AFIR Attribute Definition
//===----------------------------------------------------------------------===//

class AFIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<AFIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // AFIR_BASE
