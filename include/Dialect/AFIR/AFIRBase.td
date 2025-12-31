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

//===----------------------------------------------------------------------===//
// DataType Enumeration (from ge.proto)
//===----------------------------------------------------------------------===//

// DataType enum cases - matches ge.proto.DataType exactly
def AFIR_DT_UNDEFINED      : I32EnumAttrCase<"DT_UNDEFINED", 0>;
def AFIR_DT_FLOAT          : I32EnumAttrCase<"DT_FLOAT", 1>;
def AFIR_DT_FLOAT16        : I32EnumAttrCase<"DT_FLOAT16", 2>;
def AFIR_DT_INT8           : I32EnumAttrCase<"DT_INT8", 3>;
def AFIR_DT_UINT8          : I32EnumAttrCase<"DT_UINT8", 4>;
def AFIR_DT_INT16          : I32EnumAttrCase<"DT_INT16", 5>;
def AFIR_DT_UINT16         : I32EnumAttrCase<"DT_UINT16", 6>;
def AFIR_DT_INT32          : I32EnumAttrCase<"DT_INT32", 7>;
def AFIR_DT_INT64          : I32EnumAttrCase<"DT_INT64", 8>;
def AFIR_DT_UINT32         : I32EnumAttrCase<"DT_UINT32", 9>;
def AFIR_DT_UINT64         : I32EnumAttrCase<"DT_UINT64", 10>;
def AFIR_DT_BOOL           : I32EnumAttrCase<"DT_BOOL", 11>;
def AFIR_DT_DOUBLE         : I32EnumAttrCase<"DT_DOUBLE", 12>;
def AFIR_DT_STRING         : I32EnumAttrCase<"DT_STRING", 13>;
def AFIR_DT_DUAL_SUB_INT8  : I32EnumAttrCase<"DT_DUAL_SUB_INT8", 14>;
def AFIR_DT_DUAL_SUB_UINT8 : I32EnumAttrCase<"DT_DUAL_SUB_UINT8", 15>;
def AFIR_DT_COMPLEX64      : I32EnumAttrCase<"DT_COMPLEX64", 16>;
def AFIR_DT_COMPLEX128     : I32EnumAttrCase<"DT_COMPLEX128", 17>;
def AFIR_DT_QINT8          : I32EnumAttrCase<"DT_QINT8", 18>;
def AFIR_DT_QINT16         : I32EnumAttrCase<"DT_QINT16", 19>;
def AFIR_DT_QINT32         : I32EnumAttrCase<"DT_QINT32", 20>;
def AFIR_DT_QUINT8         : I32EnumAttrCase<"DT_QUINT8", 21>;
def AFIR_DT_QUINT16        : I32EnumAttrCase<"DT_QUINT16", 22>;
def AFIR_DT_RESOURCE       : I32EnumAttrCase<"DT_RESOURCE", 23>;
def AFIR_DT_STRING_REF     : I32EnumAttrCase<"DT_STRING_REF", 24>;
def AFIR_DT_DUAL           : I32EnumAttrCase<"DT_DUAL", 25>;
def AFIR_DT_VARIANT        : I32EnumAttrCase<"DT_VARIANT", 26>;
def AFIR_DT_BF16           : I32EnumAttrCase<"DT_BF16", 27>;
def AFIR_DT_INT4           : I32EnumAttrCase<"DT_INT4", 28>;
def AFIR_DT_UINT1          : I32EnumAttrCase<"DT_UINT1", 29>;
def AFIR_DT_INT2           : I32EnumAttrCase<"DT_INT2", 30>;
def AFIR_DT_UINT2          : I32EnumAttrCase<"DT_UINT2", 31>;
def AFIR_DT_COMPLEX32      : I32EnumAttrCase<"DT_COMPLEX32", 32>;
def AFIR_DT_HIFLOAT8       : I32EnumAttrCase<"DT_HIFLOAT8", 33>;
def AFIR_DT_FLOAT8_E5M2    : I32EnumAttrCase<"DT_FLOAT8_E5M2", 34>;
def AFIR_DT_FLOAT8_E4M3FN  : I32EnumAttrCase<"DT_FLOAT8_E4M3FN", 35>;
def AFIR_DT_FLOAT8_E8M0    : I32EnumAttrCase<"DT_FLOAT8_E8M0", 36>;
def AFIR_DT_FLOAT6_E3M2    : I32EnumAttrCase<"DT_FLOAT6_E3M2", 37>;
def AFIR_DT_FLOAT6_E2M3    : I32EnumAttrCase<"DT_FLOAT6_E2M3", 38>;
def AFIR_DT_FLOAT4_E2M1    : I32EnumAttrCase<"DT_FLOAT4_E2M1", 39>;
def AFIR_DT_FLOAT4_E1M2    : I32EnumAttrCase<"DT_FLOAT4_E1M2", 40>;

def AFIR_DataTypeEnum : I32EnumAttr<"DataType",
    "Data type enumeration matching ge.proto.DataType",
    [
      AFIR_DT_UNDEFINED, AFIR_DT_FLOAT, AFIR_DT_FLOAT16, AFIR_DT_INT8,
      AFIR_DT_UINT8, AFIR_DT_INT16, AFIR_DT_UINT16, AFIR_DT_INT32,
      AFIR_DT_INT64, AFIR_DT_UINT32, AFIR_DT_UINT64, AFIR_DT_BOOL,
      AFIR_DT_DOUBLE, AFIR_DT_STRING, AFIR_DT_DUAL_SUB_INT8, AFIR_DT_DUAL_SUB_UINT8,
      AFIR_DT_COMPLEX64, AFIR_DT_COMPLEX128, AFIR_DT_QINT8, AFIR_DT_QINT16,
      AFIR_DT_QINT32, AFIR_DT_QUINT8, AFIR_DT_QUINT16, AFIR_DT_RESOURCE,
      AFIR_DT_STRING_REF, AFIR_DT_DUAL, AFIR_DT_VARIANT, AFIR_DT_BF16,
      AFIR_DT_INT4, AFIR_DT_UINT1, AFIR_DT_INT2, AFIR_DT_UINT2,
      AFIR_DT_COMPLEX32, AFIR_DT_HIFLOAT8, AFIR_DT_FLOAT8_E5M2, AFIR_DT_FLOAT8_E4M3FN,
      AFIR_DT_FLOAT8_E8M0, AFIR_DT_FLOAT6_E3M2, AFIR_DT_FLOAT6_E2M3, AFIR_DT_FLOAT4_E2M1,
      AFIR_DT_FLOAT4_E1M2
    ]> {
  let cppNamespace = "::mlir::afir";
}

def AFIR_DataTypeAttr : EnumAttr<AFIR_Dialect, AFIR_DataTypeEnum, "dtype">;

#endif // AFIR_BASE
