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
include "mlir/IR/BuiltinTypeInterfaces.td"

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

def AFIR_Tensor : AFIR_Type<"AFIRTensor", "tensor", [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "AFIR tensor type with memory attributes";
  let description = [{
    Represents a tensor with additional memory attributes including tensor ID,
    position information, depth, and double-buffer flag.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "int64_t":$tensorId,
    "PositionInfoAttr":$position
  );

  let assemblyFormat = "`<` $shape `,` $elementType `,` $tensorId `,` $position `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape, "Type":$elementType, "int64_t":$tensorId,
      "PositionInfoAttr":$position)>
  ];
}

//===----------------------------------------------------------------------===//
// Base AFIR Attribute Definition
//===----------------------------------------------------------------------===//

class AFIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<AFIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Position Attribute Definition
//===----------------------------------------------------------------------===//

def AFIR_PositionInfoAttr : AFIR_Attr<"PositionInfo", "pos"> {
  let summary = "Position information with id, type, depth, and double-buffer flag";
  let description = [{
    Represents position information including:
    - rid: Position resource ID
    - position: Position type (VEC_IN, VEC_OUT, VEC_CALC, L1, L0_A, L0_B, L0_C, GM)
    - depth: TQueue's depth (only shown for VEC_IN/VEC_OUT)
    - buf_num: TQueue's buf_num (only shown for VEC_IN/VEC_OUT)
    - db: Double-buffer flag (only shown for VEC_IN/VEC_OUT)
  }];

  let parameters = (ins
    "int64_t":$rid,
    "Position":$position,
    DefaultValuedParameter<"int64_t", "2">:$depth,
    DefaultValuedParameter<"int64_t", "-1">:$buf_num,
    DefaultValuedParameter<"bool", "false">:$db
  );

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

// A shaped type (AFIRTensor or standard tensor) with element types from the given list
class ShapedTypeOf<list<Type> allowedTypes, string summary = "shaped type"> :
    Type<Or<[
      CPred<"::llvm::isa<mlir::afir::AFIRTensorType>($_self)">,
      CPred<"::llvm::isa<::mlir::ShapedType>($_self)">
    ]>,
         summary # " of " # AnyTypeOf<allowedTypes>.summary # " values">;

// Common shaped type constraints with specific element types
def ShapedTypeOf_F16_F32_I16_I32 : ShapedTypeOf<[F16, F32, I16, I32], "shaped type">;

#endif // AFIR_BASE
