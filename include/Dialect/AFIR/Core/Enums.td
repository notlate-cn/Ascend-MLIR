//===- AFIREnums.td - AFIR dialect enum definitions -------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines enumerations for the AFIR dialect. All enum definitions
// should be placed here to avoid redefinition issues.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_ENUMS
#define AFIR_ENUMS

include "Dialect/AFIR/Core/Base.td"
include "mlir/IR/EnumAttr.td"


//===----------------------------------------------------------------------===//
// Position Enumeration
//===----------------------------------------------------------------------===//

def AFIR_PositionAttr : I32EnumAttr<"Position",
    "Memory position",
    [
      I32EnumAttrCase<"GM", 0, "gm">,
      I32EnumAttrCase<"VECTOR_IN", 1, "vector_in">,
      I32EnumAttrCase<"VECTOR_OUT", 2, "vector_out">,
      I32EnumAttrCase<"VECTOR_CALC", 3, "vector_calc">,
      I32EnumAttrCase<"L1", 4, "l1">,
      I32EnumAttrCase<"L2", 5, "l2">,
      I32EnumAttrCase<"L0A", 6, "l0a">,
      I32EnumAttrCase<"L0B", 7, "l0b">,
      I32EnumAttrCase<"L0C", 8, "l0c">
    ]> {
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

//===----------------------------------------------------------------------===//
// Axis Type Enumeration
//===----------------------------------------------------------------------===//

def AFIR_AxisTypeAttr : I32EnumAttr<"AxisType",
    "Axis type for computational loops",
    [
      I32EnumAttrCase<"Original", 0>,
      I32EnumAttrCase<"BlockOuter", 1>,
      I32EnumAttrCase<"BlockInner", 2>,
      I32EnumAttrCase<"TileOuter", 3>,
      I32EnumAttrCase<"TileInner", 4>,
      I32EnumAttrCase<"Merged", 5>,
      I32EnumAttrCase<"Invalid", 6>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

//===----------------------------------------------------------------------===//
// AscGraphType Enumeration
//===----------------------------------------------------------------------===//

def AFIR_AscGraphTypeAttr : I32EnumAttr<"AscGraphType",
    "Ascend graph type",
    [
      I32EnumAttrCase<"COMPUTE", 0>,
      I32EnumAttrCase<"Invalid", 1>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

#endif // AFIR_ENUMS