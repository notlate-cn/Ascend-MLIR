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

include "Dialect/AFIR/AFIRBase.td"
include "mlir/IR/EnumAttr.td"


//===----------------------------------------------------------------------===//
// Position Enumeration
//===----------------------------------------------------------------------===//

def AFIR_POS_GM          : I32EnumAttrCase<"GM", 0>;
def AFIR_POS_VECTOR_IN   : I32EnumAttrCase<"VECTOR_IN", 1>;
def AFIR_POS_VECTOR_OUT  : I32EnumAttrCase<"VECTOR_OUT", 2>;
def AFIR_POS_VECTOR_CALC : I32EnumAttrCase<"VECTOR_CALC", 3>;
def AFIR_POS_L1 : I32EnumAttrCase<"L1", 4>;
def AFIR_POS_L2 : I32EnumAttrCase<"L2", 5>;
def AFIR_POS_L0A : I32EnumAttrCase<"L0A", 6>;
def AFIR_POS_L0B : I32EnumAttrCase<"L0B", 7>;
def AFIR_POS_L0C : I32EnumAttrCase<"L0C", 8>;

def AFIR_PositionEnum : I32EnumAttr<"Position",
    "Memory position",
    [AFIR_POS_GM, AFIR_POS_VECTOR_IN, AFIR_POS_VECTOR_OUT, AFIR_POS_VECTOR_CALC, AFIR_POS_L1, AFIR_POS_L2, AFIR_POS_L0A, AFIR_POS_L0B, AFIR_POS_L0C]> {
  let cppNamespace = "::mlir::afir";
}

//===----------------------------------------------------------------------===//
// Axis Type Enumeration
//===----------------------------------------------------------------------===//

def AFIR_AXIS_ORIGINAL    : I32EnumAttrCase<"Original", 0>;
def AFIR_AXIS_BLOCK_OUTER : I32EnumAttrCase<"BlockOuter", 1>;
def AFIR_AXIS_BLOCK_INNER : I32EnumAttrCase<"BlockInner", 2>;
def AFIR_AXIS_TILE_OUTER  : I32EnumAttrCase<"TileOuter", 3>;
def AFIR_AXIS_TILE_INNER  : I32EnumAttrCase<"TileInner", 4>;
def AFIR_AXIS_MERGED      : I32EnumAttrCase<"Merged", 5>;
def AFIR_AXIS_INVALID     : I32EnumAttrCase<"Invalid", 6>;

def AFIR_AxisTypeEnum : I32EnumAttr<"AxisType",
    "Axis type for computational loops",
    [AFIR_AXIS_ORIGINAL, AFIR_AXIS_BLOCK_OUTER, AFIR_AXIS_BLOCK_INNER,
     AFIR_AXIS_TILE_OUTER, AFIR_AXIS_TILE_INNER, AFIR_AXIS_MERGED,
     AFIR_AXIS_INVALID]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
}

def AFIR_AxisTypeAttr : EnumAttr<AFIR_Dialect, AFIR_AxisTypeEnum, "axis_type">;

//===----------------------------------------------------------------------===//
// AscGraphType Enumeration
//===----------------------------------------------------------------------===//

def AFIR_GRAPH_COMPUTE : I32EnumAttrCase<"COMPUTE", 0>;
def AFIR_GRAPH_INVALID : I32EnumAttrCase<"Invalid", 1>;

def AFIR_AscGraphTypeEnum : I32EnumAttr<"AscGraphType",
    "Ascend graph type",
    [AFIR_GRAPH_COMPUTE, AFIR_GRAPH_INVALID]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
}

def AFIR_AscGraphTypeAttr : EnumAttr<AFIR_Dialect, AFIR_AscGraphTypeEnum, "graph_type">;

#endif // AFIR_ENUMS
