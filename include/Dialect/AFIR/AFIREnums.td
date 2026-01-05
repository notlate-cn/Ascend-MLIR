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
// Allocation Type Enumeration
//===----------------------------------------------------------------------===//

def AFIR_ALLOC_GLOBAL : I32EnumAttrCase<"GLOBAL", 0>;
def AFIR_ALLOC_L1     : I32EnumAttrCase<"L1", 1>;
def AFIR_ALLOC_L2     : I32EnumAttrCase<"L2", 2>;
def AFIR_ALLOC_QBUF   : I32EnumAttrCase<"QBUF", 3>;
def AFIR_ALLOC_TBUF   : I32EnumAttrCase<"TBUF", 4>;

def AFIR_AllocTypeEnum : I32EnumAttr<"AllocType",
    "Memory allocation type",
    [AFIR_ALLOC_GLOBAL, AFIR_ALLOC_L1, AFIR_ALLOC_L2, AFIR_ALLOC_QBUF, AFIR_ALLOC_TBUF]> {
  let cppNamespace = "::mlir::afir";
}

def AFIR_AllocTypeAttr : EnumAttr<AFIR_Dialect, AFIR_AllocTypeEnum, "alloc_type">;

//===----------------------------------------------------------------------===//
// Position Enumeration
//===----------------------------------------------------------------------===//

def AFIR_POS_VEC_IN   : I32EnumAttrCase<"VEC_IN", 0>;
def AFIR_POS_VEC_OUT  : I32EnumAttrCase<"VEC_OUT", 1>;
def AFIR_POS_VEC_CALC : I32EnumAttrCase<"VEC_CALC", 2>;
def AFIR_POS_L1       : I32EnumAttrCase<"L1", 3>;
def AFIR_POS_L0_A     : I32EnumAttrCase<"L0_A", 4>;
def AFIR_POS_L0_B     : I32EnumAttrCase<"L0_B", 5>;
def AFIR_POS_L0_C     : I32EnumAttrCase<"L0_C", 6>;
def AFIR_POS_GM       : I32EnumAttrCase<"GM", 7>;

def AFIR_PositionEnum : I32EnumAttr<"Position",
    "Memory position",
    [AFIR_POS_VEC_IN, AFIR_POS_VEC_OUT, AFIR_POS_VEC_CALC,
     AFIR_POS_L1, AFIR_POS_L0_A, AFIR_POS_L0_B, AFIR_POS_L0_C, AFIR_POS_GM]> {
  let cppNamespace = "::mlir::afir";
}

def AFIR_PositionAttr : EnumAttr<AFIR_Dialect, AFIR_PositionEnum, "position">;

//===----------------------------------------------------------------------===//
// Hardware Enumeration
//===----------------------------------------------------------------------===//

def AFIR_HW_GM : I32EnumAttrCase<"GM", 0>;
def AFIR_HW_UB : I32EnumAttrCase<"UB", 1>;

def AFIR_HardwareEnum : I32EnumAttr<"Hardware",
    "Hardware target",
    [AFIR_HW_GM, AFIR_HW_UB]> {
  let cppNamespace = "::mlir::afir";
}

def AFIR_HardwareAttr : EnumAttr<AFIR_Dialect, AFIR_HardwareEnum, "hardware">;

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
  let cppNamespace = "::mlir::afir";
}

def AFIR_AxisTypeAttr : EnumAttr<AFIR_Dialect, AFIR_AxisTypeEnum, "axis_type">;

//===----------------------------------------------------------------------===//
// Execute Condition Enumeration
//===----------------------------------------------------------------------===//

def AFIR_EXEC_NOCACHE                                 : I32EnumAttrCase<"NoCache", 0>;
def AFIR_EXEC_CACHE_BLOCK_SPLIT_FUSED_BROADCAST_AXIS  : I32EnumAttrCase<"CacheBlockSplitFusedBroadcastAxis", 1>;
def AFIR_EXEC_CACHE_BLOCK_SPLIT_ORIGIN_BROADCAST_AXIS : I32EnumAttrCase<"CacheBlockSplitOriginBroadcastAxis", 2>;
def AFIR_EXEC_CONDITION_INVALID                       : I32EnumAttrCase<"ConditionInvalid", 3>;

def AFIR_ExecuteConditionEnum : I32EnumAttr<"ExecuteCondition",
    "Execute condition for node execution",
    [AFIR_EXEC_NOCACHE, AFIR_EXEC_CACHE_BLOCK_SPLIT_FUSED_BROADCAST_AXIS,
     AFIR_EXEC_CACHE_BLOCK_SPLIT_ORIGIN_BROADCAST_AXIS, AFIR_EXEC_CONDITION_INVALID]> {
  let cppNamespace = "::mlir::afir";
}

def AFIR_ExecuteConditionAttr : EnumAttr<AFIR_Dialect, AFIR_ExecuteConditionEnum, "exec_condition">;

//===----------------------------------------------------------------------===//
// API Type Enumeration
//===----------------------------------------------------------------------===//

def AFIR_API_BUFFER  : I32EnumAttrCase<"Buffer", 0>;
def AFIR_API_COMPUTE : I32EnumAttrCase<"Compute", 1>;
def AFIR_API_INVALID : I32EnumAttrCase<"Invalid", 2>;

def AFIR_ApiTypeEnum : I32EnumAttr<"ApiType",
    "API type for node execution",
    [AFIR_API_BUFFER, AFIR_API_COMPUTE, AFIR_API_INVALID]> {
  let cppNamespace = "::mlir::afir";
}

def AFIR_ApiTypeAttr : EnumAttr<AFIR_Dialect, AFIR_ApiTypeEnum, "type">;

//===----------------------------------------------------------------------===//
// Compute Unit Enumeration
//===----------------------------------------------------------------------===//

def AFIR_UNIT_NONE    : I32EnumAttrCase<"None", 0>;
def AFIR_UNIT_MTE1    : I32EnumAttrCase<"MTE1", 1>;
def AFIR_UNIT_MTE2    : I32EnumAttrCase<"MTE2", 2>;
def AFIR_UNIT_MTE3    : I32EnumAttrCase<"MTE3", 3>;
def AFIR_UNIT_SCALAR : I32EnumAttrCase<"Scalar", 4>;
def AFIR_UNIT_VECTOR : I32EnumAttrCase<"Vector", 5>;
def AFIR_UNIT_CUBE   : I32EnumAttrCase<"Cube", 6>;
def AFIR_UNIT_INVALID: I32EnumAttrCase<"Invalid", 7>;

def AFIR_ComputeUnitEnum : I32EnumAttr<"ComputeUnit",
    "Compute unit for node execution",
    [AFIR_UNIT_NONE, AFIR_UNIT_MTE1, AFIR_UNIT_MTE2, AFIR_UNIT_MTE3,
     AFIR_UNIT_SCALAR, AFIR_UNIT_VECTOR, AFIR_UNIT_CUBE, AFIR_UNIT_INVALID]> {
  let cppNamespace = "::mlir::afir";
}

def AFIR_ComputeUnitAttr : EnumAttr<AFIR_Dialect, AFIR_ComputeUnitEnum, "unit">;

//===----------------------------------------------------------------------===//
// Compute Type Enumeration
//===----------------------------------------------------------------------===//

def AFIR_COMPUTE_LOAD         : I32EnumAttrCase<"Load", 0>;
def AFIR_COMPUTE_STORE        : I32EnumAttrCase<"Store", 1>;
def AFIR_COMPUTE_REDUCE_STORE : I32EnumAttrCase<"ReduceStore", 2>;
def AFIR_COMPUTE_ELEWISE      : I32EnumAttrCase<"Elewise", 3>;
def AFIR_COMPUTE_BROADCAST    : I32EnumAttrCase<"Broadcast", 4>;
def AFIR_COMPUTE_REDUCE       : I32EnumAttrCase<"Reduce", 5>;
def AFIR_COMPUTE_TRANSPOSE    : I32EnumAttrCase<"Transpose", 6>;
def AFIR_COMPUTE_CONCAT       : I32EnumAttrCase<"Concat", 7>;
def AFIR_COMPUTE_GATHER       : I32EnumAttrCase<"Gather", 8>;
def AFIR_COMPUTE_CUBE         : I32EnumAttrCase<"Cube", 9>;
def AFIR_COMPUTE_SPLIT        : I32EnumAttrCase<"Split", 10>;
def AFIR_COMPUTE_INVALID      : I32EnumAttrCase<"Invalid", 11>;

def AFIR_ComputeTypeEnum : I32EnumAttr<"ComputeType",
    "Compute type for node execution",
    [AFIR_COMPUTE_LOAD, AFIR_COMPUTE_STORE, AFIR_COMPUTE_REDUCE_STORE,
     AFIR_COMPUTE_ELEWISE, AFIR_COMPUTE_BROADCAST, AFIR_COMPUTE_REDUCE,
     AFIR_COMPUTE_TRANSPOSE, AFIR_COMPUTE_CONCAT, AFIR_COMPUTE_GATHER,
     AFIR_COMPUTE_CUBE, AFIR_COMPUTE_SPLIT, AFIR_COMPUTE_INVALID]> {
  let cppNamespace = "::mlir::afir";
}

def AFIR_ComputeTypeAttr : EnumAttr<AFIR_Dialect, AFIR_ComputeTypeEnum, "compute_type">;

//===----------------------------------------------------------------------===//
// AscGraph Type Enumeration
//===----------------------------------------------------------------------===//

def AFIR_GRAPH_HINT : I32EnumAttrCase<"HintGraph", 0>;
def AFIR_GRAPH_IMPL : I32EnumAttrCase<"ImplGraph", 1>;

def AFIR_AscGraphTypeEnum : I32EnumAttr<"AscGraphType",
    "Ascend graph type",
    [AFIR_GRAPH_HINT, AFIR_GRAPH_IMPL]> {
  let cppNamespace = "::mlir::afir";
}

def AFIR_AscGraphTypeAttr : EnumAttr<AFIR_Dialect, AFIR_AscGraphTypeEnum, "type">;

#endif // AFIR_ENUMS
