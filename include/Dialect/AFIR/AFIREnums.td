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
// DataType Enumeration
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

def AFIR_POS_GM          : I32EnumAttrCase<"GM", 0>;
def AFIR_POS_VECTOR_IN   : I32EnumAttrCase<"VECTOR_IN", 1>;
def AFIR_POS_VECTOR_OUT  : I32EnumAttrCase<"VECTOR_OUT", 2>;
def AFIR_POS_VECTOR_CALC : I32EnumAttrCase<"VECTOR_CALC", 3>;

def AFIR_PositionEnum : I32EnumAttr<"Position",
    "Memory position",
    [AFIR_POS_GM, AFIR_POS_VECTOR_IN, AFIR_POS_VECTOR_OUT, AFIR_POS_VECTOR_CALC]> {
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
