//===- AFIRAttrs.td - AFIR dialect attribute definitions --*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines attributes for the AFIR dialect, including both
// operation-level attributes and graph-level attributes that map 1:1
// with AscGraph protobuf definitions.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_ATTRS
#define AFIR_ATTRS

include "Dialect/AFIR/AFIRBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// AFIR Attribute Definitions
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

//===----------------------------------------------------------------------===//
// Graph-Level Attributes (AscGraph Mapping)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Memory Management Attributes
//===----------------------------------------------------------------------===//

// Maps from: ge.proto.MemAttrDef
def AFIR_MemAttr : AFIR_Attr<"MemAttr", "mem"> {
  let summary = "Memory attribute for tensor allocation";
  let description = [{
    Defines memory allocation attributes for tensors, including allocation type,
    position, hardware target, and buffer management information.
  }];

  let parameters = (ins
    "int64_t":$tensor_id,
    "AFIR_AllocTypeAttr":$alloc_type, // global l1 l2 qbuf tbuf
    "AFIR_PositionAttr":$position, // gm vector_in vector_out vector_calc...
    "AFIR_HardwareAttr":$hardware, // gm ub
    DefaultValuedParameter<"int64_t", "-1">:$reuse_id
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

// Maps from: ge.proto.MemQueueAttrDef
def AFIR_MemQueueAttr : AFIR_Attr<"MemQueueAttr", "mem_queue"> {
  let summary = "Memory queue attribute";
  let description = [{
    Defines memory queue attributes including queue ID, depth, buffer count, and name.
  }];

  let parameters = (ins
    "int64_t":$id,
    DefaultValuedParameter<"int64_t", "2">:$depth, //queue最大深度，编译期间设置，建议2
    "int64_t":$buf_num // 是否是double_buffer
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

// Maps from: ge.proto.MemBufAttrDef
def AFIR_MemBufAttr : AFIR_Attr<"MemBufAttr", "mem_buf"> {
  let summary = "Memory buffer attribute";
  let description = [{
    Defines memory buffer attributes including buffer ID and name.
  }];

  let parameters = (ins
    "int64_t":$id
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Tensor Attributes
//===----------------------------------------------------------------------===//

// Maps from: ge.proto.AscTensorAttrGroupsDef
def AFIR_AscTensorAttrGroups : AFIR_Attr<"AscTensorAttrGroups", "asc_tensor"> {
  let summary = "Ascend tensor attribute groups";
  let description = [{
    Comprehensive tensor attributes including data type, axis information,
    memory layout (repeats, strides), vectorization info, and memory management.
  }];

  let parameters = (ins
    "AFIR_DataTypeAttr":$dtype,
    ArrayRefParameter<"int64_t">:$axis_ids,
    ArrayRefParameter<"Attribute", "array of expression strings">:$repeats,
    ArrayRefParameter<"Attribute", "array of expression strings">:$strides,
    ArrayRefParameter<"int64_t">:$vectorized_axis,
    ArrayRefParameter<"Attribute", "array of vectorized stride expressions">:$vectorized_strides,
    OptionalParameter<"MemAttrAttr">:$mem,
    OptionalParameter<"MemQueueAttrAttr">:$que,
    OptionalParameter<"MemBufAttrAttr">:$buf
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Axis and Loop Attributes
//===----------------------------------------------------------------------===//

// Maps from: ge.proto.AxisDef
def AFIR_AxisAttr : AFIR_Attr<"AxisAttr", "axis"> {
  let summary = "Axis definition for computational loops";
  let description = [{
    Defines an axis for loop-based computation including axis type, size,
    alignment, binding information, and optimization flags.
  }];

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "-1">:$id,
    "StringAttr":$name,
    "AFIR_AxisTypeAttr":$axis_type,
    DefaultValuedParameter<"bool", "false">:$bind_block,
    "Attribute":$size,  // expression string
    DefaultValuedParameter<"StringAttr", "\"-1\"">:$align,
    ArrayRefParameter<"int64_t">:$from,
    DefaultValuedParameter<"int64_t", "-1">:$split_pair_other_id
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Scheduling and Execution Attributes
//===----------------------------------------------------------------------===//

// Maps from: ge.proto.SchedInfoDef
def AFIR_SchedInfo : AFIR_Attr<"SchedInfo", "sched"> {
  let summary = "Scheduling information for node execution";
  let description = [{
    Defines scheduling information including execution order, axis mapping,
    loop axis, and execution conditions.
  }];

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "-1">:$exec_order, // 暂未使用
    ArrayRefParameter<"int64_t">:$axis,
    DefaultValuedParameter<"int64_t", "-1">:$loop_axis,
    "AFIR_ExecuteConditionAttr":$exec_condition
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

// Maps from: ge.proto.ApiInfoDef
def AFIR_ApiInfo : AFIR_Attr<"ApiInfo", "api"> {
  let summary = "API information for node execution";
  let description = [{
    Defines API type, compute type, and execution unit for the node.
  }];

  let parameters = (ins
    "AFIR_ApiTypeAttr":$type,
    "AFIR_ComputeTypeAttr":$compute_type,
    "AFIR_ComputeUnitAttr":$unit
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Temporary Buffer Attributes
//===----------------------------------------------------------------------===//

// Maps from: ge.proto.TmpBufDescDef
def AFIR_TmpBufDesc : AFIR_Attr<"TmpBufDesc", "tmp_buf_desc"> {
  let summary = "Temporary buffer description";
  let description = [{
    Describes a temporary buffer including its size expression and lifetime axis.
  }];

  let parameters = (ins
    "StringAttr":$size,  // expression string
    DefaultValuedParameter<"int64_t", "-1">:$life_time_axis_id
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

// Maps from: ge.proto.TmpBufferGroupDef
def AFIR_TmpBufferGroup : AFIR_Attr<"TmpBufferGroup", "tmp_buffer"> {
  let summary = "Temporary buffer group definition";
  let description = [{
    Groups temporary buffer description with memory attributes and ID.
  }];

  let parameters = (ins
    "TmpBufDescAttr":$buf_desc,
    "MemAttrAttr":$mem,
    DefaultValuedParameter<"int64_t", "-1">:$id
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Node Attributes
//===----------------------------------------------------------------------===//

// Maps from: ge.proto.AscNodeAttrGroupsDef
def AFIR_AscNodeAttrGroups : AFIR_Attr<"AscNodeAttrGroups", "asc_node"> {
  let summary = "Ascend node attribute groups";
  let description = [{
    Comprehensive node attributes including name, type, scheduling info,
    API info, IR attributes, and temporary buffers.
  }];

  let parameters = (ins
    "StringAttr":$name,
    "StringAttr":$type,
    OptionalParameter<"SchedInfoAttr">:$sched,
    OptionalParameter<"ApiInfoAttr">:$api,
    "DictionaryAttr":$ir_attr_def,  // maps to AscIrAttrDef.attr
    ArrayRefParameter<"TmpBufferGroupAttr", "array of TmpBufferGroup">:$tmp_buffers
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Graph Attributes
//===----------------------------------------------------------------------===//

// Maps from: ge.proto.AscGraphAttrGroupsDef
def AFIR_AscGraphAttrGroups : AFIR_Attr<"AscGraphAttrGroups", "asc_graph"> {
  let summary = "Ascend graph attribute groups";
  let description = [{
    Graph-level attributes including tiling key, axis definitions,
    graph type, and size variables.
  }];

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "-1">:$tiling_key,
    ArrayRefParameter<"AxisAttrAttr", "array of AxisAttr">:$axis,
    "AFIR_AscGraphTypeAttr":$type,
    ArrayRefParameter<"Attribute", "array of size variable strings">:$size_var
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Input/Output Connection Attributes
//===----------------------------------------------------------------------===//

// Maps from: ascendc_ir.proto.AscInputSourceDef
def AFIR_AscInputSource : AFIR_Attr<"AscInputSource", "input_src"> { // 用法未知
  let summary = "Input source reference";
  let description = [{
    References an output of another node as input, specifying the source
    node name and output index.
  }];

  let parameters = (ins
    "StringAttr":$src_node_name,
    "int32_t":$src_out_index
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Tensor Definition Attributes
//===----------------------------------------------------------------------===//

// Maps from: ascendc_ir.proto.AscTensorDef
def AFIR_AscTensor : AFIR_Attr<"AscTensor", "tensor_def"> {
  let summary = "Ascend tensor definition";
  let description = [{
    Tensor definition containing comprehensive attribute groups.
  }];

  let parameters = (ins
    "AscTensorAttrGroupsAttr":$attr
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// IR Definition Attributes
//===----------------------------------------------------------------------===//

// Maps from: ascendc_ir.proto.IrDef
def AFIR_IrDef : AFIR_Attr<"IrDef", "ir_def"> {
  let summary = "IR definition for node";
  let description = [{
    Defines the IR-level interface of a node including input/output names,
    types, and counts.
  }];

  let parameters = (ins
    ArrayRefParameter<"StringAttr", "array of input name strings">:$input_names,
    ArrayRefParameter<"StringAttr", "array of output name strings">:$output_names,
    ArrayRefParameter<"int64_t">:$input_ir_type,  // 待确认
    ArrayRefParameter<"int64_t">:$output_ir_type, // 待确认
    "StringAttr":$type, // 待确认
    ArrayRefParameter<"int64_t">:$input_nums,
    ArrayRefParameter<"int64_t">:$output_nums
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Node Definition Attributes
//===----------------------------------------------------------------------===//

// Maps from: ascendc_ir.proto.AscNodeDef
def AFIR_AscNode : AFIR_Attr<"AscNode", "node"> {
  let summary = "Ascend node definition";
  let description = [{
    Complete node definition including input sources, output tensors,
    node attributes, and IR definition.
  }];

  let parameters = (ins
    ArrayRefParameter<"AscInputSourceAttr", "array of AscInputSource">:$input_src,
    ArrayRefParameter<"AscTensorAttr", "array of AscTensor">:$outputs,
    "AscNodeAttrGroupsAttr":$attr,
    "IrDefAttr":$ir_def
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Graph Definition Attributes
//===----------------------------------------------------------------------===//

// Maps from: ascendc_ir.proto.AscGraphDef
def AFIR_AscGraph : AFIR_Attr<"AscGraph", "graph"> {
  let summary = "Ascend graph definition";
  let description = [{
    Top-level graph definition containing graph attributes, nodes, and graph name.
    This is the main structure representing a complete Ascend computation graph.
  }];

  let parameters = (ins
    "AscGraphAttrGroupsAttr":$asc_graph_attr,
    ArrayRefParameter<"AscNodeAttr", "array of AscNode">:$asc_node,
    "StringAttr":$graph_name
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // AFIR_ATTRS
