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

//===----------------------------------------------------------------------===//
// Graph-Level Attributes (AscGraph Mapping)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Memory Management Attributes
//===----------------------------------------------------------------------===//

// Maps from: ge.proto.MemAttrDef
def AFIR_Mem : AFIR_Attr<"Mem", "mem"> {
  let summary = "Memory attribute for tensor allocation";
  let description = [{
    Defines memory allocation attributes for tensors, including allocation type,
    position, hardware target, and buffer management information.
  }];

  let parameters = (ins
    "int64_t":$tensor_id,
    "AllocTypeAttr":$alloc_type, // global l1 l2 qbuf tbuf
    "PositionAttr":$position, // gm vector_in vector_out vector_calc...
    "HardwareAttr":$hardware, // gm ub
    DefaultValuedParameter<"int64_t", "-1">:$reuse_id
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

// Maps from: ge.proto.MemQueueAttrDef
def AFIR_MemQueue : AFIR_Attr<"MemQueue", "mem_queue"> {
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
def AFIR_MemBuf : AFIR_Attr<"MemBuf", "mem_buf"> {
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
    "DataTypeAttr":$dtype,
    ArrayRefParameter<"int64_t">:$axis_ids,
    ArrayRefParameter<"Attribute", "array of expression strings">:$repeats,
    ArrayRefParameter<"Attribute", "array of expression strings">:$strides,
    ArrayRefParameter<"int64_t">:$vectorized_axis,
    ArrayRefParameter<"Attribute", "array of vectorized stride expressions">:$vectorized_strides,
    OptionalParameter<"MemAttr">:$mem,
    OptionalParameter<"MemQueueAttr">:$que,
    OptionalParameter<"MemBufAttr">:$buf
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Axis and Loop Attributes
//===----------------------------------------------------------------------===//

// Maps from: ge.proto.AxisDef
def AFIR_Axis : AFIR_Attr<"Axis", "axis"> {
  let summary = "Axis definition for computational loops";
  let description = [{
    Defines an axis for loop-based computation including axis type, size,
    alignment, binding information, and optimization flags.
  }];

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "-1">:$id,
    "StringAttr":$name,
    "AxisTypeAttr":$axis_type,
    DefaultValuedParameter<"bool", "false">:$bind_block,
    "Attribute":$size,  // expression string
    OptionalParameter<"StringAttr">:$align,
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
    "ExecuteConditionAttr":$exec_condition
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
    "ApiTypeAttr":$type,
    "ComputeTypeAttr":$compute_type,
    "ComputeUnitAttr":$unit
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
    "MemAttr":$mem,
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
    ArrayRefParameter<"TmpBufferGroupAttr", "array of TmpBufferDesc">:$tmp_buffers
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
    ArrayRefParameter<"AxisAttr", "array of Axis">:$axis,
    "AscGraphTypeAttr":$type,
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
