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
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

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
    "int32_t":$alloc_type,
    "int32_t":$position,
    "int32_t":$hardware,
    ArrayRefParameter<"int64_t">:$buf_ids,
    "StringAttr":$name,
    "int64_t":$reuse_id
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
    "int64_t":$depth,
    "int64_t":$buf_num,
    "StringAttr":$name
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
    "int64_t":$id,
    "StringAttr":$name
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

// Maps from: ge.proto.MemOptAttrDef
def AFIR_MemOptAttr : AFIR_Attr<"MemOptAttr", "mem_opt"> {
  let summary = "Memory optimization attribute";
  let description = [{
    Defines memory optimization attributes including reuse ID, reference tensor,
    and merge scope for memory optimization.
  }];

  let parameters = (ins
    "int64_t":$reuse_id,
    "int64_t":$ref_tensor,
    "int64_t":$merge_scope
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
    "int64_t":$dtype,
    ArrayRefParameter<"int64_t">:$axis_ids,
    ArrayRefParameter<"Attribute", "array of expression strings">:$repeats,
    ArrayRefParameter<"Attribute", "array of expression strings">:$strides,
    ArrayRefParameter<"int64_t">:$vectorized_axis,
    ArrayRefParameter<"Attribute", "array of vectorized stride expressions">:$vectorized_strides,
    OptionalParameter<"AFIR_MemAttr">:$mem,
    OptionalParameter<"AFIR_MemQueueAttr">:$que,
    OptionalParameter<"AFIR_MemBufAttr">:$buf,
    OptionalParameter<"AFIR_MemOptAttr">:$opt
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
    "int64_t":$id,
    "StringAttr":$name,
    "int32_t":$axis_type,
    "bool":$bind_block,
    "StringAttr":$size,  // expression string
    "StringAttr":$align,
    ArrayRefParameter<"int64_t">:$from,
    "int64_t":$split_pair_other_id,
    "bool":$allow_oversize_axis,
    "bool":$allow_unaligned_tail
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
    "int64_t":$exec_order,
    ArrayRefParameter<"int64_t">:$axis,
    "int64_t":$loop_axis,
    "int32_t":$exec_condition
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
    "int32_t":$type,
    "int32_t":$compute_type,
    "int32_t":$unit
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
    "int64_t":$life_time_axis_id
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
    "AFIR_TmpBufDesc":$buf_desc,
    "AFIR_MemAttr":$mem,
    "int64_t":$id
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
    OptionalParameter<"AFIR_SchedInfo">:$sched,
    OptionalParameter<"AFIR_ApiInfo">:$api,
    "DictionaryAttr":$ir_attr_def,  // maps to AscIrAttrDef.attr
    ArrayRefParameter<"Attribute", "array of TmpBufferGroup">:$tmp_buffers
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
    "int64_t":$tiling_key,
    ArrayRefParameter<"Attribute", "array of AxisAttr">:$axis,
    "int64_t":$type,
    ArrayRefParameter<"Attribute", "array of size variable strings">:$size_var
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Input/Output Connection Attributes
//===----------------------------------------------------------------------===//

// Maps from: ascendc_ir.proto.AscInputSourceDef
def AFIR_AscInputSource : AFIR_Attr<"AscInputSource", "input_src"> {
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
    "AFIR_AscTensorAttrGroups":$attr
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
    ArrayRefParameter<"Attribute", "array of input name strings">:$input_names,
    ArrayRefParameter<"Attribute", "array of output name strings">:$output_names,
    ArrayRefParameter<"int64_t">:$input_ir_type,
    ArrayRefParameter<"int64_t">:$output_ir_type,
    "StringAttr":$type,
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
    ArrayRefParameter<"Attribute", "array of AscInputSource">:$input_src,
    ArrayRefParameter<"Attribute", "array of AscTensor">:$outputs,
    "AFIR_AscNodeAttrGroups":$attr,
    "AFIR_IrDef":$ir_def
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
    "AFIR_AscGraphAttrGroups":$asc_graph_attr,
    ArrayRefParameter<"Attribute", "array of AscNode">:$asc_node,
    "StringAttr":$graph_name
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // AFIR_ATTRS
