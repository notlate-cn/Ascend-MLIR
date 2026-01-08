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

include "Dialect/AFIR/Base.td"
include "mlir/IR/AttrTypeBase.td"
include "Dialect/AFIR/Enums.td"

//===----------------------------------------------------------------------===//
// Graph-Level Attributes (AscGraph Mapping)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Tensor Attributes
//===----------------------------------------------------------------------===//

// Maps from: ge.proto.AscTensorAttrGroupsDef
def AFIR_AscTensorGroups : AFIR_Attr<"AscTensorGroups", "asc_tensor"> {
  let summary = "Ascend tensor attribute groups";
  let description = [{
    Comprehensive tensor attributes including data type, axis information,
    memory layout (repeats, strides), vectorization info, and memory management.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$vectorized_axis,
    ArrayRefParameter<"Attribute", "array of vectorized stride expressions">:$vectorized_strides,
    "int64_t":$tensor_id,
    DefaultValuedParameter<"int64_t", "-1">:$reuse_id,
    EnumParameter<AFIR_PositionEnum>:$position,
    "int64_t":$position_id,
    OptionalParameter<"uint32_t">:$depth,
    OptionalParameter<"bool">:$is_double_buffer
  );

  let assemblyFormat = "`<` `vectorized_axis` `=` `[` $vectorized_axis `]` `,` `vectorized_strides` `=` `[` $vectorized_strides `]` `,` `tensor_id` `=` $tensor_id `,` `reuse_id` `=` $reuse_id `,` `position` `=` $position `,` `position_id` `=` $position_id `,` `depth` `=` $depth `,` `is_double_buffer` `=` $is_double_buffer `>`";
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
    EnumParameter<AFIR_AscGraphTypeEnum>:$type,
    ArrayRefParameter<"Attribute", "array of size variable strings">:$size_var
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Array Attributes
//===----------------------------------------------------------------------===//

def TmpBufDescArrayAttr : TypedArrayAttrBase<AFIR_TmpBufDesc, "Array of TmpBufDesc attributes">;

def AscTensorGroupsArrayAttr : TypedArrayAttrBase<AFIR_AscTensorGroups, "Array of AscTensorGroups attributes">;

#endif // AFIR_ATTRS
