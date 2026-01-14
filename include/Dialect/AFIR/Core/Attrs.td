//===- Attrs.td - AFIR attribute definitions ---------*- tablegen -*-===//
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

include "Dialect/AFIR/Core/Base.td"
include "mlir/IR/AttrTypeBase.td"
include "Dialect/AFIR/Core/Enums.td"

//===----------------------------------------------------------------------===//
// Graph-Level Attributes (AscGraph Mapping)
//===----------------------------------------------------------------------===//

// Core/Attrs.td

//===----------------------------------------------------------------------===//
// Position Attributes
//===----------------------------------------------------------------------===//
def PositionConfigAttr : AFIR_Attr<"PositionConfig", "pos_config"> {
  let summary = "Position configuration including location, depth and buffer mode";
  
  let parameters = (ins
    "::mlir::afir::Position":$position,
    DefaultValuedParameter<"uint32_t", "0">:$depth,
    DefaultValuedParameter<"bool", "false">:$is_double_buffer
  );
  
  // 子属性的 assembly format
  let assemblyFormat = "`<`"
    "$position"
    "(`,` `depth` `=` $depth^)?"
    "(`,` `is_double_buffer` `=` $is_double_buffer^)?"
    "`>`";
}

//===----------------------------------------------------------------------===//
// 主属性（使用位置配置子属性）
//===----------------------------------------------------------------------===//
def AscTensorGroupsAttr : AFIR_Attr<"AscTensorGroups", "asc_tensor"> {
  let summary = "Ascend tensor attribute groups";
  let description = [{
    Comprehensive tensor attributes including data type, axis information,
    memory layout (repeats, strides), vectorization info, and memory management.
  }];
  
  let parameters = (ins
    OptionalArrayRefParameter<"int64_t">:$vectorized_axis,
    OptionalArrayRefParameter<"Attribute", "array of vectorized stride expressions">:$vectorized_strides,
    "int64_t":$tensor_id,
    DefaultValuedParameter<"int64_t", "-1">:$reuse_id,
    "PositionConfigAttr":$position_config,
    DefaultValuedParameter<"int64_t", "-1">:$position_id
  );
  
  let assemblyFormat = "`<`"
    "(`vectorized_axis` `=` `[` $vectorized_axis^ `]` `,`"
    " `vectorized_strides` `=` `[` $vectorized_strides `]` `,`)? "
    "`tensor_id` `=` $tensor_id `,`"
    "(`reuse_id` `=` $reuse_id^ `,` )?"
    "`position` `=` $position_config"  // 打印子属性
    "(`,` `position_id` `=` $position_id^)?"
    "`>`";
}

//===----------------------------------------------------------------------===//
// Axis and Loop Attributes
//===----------------------------------------------------------------------===//

// Maps from: ge.proto.AxisDef
def AxisAttr : AFIR_Attr<"Axis", "axis"> {
  let summary = "Axis definition for computational loops";
  let description = [{
    Defines an axis for loop-based computation including axis type, size,
    alignment, binding information, and optimization flags.
  }];

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "-1">:$id,
    "StringAttr":$name,
    "::mlir::afir::AxisType":$axis_type,
    DefaultValuedParameter<"bool", "false">:$bind_block,
    "StringAttr":$size,
    OptionalParameter<"StringAttr">:$align,
    OptionalArrayRefParameter<"int64_t">:$from
  );

  let assemblyFormat = "`<`"
    "`id` `=` $id `,`"
    "`name` `=` $name `,`"
    "`axis_type` `=` $axis_type `,`"
    "(`bind_block` `=` $bind_block^ `,`)?"
    "`size` `=` $size"
    "(`,` `align` `=` $align^)?"
    "(`,` `from` `=` `[` $from^ `]`)?"
    "`>`";
}

//===----------------------------------------------------------------------===//
// Temporary Buffer Attributes
//===----------------------------------------------------------------------===//

// Maps from: ge.proto.TmpBufDescDef
def TmpBufDescAttr : AFIR_Attr<"TmpBufDesc", "tmp_buf_desc"> {
  let summary = "Temporary buffer description";
  let description = [{
    Describes a temporary buffer including its size expression and lifetime axis.
  }];

  let parameters = (ins
    "StringAttr":$size,
    DefaultValuedParameter<"int64_t", "-1">:$life_time_axis_id
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Graph Attributes
//===----------------------------------------------------------------------===//

// Maps from: ge.proto.AscGraphAttrGroupsDef
def AscGraphAttrGroupsAttr : AFIR_Attr<"AscGraphAttrGroups", "asc_graph"> {
  let summary = "Ascend graph attribute groups";
  let description = [{
    Graph-level attributes including tiling key, axis definitions,
    graph type, and size variables.
  }];

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "-1">:$tiling_key,
    ArrayRefParameter<"AxisAttr", "array of Axis">:$axes,
    "::mlir::afir::AscGraphType":$type,
    OptionalArrayRefParameter<"StringAttr", "array of size variable strings">:$size_var
  );

  let assemblyFormat = "`<`"
    "(`tiling_key` `=` $tiling_key^ `,`)?"
    "`axes` `=` `[` $axes `]` `,`"
    "`type` `=` $type"
    "(`,` `size_var` `=` `[` $size_var^ `]`)?"
    "`>`";
}

//===----------------------------------------------------------------------===//
// Attributes for Data node and Output node
//===----------------------------------------------------------------------===//

def ScalarValueAttr : AFIR_Attr<"ScalarValue", "value"> {
  let summary = "representing the value of scalar";

  let parameters = (ins
    "StringAttr":$value
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Array Attributes
//===----------------------------------------------------------------------===//

def TmpBufDescArrayAttr : TypedArrayAttrBase<TmpBufDescAttr, "Array of TmpBufDesc attributes">;

def AscTensorGroupsArrayAttr : TypedArrayAttrBase<AscTensorGroupsAttr, "Array of AscTensorGroups attributes">;

#endif // AFIR_ATTRS
