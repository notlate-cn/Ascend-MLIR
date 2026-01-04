# AscGraph 结构梳理与 MLIR AFIR 方言映射

本文档详细描述了从 protobuf 定义（`ascendc_ir.proto` 和 `ge_ir.proto`）到 MLIR AFIR 方言的 1:1 映射。

## 概述

AscGraph 是 Ascend 计算图的核心数据结构，定义在 `graph_metadef/proto/ascendc_ir.proto` 中。它描述了一个完整的计算图，包括图属性、节点定义和它们之间的连接关系。

## 结构映射关系

### 1. AscGraphDef (顶层结构)

**Proto 定义** (`ascendc_ir.proto`):
```protobuf
message AscGraphDef {
  ge.proto.AscGraphAttrGroupsDef asc_graph_attr = 1;
  repeated AscNodeDef asc_node = 2;
  string graph_name = 3;
}
```

**MLIR 定义** (`AFIRAttrs.td`):
```tablegen
def AFIR_AscGraph : AFIR_Attr<"AscGraph", "graph"> {
  let parameters = (ins
    "AscGraphAttrGroupsAttr":$asc_graph_attr,
    ArrayRefParameter<"AscNodeAttr", "array of AscNode">:$asc_node,
    "StringAttr":$graph_name
  );
}
```

**说明**:
- `asc_graph_attr`: 图级别的属性（轴定义、tiling key等）
- `asc_node`: 图中的所有节点
- `graph_name`: 图的名称

---

### 2. AscGraphAttrGroupsDef (图属性组)

**Proto 定义** (`ge_ir.proto`):
```protobuf
message AscGraphAttrGroupsDef {
  int64 tiling_key = 1;
  repeated AxisDef axis = 2;
  int64 type = 3;
  repeated string size_var = 4;
}
```

**MLIR 定义**:
```tablegen
def AFIR_AscGraphAttrGroups : AFIR_Attr<"AscGraphAttrGroups", "asc_graph"> {
  let parameters = (ins
    DefaultValuedParameter<"int64_t", "-1">:$tiling_key,
    ArrayRefParameter<"AxisAttr", "array of Axis">:$axis,
    "AscGraphTypeAttr":$type,  // 使用枚举类型替代 int64
    ArrayRefParameter<"Attribute", "array of size variable strings">:$size_var
  );
}
```

**说明**:
- `tiling_key`: Tiling 配置的唯一标识（默认值 -1）
- `axis`: 计算轴的定义数组
- `type`: 图的类型（使用 `AscGraphTypeAttr` 枚举：HintGraph=0, ImplGraph=1）
- `size_var`: 大小变量的字符串数组

---

### 3. AxisDef (轴定义)

**Proto 定义** (`ge_ir.proto`):
```protobuf
message AxisDef {
  int64 id = 1;
  string name = 2;
  int32 axis_type = 3;
  bool bind_block = 4;
  string size = 5;  // expression
  string align = 6;
  repeated int64 from = 7;
  int64 split_pair_other_id = 8;
  bool allow_oversize_axis = 9;
  bool allow_unaligned_tail = 10;
}
```

**MLIR 定义**:
```tablegen
def AFIR_Axis : AFIR_Attr<"Axis", "axis"> {
  let parameters = (ins
    DefaultValuedParameter<"int64_t", "-1">:$id,
    "StringAttr":$name,
    "AxisTypeAttr":$axis_type,  // 使用枚举类型
    DefaultValuedParameter<"bool", "false">:$bind_block,
    "Attribute":$size,
    OptionalParameter<"StringAttr">:$align,  // 可选参数
    ArrayRefParameter<"int64_t">:$from,
    DefaultValuedParameter<"int64_t", "-1">:$split_pair_other_id
    // allow_oversize_axis: 未定义
    // allow_unaligned_tail: 未定义
  );
}
```

**差异说明**:
- `axis_type`: 使用 `AxisTypeAttr` 枚举（Original, BlockOuter, BlockInner, TileOuter, TileInner, Merged, Invalid）
- `align`: 改为可选参数
- `allow_oversize_axis`: **未定义**
- `allow_unaligned_tail`: **未定义**

---

### 4. AscNodeDef (节点定义)

**Proto 定义** (`ascendc_ir.proto`):
```protobuf
message AscNodeDef {
  repeated AscInputSourceDef input_src = 1;
  repeated AscTensorDef outputs = 2;
  ge.proto.AscNodeAttrGroupsDef attr = 3;
  IrDef ir_def = 4;
}
```

**MLIR 定义**:
```tablegen
def AFIR_AscNode : AFIR_Attr<"AscNode", "node"> {
  let parameters = (ins
    ArrayRefParameter<"AscInputSourceAttr", "array of AscInputSource">:$input_src,
    ArrayRefParameter<"AscTensorAttr", "array of AscTensor">:$outputs,
    "AscNodeAttrGroupsAttr":$attr,
    "IrDefAttr":$ir_def
  );
}
```

**说明**:
- `input_src`: 输入源的引用（指向其他节点的输出）
- `outputs`: 节点的输出张量定义
- `attr`: 节点的属性组
- `ir_def`: IR 级别的定义信息

---

### 5. AscInputSourceDef (输入源定义)

**Proto 定义** (`ascendc_ir.proto`):
```protobuf
message AscInputSourceDef {
  string src_node_name = 1;
  int32 src_out_index = 2;
}
```

**MLIR 定义**:
```tablegen
def AFIR_AscInputSource : AFIR_Attr<"AscInputSource", "input_src"> {
  let parameters = (ins
    "StringAttr":$src_node_name,
    "int32_t":$src_out_index
  );
}
```

**说明**:
- `src_node_name`: 源节点的名称
- `src_out_index`: 源节点的输出索引

---

### 6. AscTensorDef (张量定义)

**Proto 定义** (`ascendc_ir.proto`):
```protobuf
message AscTensorDef {
  ge.proto.AscTensorAttrGroupsDef attr = 1;
}
```

**MLIR 定义**:
```tablegen
def AFIR_AscTensor : AFIR_Attr<"AscTensor", "tensor_def"> {
  let parameters = (ins
    "AscTensorAttrGroupsAttr":$attr
  );
}
```

---

### 7. AscTensorAttrGroupsDef (张量属性组)

**Proto 定义** (`ge_ir.proto`):
```protobuf
message AscTensorAttrGroupsDef {
  int64 dtype = 1;
  repeated int64 axis_ids = 2;
  repeated string repeats = 3;  // expression
  repeated string strides = 4;  // expression
  repeated int64 vectorized_axis = 5;
  repeated string vectorized_strides = 6;
  MemAttrDef mem = 7;
  MemQueueAttrDef que = 8;
  MemBufAttrDef buf = 9;
  MemOptAttrDef opt = 10;
}
```

**MLIR 定义**:
```tablegen
def AFIR_AscTensorAttrGroups : AFIR_Attr<"AscTensorAttrGroups", "asc_tensor"> {
  let parameters = (ins
    "DataTypeAttr":$dtype,  // 使用枚举类型
    ArrayRefParameter<"int64_t">:$axis_ids,
    ArrayRefParameter<"Attribute", "array of expression strings">:$repeats,
    ArrayRefParameter<"Attribute", "array of expression strings">:$strides,
    ArrayRefParameter<"int64_t">:$vectorized_axis,
    ArrayRefParameter<"Attribute", "array of vectorized stride expressions">:$vectorized_strides,
    OptionalParameter<"MemAttr">:$mem,
    OptionalParameter<"MemQueueAttr">:$que,
    OptionalParameter<"MemBufAttr">:$buf
    // opt (MemOptAttr): 未定义
  );
}
```

**差异说明**:
- `dtype`: 使用 `DataTypeAttr` 枚举（41种数据类型）
- `mem`, `que`, `buf`: 改为可选参数
- `opt` (MemOptAttrDef): **未定义**

---

### 8. 内存相关属性

#### MemAttrDef (内存属性)

**Proto 定义** (`ge_ir.proto`):
```protobuf
message MemAttrDef {
  int64 tensor_id = 1;
  int32 alloc_type = 2;
  int32 position = 3;
  int32 hardware = 4;
  repeated int64 buf_ids = 5;
  string name = 6;
  int64 reuse_id = 7;
}
```

**MLIR 定义**:
```tablegen
def AFIR_Mem : AFIR_Attr<"Mem", "mem"> {
  let parameters = (ins
    "int64_t":$tensor_id,
    "AllocTypeAttr":$alloc_type,  // 使用枚举类型 (GLOBAL, L1, L2, QBUF, TBUF)
    "PositionAttr":$position,     // 使用枚举类型 (GM, VECTOR_IN, VECTOR_OUT, VECTOR_CALC)
    "HardwareAttr":$hardware,     // 使用枚举类型 (GM, UB)
    // buf_ids: 未定义
    // name: 未定义
    DefaultValuedParameter<"int64_t", "-1">:$reuse_id
  );
}
```

**差异说明**:
- `alloc_type`: 使用 `AllocTypeAttr` 枚举
- `position`: 使用 `PositionAttr` 枚举
- `hardware`: 使用 `HardwareAttr` 枚举
- `buf_ids`: **未定义**
- `name`: **未定义**

#### MemQueueAttrDef (内存队列属性)

**Proto 定义**:
```protobuf
message MemQueueAttrDef {
  int64 id = 1;
  int64 depth = 2;
  int64 buf_num = 3;
  string name = 4;
}
```

**MLIR 定义**:
```tablegen
def AFIR_MemQueue : AFIR_Attr<"MemQueue", "mem_queue"> {
  let parameters = (ins
    "int64_t":$id,
    DefaultValuedParameter<"int64_t", "2">:$depth,
    "int64_t":$buf_num
    // name: 未定义
  );
}
```

**差异说明**:
- `depth`: 使用默认值 2
- `name`: **未定义**

#### MemBufAttrDef (内存缓冲区属性)

**Proto 定义**:
```protobuf
message MemBufAttrDef {
  int64 id = 1;
  string name = 2;
}
```

**MLIR 定义**:
```tablegen
def AFIR_MemBuf : AFIR_Attr<"MemBuf", "mem_buf"> {
  let parameters = (ins
    "int64_t":$id
    // name: 未定义
  );
}
```

**差异说明**:
- `name`: **未定义**

#### MemOptAttrDef (内存优化属性)

**Proto 定义**:
```protobuf
message MemOptAttrDef {
  int64 reuse_id = 1;
  int64 ref_tensor = 2;
  int64 merge_scope = 3;
}
```

**MLIR 定义**:
```tablegen
// 未定义 - 整个 MemOptAttr 属性未实现
```

**状态**: **未定义**

---

### 9. AscNodeAttrGroupsDef (节点属性组)

**Proto 定义** (`ge_ir.proto`):
```protobuf
message AscNodeAttrGroupsDef {
  string name = 1;
  string type = 2;
  SchedInfoDef sched = 3;
  ApiInfoDef api = 4;
  AscIrAttrDef ir_attr_def = 5;
  repeated TmpBufferGroupDef tmp_buffers = 6;
}
```

**MLIR 定义**:
```tablegen
def AFIR_AscNodeAttrGroups : AFIR_Attr<"AscNodeAttrGroups", "asc_node"> {
  let parameters = (ins
    "StringAttr":$name,
    "StringAttr":$type,
    OptionalParameter<"SchedInfoAttr">:$sched,
    OptionalParameter<"ApiInfoAttr">:$api,
    "DictionaryAttr":$ir_attr_def,
    ArrayRefParameter<"TmpBufferGroupAttr", "array of TmpBufferGroup">:$tmp_buffers
  );
}
```

**说明**:
- `name`: 节点名称
- `type`: 节点类型
- `sched`: 调度信息（可选）
- `api`: API 信息（可选）
- `ir_attr_def`: IR 属性字典（映射自 AscIrAttrDef.attr）
- `tmp_buffers`: 临时缓冲区组数组

---

### 10. SchedInfoDef (调度信息)

**Proto 定义** (`ge_ir.proto`):
```protobuf
message SchedInfoDef {
  int64 exec_order = 1;
  repeated int64 axis = 2;
  int64 loop_axis = 3;
  int32 exec_condition = 4;
}
```

**MLIR 定义**:
```tablegen
def AFIR_SchedInfo : AFIR_Attr<"SchedInfo", "sched"> {
  let parameters = (ins
    DefaultValuedParameter<"int64_t", "-1">:$exec_order,
    ArrayRefParameter<"int64_t">:$axis,
    DefaultValuedParameter<"int64_t", "-1">:$loop_axis,
    "ExecuteConditionAttr":$exec_condition  // 使用枚举类型
  );
}
```

**差异说明**:
- `exec_order`: 使用默认值 -1
- `loop_axis`: 使用默认值 -1
- `exec_condition`: 使用 `ExecuteConditionAttr` 枚举（NoCache, CacheBlockSplitFusedBroadcastAxis, CacheBlockSplitOriginBroadcastAxis, ConditionInvalid）

---

### 11. ApiInfoDef (API 信息)

**Proto 定义** (`ge_ir.proto`):
```protobuf
message ApiInfoDef {
  int32 type = 1;
  int32 compute_type = 2;
  int32 unit = 3;
}
```

**MLIR 定义**:
```tablegen
def AFIR_ApiInfo : AFIR_Attr<"ApiInfo", "api"> {
  let parameters = (ins
    "ApiTypeAttr":$type,          // 使用枚举类型 (Buffer, Compute, Invalid)
    "ComputeTypeAttr":$compute_type,  // 使用枚举类型 (Load, Store, ReduceStore, ...)
    "ComputeUnitAttr":$unit       // 使用枚举类型 (None, MTE1, MTE2, MTE3, Scalar, Vector, Cube, Invalid)
  );
}
```

**差异说明**:
- 所有字段都使用枚举类型替代整数类型

---

### 12. TmpBufferGroupDef (临时缓冲区组)

**Proto 定义** (`ge_ir.proto`):
```protobuf
message TmpBufferGroupDef {
  TmpBufDescDef buf_desc = 1;
  MemAttrDef mem = 2;
  int64 id = 3;
}
```

**MLIR 定义**:
```tablegen
def AFIR_TmpBufferGroup : AFIR_Attr<"TmpBufferGroup", "tmp_buffer"> {
  let parameters = (ins
    "TmpBufDescAttr":$buf_desc,
    "MemAttr":$mem,
    DefaultValuedParameter<"int64_t", "-1">:$id
  );
}
```

**差异说明**:
- `id`: 使用默认值 -1

---

### 13. TmpBufDescDef (临时缓冲区描述)

**Proto 定义** (`ge_ir.proto`):
```protobuf
message TmpBufDescDef {
  string size = 1;  // expression
  int64 life_time_axis_id = 2;
}
```

**MLIR 定义**:
```tablegen
def AFIR_TmpBufDesc : AFIR_Attr<"TmpBufDesc", "tmp_buf_desc"> {
  let parameters = (ins
    "StringAttr":$size,
    DefaultValuedParameter<"int64_t", "-1">:$life_time_axis_id
  );
}
```

**差异说明**:
- `life_time_axis_id`: 使用默认值 -1

---

### 14. IrDef (IR 定义)

**Proto 定义** (`ascendc_ir.proto`):
```protobuf
message IrDef {
  repeated string input_names = 1;
  repeated string output_names = 2;
  repeated int64 input_ir_type = 3;
  repeated int64 output_ir_type = 4;
  string type = 5;
  repeated int64 input_nums = 6;
  repeated int64 output_nums = 7;
}
```

**MLIR 定义**:
```tablegen
def AFIR_IrDef : AFIR_Attr<"IrDef", "ir_def"> {
  let parameters = (ins
    ArrayRefParameter<"StringAttr", "array of input name strings">:$input_names,
    ArrayRefParameter<"StringAttr", "array of output name strings">:$output_names,
    ArrayRefParameter<"int64_t">:$input_ir_type,
    ArrayRefParameter<"int64_t">:$output_ir_type,
    "StringAttr":$type,
    ArrayRefParameter<"int64_t">:$input_nums,
    ArrayRefParameter<"int64_t">:$output_nums
  );
}
```

**说明**:
- `input_names`: 输入名称数组
- `output_names`: 输出名称数组
- `input_ir_type`: 输入 IR 类型
- `output_ir_type`: 输出 IR 类型
- `type`: IR 类型
- `input_nums`: 输入数量
- `output_nums`: 输出数量

---

## 枚举类型定义

所有枚举类型定义在 `AFIREnums.td` 中：

### DataType 枚举

```tablegen
def AFIR_DataTypeEnum : I32EnumAttr<"DataType", ...> {
  // 41 种数据类型
  DT_UNDEFINED = 0, DT_FLOAT = 1, DT_FLOAT16 = 2, DT_INT8 = 3,
  DT_UINT8 = 4, DT_INT16 = 5, DT_UINT16 = 6, DT_INT32 = 7,
  DT_INT64 = 8, DT_UINT32 = 9, DT_UINT64 = 10, DT_BOOL = 11,
  DT_DOUBLE = 12, DT_STRING = 13, ... DT_FLOAT4_E1M2 = 40
}
```

### 其他枚举

| 枚举类型 | 说明 | 值 |
|----------|------|-----|
| `AllocTypeAttr` | 内存分配类型 | GLOBAL, L1, L2, QBUF, TBUF |
| `PositionAttr` | 内存位置 | GM, VECTOR_IN, VECTOR_OUT, VECTOR_CALC |
| `HardwareAttr` | 硬件目标 | GM, UB |
| `AxisTypeAttr` | 轴类型 | Original, BlockOuter, BlockInner, TileOuter, TileInner, Merged, Invalid |
| `ExecuteConditionAttr` | 执行条件 | NoCache, CacheBlockSplitFusedBroadcastAxis, CacheBlockSplitOriginBroadcastAxis, ConditionInvalid |
| `ApiTypeAttr` | API 类型 | Buffer, Compute, Invalid |
| `ComputeTypeAttr` | 计算类型 | Load, Store, ReduceStore, Elewise, Broadcast, Reduce, Transpose, Concat, Gather, Cube, Split, Invalid |
| `ComputeUnitAttr` | 计算单元 | None, MTE1, MTE2, MTE3, Scalar, Vector, Cube, Invalid |
| `AscGraphTypeAttr` | 图类型 | HintGraph, ImplGraph |

---

## 未定义属性汇总

以下 Proto 字段在当前 MLIR 定义中未实现：

| Proto Message | 未定义字段 |
|---------------|-----------|
| `AxisDef` | `allow_oversize_axis`, `allow_unaligned_tail` |
| `MemAttrDef` | `buf_ids`, `name` |
| `MemQueueAttrDef` | `name` |
| `MemBufAttrDef` | `name` |
| `AscTensorAttrGroupsDef` | `opt` (MemOptAttrDef) |
| `MemOptAttrDef` | **整个属性未定义** |

---

## 数据结构层次图

```
AscGraphDef
├── AscGraphAttrGroupsDef
│   ├── tiling_key (默认 -1)
│   ├── axis (array)
│   │   └── AxisAttr
│   │       ├── id (默认 -1), name
│   │       ├── axis_type (枚举)
│   │       ├── bind_block (默认 false)
│   │       ├── size, align (可选)
│   │       ├── from (array)
│   │       └── split_pair_other_id (默认 -1)
│   ├── type (AscGraphTypeAttr 枚举)
│   └── size_var (array)
├── asc_node (array)
│   └── AscNodeAttr
│       ├── input_src (array)
│       │   └── AscInputSourceAttr
│       │       ├── src_node_name
│       │       └── src_out_index
│       ├── outputs (array)
│       │   └── AscTensorAttr
│       │       └── AscTensorAttrGroupsAttr
│       │           ├── dtype (DataTypeAttr 枚举)
│       │           ├── axis_ids, repeats, strides
│       │           ├── vectorized_axis, vectorized_strides
│       │           ├── MemAttr (可选)
│       │           ├── MemQueueAttr (可选)
│       │           └── MemBufAttr (可选)
│       ├── attr (AscNodeAttrGroupsAttr)
│       │   ├── name, type
│       │   ├── SchedInfoAttr (可选)
│       │   ├── ApiInfoAttr (可选)
│       │   ├── ir_attr_def (DictionaryAttr)
│       │   └── tmp_buffers (array)
│       │       └── TmpBufferGroupAttr
│       │           ├── TmpBufDescAttr
│       │           ├── MemAttr
│       │           └── id (默认 -1)
│       └── ir_def (IrDefAttr)
│           ├── input_names, output_names
│           ├── input_ir_type, output_ir_type
│           ├── type
│           └── input_nums, output_nums
└── graph_name
```

---

## 使用说明

### 在 MLIR 中使用这些属性

```mlir
// 示例：定义一个 AscGraph
#graph = #afir.graph<
  asc_graph_attr = #afir.asc_graph<
    tiling_key = 12345,
    axis = [
      #afir.axis<
        id = 0,
        name = "block_idx",
        axis_type = #afir.axis_type<BlockOuter>,
        bind_block = true,
        size = "1024",
        from = []
      >
    ],
    type = #afir.type<HintGraph>,
    size_var = ["N", "M"]
  >,
  asc_node = [
    #afir.node<...>
  ],
  graph_name = "example_graph"
>
```

### 与 Proto 的互转

可以实现以下转换函数：
- `mlir::afir::AscGraphAttr -> ascendc_ir::proto::AscGraphDef`
- `ascendc_ir::proto::AscGraphDef -> mlir::afir::AscGraphAttr`

---

## 映射原则

1. **Proto message → MLIR Attr**: 每个 proto message 映射为一个 MLIR attribute definition
2. **repeated → ArrayRefParameter**: proto 的 repeated 字段映射为 MLIR 的数组参数
3. **基本类型映射**:
   - `string` → `StringAttr`
   - `int64` → `int64_t`
   - `int32` → `int32_t` 或枚举类型
   - `bool` → `bool`
4. **optional/nullable → OptionalParameter**: 可选字段使用 OptionalParameter
5. **map → DictionaryAttr**: proto 的 map 映射为 MLIR 的 DictionaryAttr
6. **表达式字符串**: 在 proto 中标记为 `// expression` 的 string 字段在 MLIR 中保持为 StringAttr
7. **int32 枚举 → EnumAttr**: proto 中表示枚举的 int32 字段使用 MLIR EnumAttr

---

## 文件清单

- **AFIREnums.td**: 所有枚举类型定义
- **AFIRAttrs.td**: MLIR 属性定义文件
- **ascendc_ir.proto**: AscGraph 主要结构定义
- **ge_ir.proto**: 引用的属性组定义
- **AFIRDialect.td**: 方言主文件

---

## 下一步工作

1. ✅ 实现基本属性定义
2. ⬜ 实现 Proto ↔ MLIR 的转换工具
3. ⬜ 添加 verifier 验证属性的正确性
4. ⬜ 实现缺失的属性（MemOptAttr 等）
5. ⬜ 添加单元测试
