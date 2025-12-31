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

**MLIR 定义** (`AFIRGraphAttrs.td`):
```tablegen
def AFIR_AscGraph : AFIR_Attr<"AscGraph", "graph"> {
  let parameters = (ins
    "AFIR_AscGraphAttrGroups":$asc_graph_attr,
    ArrayRefParameter<"Attribute", "array of AscNode">:$asc_node,
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
    "int64_t":$tiling_key,
    ArrayRefParameter<"Attribute", "array of AxisAttr">:$axis,
    "int64_t":$type,
    ArrayRefParameter<"Attribute", "array of size variable strings">:$size_var
  );
}
```

**说明**:
- `tiling_key`: Tiling 配置的唯一标识
- `axis`: 计算轴的定义数组
- `type`: 图的类型标识
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
def AFIR_AxisAttr : AFIR_Attr<"AxisAttr", "axis"> {
  let parameters = (ins
    "int64_t":$id,
    "StringAttr":$name,
    "int32_t":$axis_type,
    "bool":$bind_block,
    "StringAttr":$size,
    "StringAttr":$align,
    ArrayRefParameter<"int64_t">:$from,
    "int64_t":$split_pair_other_id,
    "bool":$allow_oversize_axis,
    "bool":$allow_unaligned_tail
  );
}
```

**说明**:
- `id`: 轴的唯一标识符
- `name`: 轴的名称
- `axis_type`: 轴的类型（如块轴、线程轴等）
- `bind_block`: 是否绑定到硬件块
- `size`: 轴的大小（表达式字符串）
- `align`: 对齐要求（表达式字符串）
- `from`: 轴派生自哪些其他轴
- `split_pair_other_id`: Split pair 的另一个轴 ID
- `allow_oversize_axis`: 是否允许超大轴
- `allow_unaligned_tail`: 是否允许未对齐的尾部

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
    ArrayRefParameter<"Attribute", "array of AscInputSource">:$input_src,
    ArrayRefParameter<"Attribute", "array of AscTensor">:$outputs,
    "AFIR_AscNodeAttrGroups":$attr,
    "AFIR_IrDef":$ir_def
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
    "AFIR_AscTensorAttrGroups":$attr
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
}
```

**说明**:
- `dtype`: 数据类型（参见 DataType 枚举）
- `axis_ids`: 关联的轴 ID 数组
- `repeats`: 重复次数（表达式数组）
- `strides`: 步长（表达式数组）
- `vectorized_axis`: 向量化的轴
- `vectorized_strides`: 向量化的步长
- `mem`: 内存属性
- `que`: 内存队列属性
- `buf`: 内存缓冲区属性
- `opt`: 内存优化属性

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
def AFIR_MemAttr : AFIR_Attr<"MemAttr", "mem"> {
  let parameters = (ins
    "int64_t":$tensor_id,
    "int32_t":$alloc_type,
    "int32_t":$position,
    "int32_t":$hardware,
    ArrayRefParameter<"int64_t">:$buf_ids,
    "StringAttr":$name,
    "int64_t":$reuse_id
  );
}
```

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
def AFIR_MemQueueAttr : AFIR_Attr<"MemQueueAttr", "mem_queue"> {
  let parameters = (ins
    "int64_t":$id,
    "int64_t":$depth,
    "int64_t":$buf_num,
    "StringAttr":$name
  );
}
```

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
def AFIR_MemBufAttr : AFIR_Attr<"MemBufAttr", "mem_buf"> {
  let parameters = (ins
    "int64_t":$id,
    "StringAttr":$name
  );
}
```

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
def AFIR_MemOptAttr : AFIR_Attr<"MemOptAttr", "mem_opt"> {
  let parameters = (ins
    "int64_t":$reuse_id,
    "int64_t":$ref_tensor,
    "int64_t":$merge_scope
  );
}
```

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
    OptionalParameter<"AFIR_SchedInfo">:$sched,
    OptionalParameter<"AFIR_ApiInfo">:$api,
    "DictionaryAttr":$ir_attr_def,
    ArrayRefParameter<"Attribute", "array of TmpBufferGroup">:$tmp_buffers
  );
}
```

**说明**:
- `name`: 节点名称
- `type`: 节点类型
- `sched`: 调度信息
- `api`: API 信息
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
    "int64_t":$exec_order,
    ArrayRefParameter<"int64_t">:$axis,
    "int64_t":$loop_axis,
    "int32_t":$exec_condition
  );
}
```

**说明**:
- `exec_order`: 执行顺序
- `axis`: 轴映射
- `loop_axis`: 循环轴
- `exec_condition`: 执行条件

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
    "int32_t":$type,
    "int32_t":$compute_type,
    "int32_t":$unit
  );
}
```

**说明**:
- `type`: API 类型
- `compute_type`: 计算类型
- `unit`: 执行单元

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
    "AFIR_TmpBufDesc":$buf_desc,
    "AFIR_MemAttr":$mem,
    "int64_t":$id
  );
}
```

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
    "int64_t":$life_time_axis_id
  );
}
```

**说明**:
- `size`: 缓冲区大小（表达式字符串）
- `life_time_axis_id`: 生命周期轴 ID

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
    ArrayRefParameter<"Attribute", "array of input name strings">:$input_names,
    ArrayRefParameter<"Attribute", "array of output name strings">:$output_names,
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

## DataType 枚举

DataType 是一个重要的枚举类型，定义了所有支持的数据类型：

```cpp
enum DataType {
  DT_UNDEFINED = 0,
  DT_FLOAT = 1,
  DT_FLOAT16 = 2,
  DT_INT8 = 3,
  DT_UINT8 = 4,
  // ... 共 41 种数据类型
  DT_FLOAT4_E1M2 = 40
}
```

在 MLIR 中定义为 `AFIR_DataTypeEnum`，包含所有相同的枚举值。

---

## 数据结构层次图

```
AscGraphDef
├── AscGraphAttrGroupsDef
│   ├── tiling_key
│   ├── axis (array)
│   │   └── AxisDef
│   │       ├── id, name, axis_type
│   │       ├── bind_block
│   │       ├── size, align
│   │       ├── from (array)
│   │       ├── split_pair_other_id
│   │       └── allow_oversize_axis, allow_unaligned_tail
│   ├── type
│   └── size_var (array)
├── asc_node (array)
│   └── AscNodeDef
│       ├── input_src (array)
│       │   └── AscInputSourceDef
│       │       ├── src_node_name
│       │       └── src_out_index
│       ├── outputs (array)
│       │   └── AscTensorDef
│       │       └── AscTensorAttrGroupsDef
│       │           ├── dtype
│       │           ├── axis_ids, repeats, strides
│       │           ├── vectorized_axis, vectorized_strides
│       │           ├── MemAttrDef
│       │           ├── MemQueueAttrDef
│       │           ├── MemBufAttrDef
│       │           └── MemOptAttrDef
│       ├── attr (AscNodeAttrGroupsDef)
│       │   ├── name, type
│       │   ├── SchedInfoDef
│       │   ├── ApiInfoDef
│       │   ├── ir_attr_def (dict)
│       │   └── tmp_buffers (array)
│       │       └── TmpBufferGroupDef
│       │           ├── TmpBufDescDef
│       │           ├── MemAttrDef
│       │           └── id
│       └── ir_def (IrDef)
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
        axis_type = 1,
        bind_block = true,
        size = "1024",
        align = "16",
        from = [],
        split_pair_other_id = -1,
        allow_oversize_axis = false,
        allow_unaligned_tail = false
      >
    ],
    type = 0,
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
   - `int32` → `int32_t`
   - `bool` → `bool`
4. **optional/nullable → OptionalParameter**: 可选字段使用 OptionalParameter
5. **map → DictionaryAttr**: proto 的 map 映射为 MLIR 的 DictionaryAttr
6. **表达式字符串**: 在 proto 中标记为 `// expression` 的 string 字段在 MLIR 中保持为 StringAttr

---

## 文件清单

- **AFIRGraphAttrs.td**: MLIR 属性定义文件
- **ascendc_ir.proto**: AscGraph 主要结构定义
- **ge_ir.proto**: 引用的属性组定义
- **AFIRDialect.td**: 方言主文件（已更新包含图属性）

---

## 下一步工作

1. 实现 C++ 类的 parser 和 printer 方法
2. 实现 Proto ↔ MLIR 的转换工具
3. 添加 verifier 验证属性的正确性
4. 实现图操作（GraphOp）使用这些属性
5. 添加单元测试
