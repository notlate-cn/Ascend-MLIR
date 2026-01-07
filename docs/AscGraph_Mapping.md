# AscGraph 到 AFIR 方言的映射关系

本文档描述了从 AscGraph protobuf 定义(`ascendc_ir.proto` 和 `ge_ir.proto`)到 MLIR AFIR 方言的映射关系。

## 概述

AFIR (Ascend Fusion IR) 方言是对 AscGraph 计算图的 MLIR 抽象表示。与直接 1:1 映射 protobuf 结构不同,AFIR 采用了更符合 MLIR 惯例的抽象设计:

- **数据类型抽象**: 使用 MLIR 内置类型系统(如 `f32`, `f16`, `i32`)替代 AscGraph 的 DataType 枚举
- **简化属性**: 将复杂的 protobuf 嵌套结构简化为核心计算和内存管理属性
- **操作导向**: 以 MLIR 操作(Operations)为中心,而非节点属性组

## 设计原则

### 1. 抽象层次
AFIR 方言处于两个抽象层次之间:
- **AscGraph (下层)**: Ascend 编译器的低级IR,包含详细的调度、内存分配、API调用信息
- **MLIR Standard Dialects (上层)**: 通用计算图表示

### 2. 信息保留策略
- **核心信息**: 计算轴(Axis)、张量布局(vectorization)、内存位置保留
- **调度信息**: 通过操作属性保留(如 `loop_axis`, `ir_attr_def`)
- **类型信息**: 映射到 MLIR 类型系统
- **丢弃信息**: 低级 API 细节、执行条件等可从上下文推导的信息

---

## 类型映射

### AscGraph DataType → MLIR Type

AscGraph 的 `DataType` 枚举映射为 MLIR 内置类型:

| AscGraph DataType | MLIR Type | 说明 |
|-------------------|-----------|------|
| `DT_FLOAT` (1) | `f32` | 32位浮点 |
| `DT_FLOAT16` (2) | `f16` | 16位浮点 |
| `DT_BF16` (27) | `bf16` | Brain Float 16 |
| `DT_INT8` (3) | `i8` | 8位有符号整数 |
| `DT_UINT8` (4) | `ui8` | 8位无符号整数 |
| `DT_INT16` (5) | `i16` | 16位有符号整数 |
| `DT_UINT16` (6) | `ui16` | 16位无符号整数 |
| `DT_INT32` (7) | `i32` | 32位有符号整数 |
| `DT_INT64` (8) | `i64` | 64位有符号整数 |
| `DT_BOOL` (11) | `i1` | 布尔类型 |
| `DT_DOUBLE` (12) | `f64` | 64位浮点 |

**特殊类型**: 复数、量化类型等暂不支持,可扩展。

---

## 属性映射

### 1. 图级别属性 (Graph-Level Attributes)

#### AFIR_AscGraphAttrGroups

**用途**: 描述整个计算图的属性,作为 MLIR 模块(Module)的属性附加。

**映射关系**:

```tablegen
def AFIR_AscGraphAttrGroups : AFIR_Attr<"AscGraphAttrGroups", "asc_graph"> {
  let parameters = (ins
    DefaultValuedParameter<"int64_t", "-1">:$tiling_key,        // from AscGraphAttrGroupsDef.tiling_key
    ArrayRefParameter<"AxisAttr", "array of Axis">:$axis,      // from AscGraphAttrGroupsDef.axis[]
    EnumParameter<AFIR_AscGraphTypeEnum>:$type,                // from AscGraphAttrGroupsDef.type
    ArrayRefParameter<"Attribute", "array of strings">:$size_var // from AscGraphAttrGroupsDef.size_var[]
  );
}
```

**Protobuf 源** (`ge.proto.AscGraphAttrGroupsDef`):
```protobuf
message AscGraphAttrGroupsDef {
  int64 tiling_key = 1;       // → tiling_key
  repeated AxisDef axis = 2;  // → axis[]
  int64 type = 3;             // → type (0: COMPUTE, 1: Invalid)
  repeated string size_var = 4; // → size_var[]
}
```

**类型枚举变化**:
- 原 `HintGraph=0, ImplGraph=1` 简化为 `COMPUTE=0, Invalid=1`

---

### 2. 轴定义 (Axis Definition)

#### AFIR_Axis

**用途**: 定义计算循环的迭代轴,包括轴类型、大小、对齐等。

**映射关系** (保持 1:1):

```tablegen
def AFIR_Axis : AFIR_Attr<"Axis", "axis"> {
  let parameters = (ins
    DefaultValuedParameter<"int64_t", "-1">:$id,              // from AxisDef.id
    "StringAttr":$name,                                       // from AxisDef.name
    "AxisTypeAttr":$axis_type,                                // from AxisDef.axis_type
    DefaultValuedParameter<"bool", "false">:$bind_block,      // from AxisDef.bind_block
    "Attribute":$size,                                        // from AxisDef.size (expression string)
    OptionalParameter<"StringAttr">:$align,                   // from AxisDef.align
    ArrayRefParameter<"int64_t">:$from,                       // from AxisDef.from[]
    DefaultValuedParameter<"int64_t", "-1">:$split_pair_other_id // from AxisDef.split_pair_other_id
  );
}
```

**Protobuf 源** (`ge.proto.AxisDef`):
```protobuf
message AxisDef {
  int64 id = 1;
  string name = 2;
  int32 axis_type = 3;        // Original, BlockOuter, BlockInner, TileOuter, TileInner, Merged, Invalid
  bool bind_block = 4;
  string size = 5;            // expression
  string align = 6;
  repeated int64 from = 7;
  int64 split_pair_other_id = 8;
  bool allow_oversize_axis = 9;      // ❌ 未映射
  bool allow_unaligned_tail = 10;    // ❌ 未映射
}
```

**未映射字段**: `allow_oversize_axis`, `allow_unaligned_tail` 在 AFIR 中省略。

---

### 3. 张量属性 (Tensor Attributes)

#### AFIR_AscTensorGroups

**用途**: 描述张量的内存布局和向量化信息。

**重大简化**: 原 `AscTensorAttrGroupsDef` 包含 9 个字段,新版本只保留 7 个核心字段。

```tablegen
def AFIR_AscTensorGroups : AFIR_Attr<"AscTensorGroups", "asc_tensor"> {
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$vectorized_axis,            // from AscTensorAttrGroupsDef.vectorized_axis[]
    ArrayRefParameter<"Attribute">:$vectorized_strides,       // from AscTensorAttrGroupsDef.vectorized_strides[]
    "int64_t":$tensor_id,                                     // from MemAttrDef.tensor_id
    DefaultValuedParameter<"int64_t", "-1">:$reuse_id,        // from MemAttrDef.reuse_id
    EnumParameter<AFIR_PositionEnum>:$position,               // from MemAttrDef.position + alloc_type (融合)
    "int64_t":$position_id,                                   // from MemQueueAttrDef.id / MemBufAttrDef.id
    OptionalParameter<"uint32_t">:$depth,                     // from MemQueueAttrDef.depth
    OptionalParameter<"bool">:$is_double_buffer               // from MemQueueAttrDef.buf_num > 1
  );
}
```

**Protobuf 源** (组合多个消息):

**原 `AscTensorAttrGroupsDef`**:
```protobuf
message AscTensorAttrGroupsDef {
  int64 dtype = 1;                    // ❌ 改用 MLIR 类型
  repeated int64 axis_ids = 2;        // ❌ 隐式在操作类型推断
  repeated string repeats = 3;        // ❌ 隐式在操作类型推断
  repeated string strides = 4;        // ❌ 隐式在操作类型推断
  repeated int64 vectorized_axis = 5; // ✅ → vectorized_axis
  repeated string vectorized_strides = 6; // ✅ → vectorized_strides
  MemAttrDef mem = 7;                 // ✅ 部分映射 (见下)
  MemQueueAttrDef que = 8;            // ✅ 部分映射 (见下)
  MemBufAttrDef buf = 9;              // ✅ 部分映射 (见下)
  MemOptAttrDef opt = 10;             // ❌ 未映射
}
```

**内存属性融合**:

| Protobuf 字段 | AFIR 字段 | 说明 |
|--------------|-----------|------|
| `MemAttrDef.tensor_id` | `tensor_id` | 张量ID |
| `MemAttrDef.reuse_id` | `reuse_id` | 复用ID |
| `MemAttrDef.position` + `MemAttrDef.alloc_type` | `position` | 融合为单一枚举 |
| `MemQueueAttrDef.id` 或 `MemBufAttrDef.id` | `position_id` | 队列/缓冲区ID |
| `MemQueueAttrDef.depth` | `depth` | 队列深度 |
| `MemQueueAttrDef.buf_num` | `is_double_buffer` | buf_num > 1 则为 true |

**Position 枚举扩展**:

原 AscGraph 分别使用 `Position` (4种) 和 `AllocType` (5种) 枚举,AFIR 融合为单一 `Position` 枚举:

```tablegen
def AFIR_PositionEnum : I32EnumAttr<"Position", "Memory position",
  [GM,           // 0: Global Memory (原 Position.GM)
   VECTOR_IN,    // 1: 向量输入 (原 Position.VECTOR_IN)
   VECTOR_OUT,   // 2: 向量输出 (原 Position.VECTOR_OUT)
   VECTOR_CALC,  // 3: 向量计算 (原 Position.VECTOR_CALC)
   L1,           // 4: L1缓存 (原 AllocType.L1)
   L2,           // 5: L2缓存 (原 AllocType.L2)
   L0A,          // 6: L0A缓存 (扩展)
   L0B,          // 7: L0B缓存 (扩展)
   L0C]          // 8: L0C缓存 (扩展)
>
```

---

### 4. 临时缓冲区属性 (Temporary Buffer)

#### AFIR_TmpBufDesc

**映射关系** (保持 1:1):

```tablegen
def AFIR_TmpBufDesc : AFIR_Attr<"TmpBufDesc", "tmp_buf_desc"> {
  let parameters = (ins
    "StringAttr":$size,                                       // from TmpBufDescDef.size
    DefaultValuedParameter<"int64_t", "-1">:$life_time_axis_id // from TmpBufDescDef.life_time_axis_id
  );
}
```

**Protobuf 源** (`ge.proto.TmpBufDescDef`):
```protobuf
message TmpBufDescDef {
  string size = 1;               // expression → size
  int64 life_time_axis_id = 2;   // → life_time_axis_id
}
```

---

## 操作映射 (Operation Mapping)

### AFIR 操作结构

AFIR 方言的操作不再直接对应 AscGraph 的节点,而是抽象为标准计算操作。

#### 操作定义模式

以 `afir.add` 为例:

```tablegen
def AFIR_AddOp : AFIR_Op<"add", [Pure, ShapeHelperOpInterface, ShapeInferenceOpInterface]> {
  let arguments = (ins
    TensorOf<[F16, F32, I16, I32]>:$lhs,          // 左操作数
    TensorOf<[F16, F32, I16, I32]>:$rhs,          // 右操作数
    AffineMapArrayAttr:$indexing_maps,            // 索引映射 (从 axis_ids, repeats, strides 推导)
    DefaultValuedAttr<I32Attr, "-1">:$loop_axis,  // 循环轴 (from SchedInfoDef.loop_axis)
    OptionalAttr<DictionaryAttr>:$ir_attr_def,    // IR属性字典 (from AscIrAttrDef.attr)
    OptionalAttr<TmpBufDescArrayAttr>:$tmp_buffers, // 临时缓冲区 (from TmpBufferGroupDef[])
    AscTensorGroupsArrayAttr:$outputs             // 输出张量属性 (from AscTensorDef[])
  );
  let results = (outs TensorOf<[F16, F32, I16, I32]>:$result);
}
```

#### AscGraph 节点到 AFIR 操作的映射

| AscGraph Node Type | AFIR Operation | 参数映射 |
|-------------------|----------------|----------|
| `Add` | `afir.add` | input_src → lhs/rhs, outputs → outputs |
| `Sub` | `afir.sub` | 同上 |
| `Mul` | `afir.mul` | 同上 |
| `Div` | `afir.div` | 同上 (仅支持 F16, F32) |
| `Data` | `afir.data` | outputs → result type |
| `Load` | `afir.load` | input_src → input, outputs → result |
| `Store` | `afir.store` | input_src → value |
| `Broadcast` | `afir.broadcast` | input_src → input, outputs → result |
| `Output` | `afir.output` | input_src → input |

### 节点属性组到操作参数的映射

**AscGraph 节点结构**:
```protobuf
message AscNodeDef {
  repeated AscInputSourceDef input_src = 1;    // → 操作输入 (SSA values)
  repeated AscTensorDef outputs = 2;           // → outputs 属性
  AscNodeAttrGroupsDef attr = 3;               // → 分散映射到多个属性
  IrDef ir_def = 4;                            // → 类型推断
}
```

**`AscNodeAttrGroupsDef` 拆解**:
```protobuf
message AscNodeAttrGroupsDef {
  string name = 1;                             // → 操作的符号名(可选)
  string type = 2;                             // → 确定 AFIR 操作类型
  SchedInfoDef sched = 3;                      // → loop_axis, indexing_maps
  ApiInfoDef api = 4;                          // ❌ 不映射 (可推导)
  AscIrAttrDef ir_attr_def = 5;                // → ir_attr_def 字典
  repeated TmpBufferGroupDef tmp_buffers = 6;  // → tmp_buffers 数组
}
```

**`SchedInfoDef` 映射**:
```protobuf
message SchedInfoDef {
  int64 exec_order = 1;           // ❌ 不映射 (拓扑序隐含)
  repeated int64 axis = 2;        // → indexing_maps 推导
  int64 loop_axis = 3;            // → loop_axis
  int32 exec_condition = 4;       // ❌ 不映射
}
```

**`ApiInfoDef`**: 完全不映射,可从操作类型和参数推导。

---

## 转换示例

### AscGraph Protobuf → AFIR MLIR

**输入** (AscGraph Add 节点):
```protobuf
asc_node {
  input_src { src_node_name: "Load_1" src_out_index: 0 }
  input_src { src_node_name: "Broadcast_4" src_out_index: 0 }
  outputs {
    attr {
      axis_ids: [0, 1]
      repeats: ["20", "31"]
      strides: ["31", "1"]
      vectorized_axis: []
      vectorized_strides: []
      mem { tensor_id: -1 position: 2 }
    }
  }
  attr {
    name: "Add_5"
    type: "Add"
    sched { axis: [0, 1] loop_axis: -1 }
    api { type: 1 compute_type: 3 unit: 5 }
  }
  ir_def {
    input_names: ["x1", "x2"]
    output_names: ["y"]
    type: "Add"
  }
}
```

**输出** (AFIR MLIR):
```mlir
%result = afir.add %load_1, %broadcast_4 {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,  // 从 axis_ids, repeats, strides 推导
    affine_map<(d0, d1) -> (d0, d1)>
  ],
  loop_axis = -1,
  outputs = [
    #afir.asc_tensor<
      vectorized_axis = [],
      vectorized_strides = [],
      tensor_id = -1,
      reuse_id = -1,
      position = VECTOR_OUT,
      position_id = -1,
      depth = none,
      is_double_buffer = none
    >
  ]
} : (tensor<20x31xf32>, tensor<20x31xf32>) -> tensor<20x31xf32>
```

---

## 未映射的 AscGraph 结构

以下 protobuf 定义在 AFIR 中**完全移除**或**隐式推导**:

### 完全移除

| Protobuf 消息 | 原用途 | AFIR 处理方式 |
|--------------|--------|--------------|
| `MemAttrDef` | 内存分配属性 | 融合到 `AscTensorGroups` |
| `MemQueueAttrDef` | 队列属性 | 融合到 `AscTensorGroups` |
| `MemBufAttrDef` | 缓冲区属性 | 融合到 `AscTensorGroups` |
| `MemOptAttrDef` | 内存优化属性 | 不映射 (优化pass处理) |
| `SchedInfoDef` | 调度信息 | 部分映射到操作属性 |
| `ApiInfoDef` | API调用信息 | 不映射 (从操作类型推导) |
| `AscNodeAttrGroupsDef` | 节点属性组 | 拆解到操作参数 |
| `AscInputSourceDef` | 输入源引用 | 映射为 SSA value |
| `IrDef` | IR定义 | 用于类型推断,不保留 |

### 隐式推导

| 原 AscGraph 字段 | 推导来源 |
|-----------------|----------|
| `dtype` | MLIR 张量类型 (`tensor<*xf32>`) |
| `axis_ids`, `repeats`, `strides` | `indexing_maps` (AffineMap) |
| `exec_order` | MLIR 基本块内的拓扑序 |
| `api.type`, `api.compute_type`, `api.unit` | 操作名称 (`afir.add` → Compute, Elewise, Vector) |

---

## 枚举映射对照表

### Position (融合 AllocType)

| AscGraph | AFIR | 值 |
|----------|------|---|
| Position.GM | Position.GM | 0 |
| Position.VECTOR_IN | Position.VECTOR_IN | 1 |
| Position.VECTOR_OUT | Position.VECTOR_OUT | 2 |
| Position.VECTOR_CALC | Position.VECTOR_CALC | 3 |
| AllocType.L1 | Position.L1 | 4 |
| AllocType.L2 | Position.L2 | 5 |
| (新增) | Position.L0A | 6 |
| (新增) | Position.L0B | 7 |
| (新增) | Position.L0C | 8 |

### AxisType (无变化)

| AscGraph | AFIR | 值 |
|----------|------|---|
| Original | Original | 0 |
| BlockOuter | BlockOuter | 1 |
| BlockInner | BlockInner | 2 |
| TileOuter | TileOuter | 3 |
| TileInner | TileInner | 4 |
| Merged | Merged | 5 |
| Invalid | Invalid | 6 |

### AscGraphType (简化)

| AscGraph | AFIR | 值 |
|----------|------|---|
| HintGraph | COMPUTE | 0 |
| ImplGraph | Invalid | 1 |

---

## 设计考虑

### 1. 为什么移除 DataType 枚举?

**原因**:
- MLIR 已有完善的类型系统,重复定义会增加维护负担
- 类型推断(Type Inference)在 MLIR 中是标准流程
- 使用内置类型可直接复用 MLIR 的类型验证和转换基础设施

**代价**:
- 特殊类型(如量化类型 `QINT8`)需要单独定义 MLIR 类型
- 转换时需要显式映射 protobuf dtype 到 MLIR 类型

### 2. 为什么融合 MemAttr, MemQueueAttr, MemBufAttr?

**原因**:
- 这三个属性在实际使用中高度关联,分离定义增加复杂度
- 大部分字段(如 `name`, `buf_ids`)在编译流程中未使用
- 简化后的属性足以支持内存分配和优化

### 3. 为什么移除 SchedInfoDef 和 ApiInfoDef?

**原因**:
- `exec_order`: MLIR 基本块已保证拓扑序
- `exec_condition`: 可在优化 Pass 中根据操作模式推导
- `api.type/compute_type/unit`: 与操作类型一一对应,冗余

---

## 转换工具实现指南

### 推荐转换流程

```python
def convert_ascgraph_to_afir(ascgraph: AscGraphDef) -> mlir.Module:
    # 1. 创建 Module 并附加图属性
    module = create_module_with_graph_attr(ascgraph.asc_graph_attr)

    # 2. 构建节点依赖图
    node_map = build_dependency_graph(ascgraph.asc_node)

    # 3. 按拓扑序生成操作
    for node in topological_sort(ascgraph.asc_node):
        # 3.1 确定操作类型
        op_type = map_node_type_to_afir_op(node.attr.type)

        # 3.2 推断 MLIR 类型
        result_type = infer_mlir_type(node.outputs[0].attr.dtype,
                                       node.outputs[0].attr.axis_ids,
                                       node.outputs[0].attr.repeats)

        # 3.3 构建 indexing_maps
        indexing_maps = build_affine_maps(node.outputs[0].attr.axis_ids,
                                           node.outputs[0].attr.repeats,
                                           node.outputs[0].attr.strides)

        # 3.4 转换张量属性
        outputs_attr = convert_tensor_attr(node.outputs[0].attr)

        # 3.5 生成操作
        create_afir_op(op_type,
                       inputs=resolve_inputs(node.input_src, node_map),
                       result_type=result_type,
                       indexing_maps=indexing_maps,
                       loop_axis=node.attr.sched.loop_axis,
                       ir_attr_def=node.attr.ir_attr_def,
                       tmp_buffers=node.attr.tmp_buffers,
                       outputs=[outputs_attr])

    return module
```

### 关键辅助函数

#### 1. DataType → MLIR Type

```python
def map_dtype_to_mlir_type(dtype: int, shape: List[int]) -> str:
    type_map = {
        1: "f32", 2: "f16", 27: "bf16",
        3: "i8", 4: "ui8", 7: "i32", 8: "i64",
        11: "i1", 12: "f64"
    }
    elem_type = type_map.get(dtype, "f32")
    if shape:
        dims = "x".join(map(str, shape))
        return f"tensor<{dims}x{elem_type}>"
    else:
        return f"tensor<*x{elem_type}>"  # 未知维度
```

#### 2. axis_ids/repeats/strides → AffineMap

```python
def build_affine_map(axis_ids: List[int], repeats: List[str], strides: List[str]) -> str:
    # 示例: axis_ids=[0,1], repeats=["20","31"], strides=["31","1"]
    # → affine_map<(d0, d1) -> (d0 * 31 + d1)>

    # 简化版: 假设连续布局
    if all(int(s) == 1 for s in strides[-1:]) and len(axis_ids) == len(repeats):
        dims = ", ".join(f"d{i}" for i in range(len(axis_ids)))
        return f"affine_map<({dims}) -> ({dims})>"

    # 复杂情况需要解析 strides 和 repeats 表达式
    ...
```

#### 3. 内存属性融合

```python
def convert_tensor_attr(attr: AscTensorAttrGroupsDef) -> dict:
    # 融合 mem, que, buf 到单一属性
    position = map_position(attr.mem.position if attr.mem else 0,
                            attr.mem.alloc_type if attr.mem else 0)

    return {
        "vectorized_axis": attr.vectorized_axis,
        "vectorized_strides": attr.vectorized_strides,
        "tensor_id": attr.mem.tensor_id if attr.mem else -1,
        "reuse_id": attr.mem.reuse_id if attr.mem else -1,
        "position": position,
        "position_id": attr.que.id if attr.que else (attr.buf.id if attr.buf else -1),
        "depth": attr.que.depth if attr.que else None,
        "is_double_buffer": attr.que.buf_num > 1 if attr.que else None
    }

def map_position(proto_position: int, alloc_type: int) -> str:
    # Position 枚举融合逻辑
    if alloc_type == 1: return "L1"
    elif alloc_type == 2: return "L2"
    elif proto_position == 0: return "GM"
    elif proto_position == 1: return "VECTOR_IN"
    elif proto_position == 2: return "VECTOR_OUT"
    elif proto_position == 3: return "VECTOR_CALC"
    else: return "GM"
```

---

## 参考文件

- **AFIR 方言定义**: `include/Dialect/AFIR/AFIRAttrs.td`, `AFIREnums.td`, `AFIROps.td`
- **AscGraph Protobuf**: `graph_metadef/proto/ascendc_ir.proto`, `ge_ir.proto`
- **转换工具**: `python/ascir-to-afir/ascir_to_afir.py`

---

## 版本历史

- **v1.0 (2025-01)**: 初始 1:1 映射版本,保留所有 protobuf 结构
- **v2.0 (2026-01)**: 抽象版本,融合内存属性,使用 MLIR 类型系统

---

## 附录: 完整映射表

### Protobuf 消息 → AFIR 属性

| Protobuf Message | AFIR Attribute | 状态 |
|-----------------|----------------|------|
| `AscGraphAttrGroupsDef` | `AscGraphAttrGroups` | ✅ 1:1 映射 (简化 type) |
| `AxisDef` | `Axis` | ✅ 1:1 映射 (忽略 2 字段) |
| `AscTensorAttrGroupsDef` | `AscTensorGroups` | ⚠️ 简化映射 |
| `MemAttrDef` | (融合到 `AscTensorGroups`) | ⚠️ 部分映射 |
| `MemQueueAttrDef` | (融合到 `AscTensorGroups`) | ⚠️ 部分映射 |
| `MemBufAttrDef` | (融合到 `AscTensorGroups`) | ⚠️ 部分映射 |
| `MemOptAttrDef` | - | ❌ 不映射 |
| `TmpBufDescDef` | `TmpBufDesc` | ✅ 1:1 映射 |
| `TmpBufferGroupDef` | - | ❌ 移除 (直接使用 TmpBufDesc 数组) |
| `SchedInfoDef` | (拆解到操作属性) | ⚠️ 部分映射 |
| `ApiInfoDef` | - | ❌ 不映射 |
| `AscNodeAttrGroupsDef` | (拆解到操作属性) | ⚠️ 拆解 |
| `AscInputSourceDef` | (SSA value 引用) | ✅ 隐式映射 |
| `IrDef` | (类型推断) | ⚠️ 用于转换,不保留 |

### Protobuf 字段状态图例
- ✅ **1:1 映射**: 直接对应,无修改
- ⚠️ **部分映射**: 简化或融合到其他属性
- ❌ **不映射**: 移除或可推导
