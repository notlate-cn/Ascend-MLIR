# AFIR Dialect 架构设计文档

## 概述

AFIR (Ascend Frontend IR) 是为 Ascend 硬件设计的高级方言，它提供了从高级 ML 框架到硬件特定 IR 的桥梁。本文档解释了 AFIR 的核心架构决策，特别是关于计算图表示的设计选择。

---

## MLIR 的表示方式：SSA + DAG

MLIR 使用 SSA (Static Single Assignment) 形式，**天然就是 DAG 结构**。

### 什么是 SSA？

SSA (Static Single Assignment) 是一种 IR 形式，其中：
- **每个变量只被赋值一次**
- **每个值都有唯一的定义点**
- **通过 Use-Def 链隐式表示数据依赖**

### SSA 如何形成 DAG？

在 SSA 形式中：
1. 每个操作产生一个或多个 **Value**
2. 每个 Value 可以被**多个操作使用**
3. 通过 Value 的使用关系自动形成 **DAG**

### 示例对比

#### AST (树形结构)
```
表达式: (a + b) * (a - b)

    *
   / \
  +   -
 / \ / \
a  b a  b
```
注意：`a` 和 `b` 必须在树中**重复出现**（因为树的节点只能有一个父节点）

#### MLIR SSA (DAG 结构)
```mlir
func.func @example(%a: f32, %b: f32) -> f32 {
  %0 = arith.addf %a, %b : f32    // a + b
  %1 = arith.subf %a, %b : f32    // a - b
  %2 = arith.mulf %0, %1 : f32    // (%0) * (%1)
  return %2 : f32
}
```

**数据流图 (DAG)**：
```
    %a      %b
    / \    / \
   /   \  /   \
  /     \/     \
 |      /\      |
 |     /  \     |
 v    v    v    v
 [addf]   [subf]
    \      /
     v    v
     [mulf]
       |
       v
    return
```

注意：`%a` 和 `%b` 被**多个操作使用**，形成了 DAG！

---

## AscGraph 到 MLIR 的映射

### AscGraph 结构 (Proto)

```protobuf
message AscGraphDef {
  AscGraphAttrGroupsDef asc_graph_attr = 1;
  repeated AscNodeDef asc_node = 2;        // 节点列表
  string graph_name = 3;
}

message AscNodeDef {
  repeated AscInputSourceDef input_src = 1;  // 输入边
  repeated AscTensorDef outputs = 2;         // 输出
  AscNodeAttrGroupsDef attr = 3;
  IrDef ir_def = 4;
}

message AscInputSourceDef {
  string src_node_name = 1;      // 源节点名称
  int32 src_out_index = 2;       // 源节点的输出索引
}
```

**特点**：
- 显式存储节点列表
- 显式存储边的连接（src_node_name + src_out_index）
- DAG 结构

### MLIR 表示 (SSA)

```mlir
func.func @graph(%input: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.constant dense<1.0> : tensor<4x4xf32>

  // %input 被两个操作使用 - DAG
  %1 = afir.add %input, %0 : tensor<4x4xf32>
  %2 = afir.mul %input, %0 : tensor<4x4xf32>

  // 汇聚
  %3 = afir.add %1, %2 : tensor<4x4xf32>

  return %3 : tensor<4x4xf32>
}
```

**特点**：
- 隐式存储边（通过 Value 引用）
- 不需要节点名称（通过 SSA Value 标识）
- **同样是 DAG 结构**

### 映射对照表

| AscGraph 概念 | MLIR 概念 | 说明 |
|---------------|-----------|------|
| `AscGraphDef` | `func.func` 或 `Region` | 整个计算图 |
| `AscNodeDef` | `Operation` | 计算节点 |
| `AscInputSourceDef` | `Value` (SSA) | 数据依赖边 |
| `src_node_name` | 不需要（通过 Value 自动关联） | MLIR 更简洁 |
| `src_out_index` | Value 索引（如果多输出） | 自动管理 |

---

## AFIR 的两种表示方式

### 方式1：操作级表示（推荐）⭐️

将 AscGraph 的每个节点映射为一个 MLIR Operation。

**示例**：
```mlir
// AscGraph with 3 nodes: Input -> Add -> Output
func.func @example(%input: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %const = afir.constant dense<1.0> : tensor<4x4xf32>
  %result = afir.add %input, %const : tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}
```

**优点**：
- ✅ 符合 MLIR 习惯
- ✅ 可以直接使用 MLIR 的优化 Pass
- ✅ 易于逐步 lowering 到硬件指令
- ✅ 编译器可以自动推导数据依赖

**缺点**：
- ❌ 丢失了 AscGraph 的一些元数据（节点名称等）

**解决方案**：
```mlir
// 通过属性保留元数据
%result = afir.add %input, %const {
  node_name = "add_node_1",
  node_attr = #afir.asc_node<...>
} : tensor<4x4xf32>
```

### 方式2：属性级表示（保留完整结构）

将整个 AscGraph 作为一个属性附加到特殊操作上。

**示例**：
```mlir
%result = afir.graph(%input) {
  graph_attr = #afir.graph<
    asc_graph_attr = #afir.asc_graph<
      tiling_key = 12345,
      axis = [...],
      type = 0,
      size_var = ["N", "M"]
    >,
    asc_node = [
      #afir.node<
        input_src = [...],
        outputs = [...],
        attr = ...,
        ir_def = ...
      >,
      ...
    ],
    graph_name = "my_graph"
  >
} : (tensor<4x4xf32>) -> tensor<4x4xf32>
```

**优点**：
- ✅ 完整保留 AscGraph 的所有信息
- ✅ 1:1 映射，便于序列化/反序列化

**缺点**：
- ❌ 作为 "黑盒"，编译器难以优化
- ❌ 需要专门的 Pass 来解包展开

**使用场景**：
- 需要完整保留 Proto 结构
- 作为中间转换格式
- 在特定 Pass 中展开成操作级表示

---

## 业界最佳实践

### TensorFlow MLIR

**TensorFlow 计算图 → MLIR**

```python
# TensorFlow 代码
import tensorflow as tf

@tf.function
def my_func(x, y):
    return x + y
```

**生成的 MLIR（简化）**：
```mlir
func.func @my_func(%x: tensor<?xf32>, %y: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tf.AddV2"(%x, %y) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
```

**关键点**：TensorFlow 的 DAG 计算图直接映射为 MLIR 的 SSA 形式，**无需转换**。

### PyTorch MLIR (Torch-MLIR)

**PyTorch JIT 图 → MLIR**

```python
import torch

@torch.jit.script
def my_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = x + y
    return z * z  # z 被使用两次 - DAG
```

**生成的 MLIR（简化）**：
```mlir
func.func @my_func(%x: !torch.tensor, %y: !torch.tensor) -> !torch.tensor {
  %0 = torch.aten.add %x, %y : !torch.tensor
  %1 = torch.aten.mul %0, %0 : !torch.tensor  // %0 被使用两次 - DAG
  return %1 : !torch.tensor
}
```

**关键点**：`%0` 被 `mul` 使用两次，形成 DAG 结构。

### ONNX MLIR

**ONNX 图 → MLIR**

ONNX 本身就是 DAG，直接映射为 MLIR SSA。

```mlir
func.func @onnx_graph(%input: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
  %0 = "onnx.Conv"(%input, %weights) : (...) -> tensor<1x64x112x112xf32>
  %1 = "onnx.Relu"(%0) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
  ...
}
```

---

## AFIR 架构决策

### 推荐的转换流程

```
AscGraph (Proto)
      ↓
  [Proto Reader]
      ↓
AFIR Operations (SSA/DAG) ← 主要工作在这里
      ↓
  [Optimization Passes]
      ↓
Lower-level Dialect (如 ASC-IR)
      ↓
  [Codegen]
      ↓
Ascend Binary
```

---

## 实现示例

### 从 AscGraph Proto 转换为 AFIR

**输入 Proto**：
```protobuf
AscGraphDef {
  graph_name: "example"
  asc_node: [
    {
      name: "const1"
      outputs: [{dtype: DT_FLOAT, ...}]
    },
    {
      name: "add1"
      input_src: [
        {src_node_name: "input", src_out_index: 0},
        {src_node_name: "const1", src_out_index: 0}
      ]
      outputs: [{dtype: DT_FLOAT, ...}]
    }
  ]
}
```

**生成的 MLIR**：
```mlir
func.func @example(%input: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // 节点 const1
  %const = afir.constant dense<1.0> {node_name = "const1"} : tensor<4x4xf32>

  // 节点 add1 - 自动通过 SSA Value 连接
  %result = afir.add %input, %const {node_name = "add1"} : tensor<4x4xf32>

  return %result : tensor<4x4xf32>
}
```

**转换逻辑**：
```cpp
// 伪代码
void convertAscGraphToMLIR(AscGraphDef& graph, mlir::OpBuilder& builder) {
  // 1. 创建节点名到 Value 的映射
  llvm::DenseMap<std::string, mlir::Value> nodeMap;

  // 2. 按拓扑顺序遍历节点
  for (auto& node : graph.asc_node()) {
    // 3. 查找输入 Value
    SmallVector<Value> inputs;
    for (auto& inputSrc : node.input_src()) {
      Value inputValue = nodeMap[inputSrc.src_node_name()];
      inputs.push_back(inputValue);
    }

    // 4. 创建 Operation
    auto op = builder.create<afir::AddOp>(inputs[0], inputs[1]);

    // 5. 保存输出到映射
    nodeMap[node.name()] = op.getResult();
  }
}
```

---

## DAG 的优势示例

### 示例：共享子表达式

**代码**：
```python
# Python 伪代码
def compute(x, y):
    temp = x + y       # temp 被计算一次
    result1 = temp * 2  # 使用 temp
    result2 = temp * 3  # 再次使用 temp
    return result1 + result2
```

**MLIR DAG 表示**：
```mlir
func.func @compute(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
  %temp = afir.add %x, %y : tensor<4xf32>

  // temp 被两个操作使用 - DAG 结构
  %c2 = arith.constant dense<2.0> : tensor<4xf32>
  %c3 = arith.constant dense<3.0> : tensor<4xf32>
  %r1 = afir.mul %temp, %c2 : tensor<4xf32>
  %r2 = afir.mul %temp, %c3 : tensor<4xf32>

  %result = afir.add %r1, %r2 : tensor<4xf32>
  return %result : tensor<4xf32>
}
```

**数据流图**：
```
    %x    %y
     \    /
      \  /
      [add] ← %temp
      /   \
     /     \
 [mul c2] [mul c3]
    |       |
   %r1     %r2
     \     /
      \   /
      [add]
        |
     %result
```

**如果是 AST**（每次都重新计算）：
```
     [add]           [add]
     /   \           /   \
   [x]   [y]       [x]   [y]
    |               |
 [mul c2]        [mul c3]
    |               |
   %r1             %r2
    \              /
     \            /
         [add]
           |
        %result
```

注意：在 AST 中，`x + y` 需要计算两次！

---

## 参考资料

- MLIR Language Reference: https://mlir.llvm.org/docs/LangRef/
- SSA Form: https://en.wikipedia.org/wiki/Static_single-assignment_form
- TensorFlow MLIR: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/mlir
- Torch-MLIR: https://github.com/llvm/torch-mlir
- ONNX-MLIR: https://github.com/onnx/onnx-mlir
