# AFIR Dialect 文件组织结构

## 文件依赖图

```
┌──────────────────────────────────────────────────────────────────┐
│                      MLIR Standard Headers                        │
│  (mlir/IR/OpBase.td, mlir/IR/AttrTypeBase.td, etc.)             │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  AFIRBase.td   │ ◄─── 基础层
                    │                │
                    │ - Dialect      │
                    │ - Base Classes │
                    │ - DataType     │
                    │ - Predicates   │
                    └────┬──────┬────┘
                         │      │
              ┌──────────┘      └──────────┐
              │                            │
              ▼                            ▼
    ┌─────────────────┐          ┌──────────────────┐
    │ AFIRTypes.td    │          │  AFIRAttrs.td    │ ◄─── 定义层
    │                 │          │                  │
    │ - Custom Types  │          │ - Op Attrs       │
    │   (reserved)    │          │ - Graph Attrs    │
    └─────────────────┘          │ - AscGraph Defs  │
                                 └──────────────────┘
                                          │
              ┌───────────────────────────┴───────────────────────┐
              │                                                   │
              ▼                                                   │
    ┌──────────────────┐                                         │
    │ AFIRDialect.td   │ ◄─── 入口层                             │
    │                  │                                          │
    │ - Include Base   │                                          │
    │ - Include Types  │                                          │
    │ - Include Attrs  │                                          │
    └────────┬─────────┘                                          │
             │                                                    │
             └────────────────────┬───────────────────────────────┘
                                  ▼
                        ┌──────────────────┐
                        │   AFIROps.td     │ ◄─── 操作层
                        │                  │
                        │ - Operations     │
                        │ - Traits         │
                        │ - Verifiers      │
                        └──────────────────┘
```

## 文件详细说明

### 1. AFIRBase.td
**职责**：基础定义和通用组件

**包含内容**：
- ✅ `AFIR_Dialect` - 方言定义
- ✅ `AFIR_Op` - 操作基类
- ✅ `AFIR_Type` - 类型基类
- ✅ `AFIR_Attr` - 属性基类
- ✅ `AFIR_DataTypeEnum` - 数据类型枚举（41种）
- ✅ 类型谓词（AFIR_Tensor, AFIR_Float, 等）

**依赖**：
```
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
```

**被依赖**：AFIRTypes.td, AFIRAttrs.td, AFIRDialect.td

---

### 2. AFIRTypes.td
**职责**：自定义类型定义

**包含内容**：
- 📝 预留自定义类型空间
- 📝 当前使用标准 MLIR TensorType

**为什么保留这个文件？**
虽然当前文件几乎为空（仅17行），但我们选择保留它，原因如下：
1. **符合 MLIR 标准实践**：Types、Attrs、Ops 应该分开
2. **预留扩展空间**：未来可能需要自定义类型
3. **清晰的文件职责**：保持文件结构的一致性和可预测性
4. **最小成本**：保留一个17行的文件成本极低

**依赖**：
```
include "Dialect/AFIR/AFIRBase.td"
include "mlir/IR/AttrTypeBase.td"
```

**未来扩展**：
- QuantizedTensorType - 量化张量类型
- SparseTensorType - 稀疏张量类型
- DeviceTensorType - 设备特定张量类型
- StreamType - 流式计算类型

---

### 3. AFIRAttrs.td
**职责**：所有属性定义（操作级 + 图级）

**为什么不拆分这个文件？**
虽然当前文件有379行，但我们选择不拆分，原因如下：
1. **文件大小合理**：379行在可接受范围内（参考：MLIR的ArithAttrs.td ~400行）
2. **职责清晰**：所有属性定义都与 AscGraph 映射相关，拆分会增加文件跳转
3. **维护成本**：目前无操作级属性，拆分意义不大
4. **未来可扩展**：当文件超过500行或添加大量操作级属性时再考虑拆分

**包含内容**：

#### 图级属性（AscGraph 1:1 映射）：
- ✅ `AFIR_MemAttr` - 内存管理
- ✅ `AFIR_MemQueueAttr` - 内存队列
- ✅ `AFIR_MemBufAttr` - 内存缓冲区
- ✅ `AFIR_MemOptAttr` - 内存优化
- ✅ `AFIR_AscTensorAttrGroups` - 张量属性组
- ✅ `AFIR_AxisAttr` - 轴定义
- ✅ `AFIR_SchedInfo` - 调度信息
- ✅ `AFIR_ApiInfo` - API 信息
- ✅ `AFIR_TmpBufDesc` - 临时缓冲区描述
- ✅ `AFIR_TmpBufferGroup` - 临时缓冲区组
- ✅ `AFIR_AscNodeAttrGroups` - 节点属性组
- ✅ `AFIR_AscGraphAttrGroups` - 图属性组
- ✅ `AFIR_AscInputSource` - 输入源引用
- ✅ `AFIR_AscTensor` - 张量定义
- ✅ `AFIR_IrDef` - IR 定义
- ✅ `AFIR_AscNode` - 节点定义
- ✅ `AFIR_AscGraph` - 图定义（顶层）

**依赖**：
```
include "Dialect/AFIR/AFIRBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
```

**文件大小**：~523 行

---

### 4. AFIRDialect.td
**职责**：方言入口文件

**包含内容**：
- Include 指令的集中管理

**结构**：
```tablegen
include "Dialect/AFIR/AFIRBase.td"    // 基础定义
include "Dialect/AFIR/AFIRTypes.td"   // 类型定义
include "Dialect/AFIR/AFIRAttrs.td"   // 属性定义
```

**说明**：这是使用 AFIR 方言时唯一需要 include 的文件

---

### 5. AFIROps.td
**职责**：操作定义

**包含内容**：
- ✅ `AFIR_BinaryOp` - 二元操作基类
  - 统一的 lhs/rhs 参数
  - Pure trait（无副作用）
  - SameOperandsAndResultType（类型约束）
  - Elementwise trait（元素级操作）
- ✅ `AFIR_AddOp` - 加法操作（支持 Commutative）
- ✅ `AFIR_SubOp` - 减法操作
- ✅ `AFIR_MulOp` - 乘法操作（支持 Commutative）
- ✅ `AFIR_DivOp` - 除法操作

**设计要点**：
1. **SSA 形式**：每个操作产生一个 SSA Value
2. **DAG 结构**：Value 可以被多个操作使用，形成 DAG
3. **广播语义**：支持标准的张量广播
4. **类型安全**：使用 SameOperandsAndResultType 确保类型一致

**依赖**：
```
include "Dialect/AFIR/AFIRDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
```

**示例用法**：
```mlir
// DAG 示例：%0 被多个操作使用
func.func @example(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.constant dense<1.0> : tensor<4x4xf32>
  %1 = afir.add %arg0, %0 : tensor<4x4xf32>  // 使用 %arg0
  %2 = afir.mul %arg0, %0 : tensor<4x4xf32>  // 再次使用 %arg0 - DAG!
  %3 = afir.add %1, %2 : tensor<4x4xf32>
  return %3 : tensor<4x4xf32>
}
```

---

## 使用指南

### 如何使用这些文件

#### 在其他 TableGen 文件中使用 AFIR：
```tablegen
// 只需 include 方言文件即可
include "Dialect/AFIR/AFIRDialect.td"

// 现在可以使用所有 AFIR 定义
def MyOp : AFIR_Op<"my_op"> {
  let arguments = (ins AFIR_Tensor:$input);
}
```

#### 在 C++ 中使用：
```cpp
#include "Dialect/AFIR/AFIRDialect.h"
#include "Dialect/AFIR/AFIROps.h"

// 使用 AFIR 方言
mlir::afir::AFIRDialect *dialect = ...;
mlir::afir::AscGraphAttr graphAttr = ...;
```

---

## 属性定义查找表

| Proto Message | MLIR Attribute | 文件位置 |
|---------------|----------------|----------|
| `ge.proto.DataType` | `AFIR_DataTypeEnum` | AFIRBase.td |
| `ge.proto.MemAttrDef` | `AFIR_MemAttr` | AFIRAttrs.td |
| `ge.proto.MemQueueAttrDef` | `AFIR_MemQueueAttr` | AFIRAttrs.td |
| `ge.proto.MemBufAttrDef` | `AFIR_MemBufAttr` | AFIRAttrs.td |
| `ge.proto.MemOptAttrDef` | `AFIR_MemOptAttr` | AFIRAttrs.td |
| `ge.proto.AscTensorAttrGroupsDef` | `AFIR_AscTensorAttrGroups` | AFIRAttrs.td |
| `ge.proto.AxisDef` | `AFIR_AxisAttr` | AFIRAttrs.td |
| `ge.proto.SchedInfoDef` | `AFIR_SchedInfo` | AFIRAttrs.td |
| `ge.proto.ApiInfoDef` | `AFIR_ApiInfo` | AFIRAttrs.td |
| `ge.proto.TmpBufDescDef` | `AFIR_TmpBufDesc` | AFIRAttrs.td |
| `ge.proto.TmpBufferGroupDef` | `AFIR_TmpBufferGroup` | AFIRAttrs.td |
| `ge.proto.AscNodeAttrGroupsDef` | `AFIR_AscNodeAttrGroups` | AFIRAttrs.td |
| `ge.proto.AscGraphAttrGroupsDef` | `AFIR_AscGraphAttrGroups` | AFIRAttrs.td |
| `ascendc_ir.proto.AscInputSourceDef` | `AFIR_AscInputSource` | AFIRAttrs.td |
| `ascendc_ir.proto.AscTensorDef` | `AFIR_AscTensor` | AFIRAttrs.td |
| `ascendc_ir.proto.IrDef` | `AFIR_IrDef` | AFIRAttrs.td |
| `ascendc_ir.proto.AscNodeDef` | `AFIR_AscNode` | AFIRAttrs.td |
| `ascendc_ir.proto.AscGraphDef` | `AFIR_AscGraph` | AFIRAttrs.td |

---

## 文件统计

| 文件 | 行数 | 大小 | 主要内容 |
|------|------|------|----------|
| AFIRBase.td | ~175 | 7.8 KB | 方言定义 + DataType |
| AFIRTypes.td | ~81 | 2.5 KB | 类型定义（预留） |
| AFIRAttrs.td | ~523 | 17.9 KB | 所有属性定义 |
| AFIRDialect.td | ~24 | 0.8 KB | 入口文件 |
| AFIROps.td | ~108 | 3.3 KB | 操作定义 |
| **总计** | **~911** | **~32 KB** | |

---

## 维护指南

### 添加新的属性：
1. 确定属性类别（操作级/图级）
2. 在 `AFIRAttrs.td` 中添加定义
3. 按照现有格式添加注释和映射信息

### 添加新的类型：
1. 在 `AFIRTypes.td` 中定义
2. 可能需要在 `AFIRBase.td` 中添加相应的谓词

### 添加新的操作：
1. 在 `AFIROps.td` 中定义
2. 使用 `AFIRDialect.td` 中的属性和类型

### 添加新的基础枚举：
1. 在 `AFIRBase.td` 中定义
2. 确保其他文件可以直接使用

---

## 最佳实践

1. ✅ **单一入口**：外部只 include AFIRDialect.td
2. ✅ **清晰分层**：Base → Types/Attrs → Dialect → Ops
3. ✅ **详细注释**：每个定义都有 summary 和 description
4. ✅ **Proto 映射**：每个图级属性都注明对应的 proto message
5. ✅ **格式一致**：使用统一的 assemblyFormat
6. ✅ **命名规范**：AFIR_ 前缀 + 描述性名称

---

## 架构设计文档

关于 AFIR 的核心架构决策（特别是 DAG vs AST 的设计选择），请参考：
- **ARCHITECTURE.md** - 详细解释了为什么 MLIR 原生支持 DAG，以及 AscGraph 到 MLIR 的映射策略

**核心结论**：
- ✅ MLIR 使用 SSA 形式，天然就是 DAG 结构
- ✅ AscGraph 的 DAG 可以直接映射为 MLIR SSA，无需转换
- ✅ AFIR 采用操作级表示（推荐），每个 AscGraph 节点映射为一个 MLIR Operation

---

## 相关文档

- `ARCHITECTURE.md` - AFIR 架构设计和 DAG/AST 分析
- `AscGraph_Mapping.md` - AscGraph Proto 到 MLIR 的详细映射
- `RESTRUCTURE_NOTES.md` - 文件重构历史和原因
