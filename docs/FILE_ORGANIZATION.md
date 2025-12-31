# AFIR Dialect 文件组织结构

## 文件依赖图

```
┌──────────────────────────────────────────────────────────────────┐
│                      MLIR Standard Headers                        │
│  (mlir/IR/OpBase.td, mlir/IR/AttrTypeBase.td, mlir/IR/EnumAttr.td)│
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  AFIRBase.td   │ ◄─── 基础层
                    │                │
                    │ - Dialect      │
                    │ - AFIR_Op      │
                    │ - AFIR_Type    │
                    │ - AFIR_Attr    │
                    └────┬──────┬────┘
                         │      │
              ┌──────────┘      └──────────┐
              │                            │
              ▼                            ▼
    ┌─────────────────┐          ┌──────────────────┐
    │  AFIREnums.td   │          │  AFIRAttrs.td    │ ◄─── 定义层
    │                 │          │                  │
    │ - DataType      │          │ - MemAttr        │
    │ - AllocType     │          │ - AxisAttr       │
    │ - Position      │          │ - SchedInfo      │
    │ - Hardware      │          │ - ApiInfo        │
    │ - AxisType      │          │ - AscGraph...    │
    │ - ApiType       │          │                  │
    │ - ComputeType   │          │ (使用枚举类型    │
    │ - ComputeUnit   │          │  作为参数)       │
    │ - ...           │          │                  │
    └─────────────────┘          └──────────────────┘
              │                            │
              │    ┌───────────────────────┘
              │    │
              ▼    ▼
    ┌──────────────────┐
    │ AFIRDialect.td   │ ◄─── 入口层
    │                  │
    │ - Include Base   │
    │ - Include Attrs  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │   AFIROps.td     │ ◄─── 操作层
    │                  │
    │ - Include 子目录  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │  AFIROps/Math/Elementwise.td │ ◄─── 具体操作
    │                              │
    │ - AddOp, SubOp, MulOp, DivOp │
    └──────────────────────────────┘
```

## 关键设计决策

### 枚举与属性分离

为避免 C++ 类重复定义问题，枚举定义和属性定义**必须**分离到不同文件：

| 文件 | TableGen 处理方式 | 生成产物 |
|------|------------------|----------|
| AFIREnums.td | `-gen-enum-decls/defs` | AFIREnums.h.inc, AFIREnums.cpp.inc |
| AFIRAttrs.td | `-gen-attrdef-decls/defs` | AFIRAttrs.h.inc, AFIRAttrs.cpp.inc |

**重要**：`AFIRAttrs.td` 不能 include `AFIREnums.td`，否则会导致重复定义。两者通过 C++ 类型名（如 `"DataTypeAttr"`）在参数中引用。

---

## 文件详细说明

### 1. AFIRBase.td
**职责**：基础定义和通用组件

**包含内容**：
- `AFIR_Dialect` - 方言定义
- `AFIR_Op` - 操作基类
- `AFIR_Type` - 类型基类（预留）
- `AFIR_Attr` - 属性基类

**依赖**：
```tablegen
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
```

**被依赖**：AFIREnums.td, AFIRAttrs.td

---

### 2. AFIREnums.td
**职责**：所有枚举类型定义

**包含内容**：

| 枚举 | 说明 | 值数量 |
|------|------|--------|
| `AFIR_DataTypeEnum` | 数据类型（匹配 ge.proto.DataType） | 41 |
| `AFIR_AllocTypeEnum` | 内存分配类型 | 5 |
| `AFIR_PositionEnum` | 内存位置 | 4 |
| `AFIR_HardwareEnum` | 硬件目标 | 2 |
| `AFIR_AxisTypeEnum` | 轴类型 | 7 |
| `AFIR_ExecuteConditionEnum` | 执行条件 | 4 |
| `AFIR_ApiTypeEnum` | API 类型 | 3 |
| `AFIR_ComputeUnitEnum` | 计算单元 | 8 |
| `AFIR_ComputeTypeEnum` | 计算类型 | 12 |
| `AFIR_AscGraphTypeEnum` | 图类型 | 2 |

**依赖**：
```tablegen
include "Dialect/AFIR/AFIRBase.td"
include "mlir/IR/EnumAttr.td"
```

**生成的 C++ 类型**（在 AFIREnums.h.inc 中）：
- `enum class DataType { ... }`
- `class DataTypeAttr : public IntegerAttr { ... }`
- 其他枚举类似...

---

### 3. AFIRAttrs.td
**职责**：所有 AttrDef 属性定义（图级属性）

**包含内容**：

#### 内存管理属性：
| 属性 | Proto 映射 | 说明 |
|------|------------|------|
| `AFIR_MemAttr` | ge.proto.MemAttrDef | 内存分配属性 |
| `AFIR_MemQueueAttr` | ge.proto.MemQueueAttrDef | 内存队列属性 |
| `AFIR_MemBufAttr` | ge.proto.MemBufAttrDef | 内存缓冲区属性 |

#### 张量属性：
| 属性 | Proto 映射 | 说明 |
|------|------------|------|
| `AFIR_AscTensorAttrGroups` | ge.proto.AscTensorAttrGroupsDef | 张量属性组 |
| `AFIR_AscTensor` | ascendc_ir.proto.AscTensorDef | 张量定义 |

#### 轴和调度属性：
| 属性 | Proto 映射 | 说明 |
|------|------------|------|
| `AFIR_AxisAttr` | ge.proto.AxisDef | 轴定义 |
| `AFIR_SchedInfo` | ge.proto.SchedInfoDef | 调度信息 |
| `AFIR_ApiInfo` | ge.proto.ApiInfoDef | API 信息 |

#### 缓冲区属性：
| 属性 | Proto 映射 | 说明 |
|------|------------|------|
| `AFIR_TmpBufDesc` | ge.proto.TmpBufDescDef | 临时缓冲区描述 |
| `AFIR_TmpBufferGroup` | ge.proto.TmpBufferGroupDef | 临时缓冲区组 |

#### 节点和图属性：
| 属性 | Proto 映射 | 说明 |
|------|------------|------|
| `AFIR_AscNodeAttrGroups` | ge.proto.AscNodeAttrGroupsDef | 节点属性组 |
| `AFIR_AscGraphAttrGroups` | ge.proto.AscGraphAttrGroupsDef | 图属性组 |
| `AFIR_AscInputSource` | ascendc_ir.proto.AscInputSourceDef | 输入源引用 |
| `AFIR_IrDef` | ascendc_ir.proto.IrDef | IR 定义 |
| `AFIR_AscNode` | ascendc_ir.proto.AscNodeDef | 节点定义 |
| `AFIR_AscGraph` | ascendc_ir.proto.AscGraphDef | 图定义（顶层） |

**依赖**：
```tablegen
include "Dialect/AFIR/AFIRBase.td"
include "mlir/IR/AttrTypeBase.td"
```

**注意**：参数中使用枚举类型时，使用 C++ 类型名（如 `"DataTypeAttr"`），而非 TableGen 定义名（如 `AFIR_DataTypeAttr`）。

---

### 4. AFIRDialect.td
**职责**：方言入口文件

**结构**：
```tablegen
include "Dialect/AFIR/AFIRBase.td"    // 基础定义
include "Dialect/AFIR/AFIRAttrs.td"   // 属性定义
```

**说明**：这是使用 AFIR 方言时主要 include 的文件

---

### 5. AFIROps.td
**职责**：操作定义入口

**结构**：
```tablegen
include "Dialect/AFIR/AFIRDialect.td"
include "Interface/ShapeHelperOpInterface.td"
include "Interface/ShapeInferenceOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Core/Types.td"

// 包含具体操作定义
include "Dialect/AFIR/AFIROps/Math/Elementwise.td"
```

---

### 6. AFIROps/Math/Elementwise.td
**职责**：元素级数学操作

**包含操作**：
| 操作 | 说明 | Traits |
|------|------|--------|
| `AFIR_AddOp` | 元素级加法 | Pure, ShapeHelper, ShapeInference |
| `AFIR_SubOp` | 元素级减法 | Pure, ShapeHelper, ShapeInference |
| `AFIR_MulOp` | 元素级乘法 | Pure, ShapeHelper, ShapeInference |
| `AFIR_DivOp` | 元素级除法 | Pure, ShapeHelper, ShapeInference |

**设计特点**：
- 支持形状推断接口（ShapeInferenceOpInterface）
- 支持形状辅助接口（ShapeHelperOpInterface）
- 使用自定义验证器（hasVerifier = 1）

---

## C++ 头文件包含顺序

在 `AFIRDialect.h` 中，头文件必须按以下顺序包含：

```cpp
#include "mlir/IR/Dialect.h"

// 1. 方言定义
#include "Dialect/AFIR/AFIRDialect.h.inc"

// 2. 枚举定义（必须在属性之前！）
#include "Dialect/AFIR/AFIREnums.h.inc"

// 3. 属性定义
#define GET_ATTRDEF_CLASSES
#include "Dialect/AFIR/AFIRAttrs.h.inc"
```

---

## 文件统计

| 文件 | 行数 | 主要内容 |
|------|------|----------|
| AFIRBase.td | 86 | 方言和基类定义 |
| AFIREnums.td | 256 | 所有枚举定义 |
| AFIRAttrs.td | 354 | 图级属性定义 |
| AFIRDialect.td | 21 | 入口文件 |
| AFIROps.td | 26 | 操作入口 |
| AFIROps/Math/Elementwise.td | 188 | 元素级操作 |
| **总计** | **931** | |

---

## 目录结构

```
include/Dialect/AFIR/
├── AFIRBase.td                    # 基础定义
├── AFIREnums.td                   # 枚举定义
├── AFIRAttrs.td                   # 属性定义
├── AFIRDialect.td                 # 方言入口
├── AFIROps.td                     # 操作入口
├── AFIRDialect.h                  # C++ 头文件
├── AFIROps.h                      # 操作 C++ 头文件
├── AFIRDialectBuilder.h           # Builder 辅助类
├── AFIROps/                       # 操作子目录
│   └── Math/
│       └── Elementwise.td         # 元素级操作
├── Transforms/                    # 变换 Pass
│   ├── Passes.td
│   └── Passes.h
├── CMakeLists.txt
├── FILE_ORGANIZATION.md           # 本文件
└── ARCHITECTURE.md                # 架构设计文档
```

---

## 维护指南

### 添加新的枚举：
1. 在 `AFIREnums.td` 中定义枚举和 EnumAttr
2. 在 C++ 中通过生成的类名使用（如 `DataTypeAttr`）

### 添加新的属性：
1. 在 `AFIRAttrs.td` 中使用 `AFIR_Attr<>` 定义
2. 枚举类型参数使用 C++ 类型名（不带 `AFIR_` 前缀）
3. 示例：`"DataTypeAttr":$dtype`（而非 `AFIR_DataTypeAttr`）

### 添加新的操作：
1. 在 `AFIROps/` 对应子目录中创建 .td 文件
2. 在 `AFIROps.td` 中添加 include
3. 实现必要的接口方法

### 常见问题：

**Q: 为什么会出现 "redefinition of class" 错误？**

A: 通常是因为 AFIRAttrs.td 错误地 include 了 AFIREnums.td。两者必须独立处理。

**Q: 为什么属性参数中枚举类型要用字符串形式？**

A: 因为 TableGen 的 AttrDef 参数期望的是 C++ 类型名。使用 `"DataTypeAttr"` 会在生成的 C++ 代码中正确引用枚举属性类。

---

## 相关文档

- `ARCHITECTURE.md` - AFIR 架构设计和 DAG/AST 分析
- `AscGraph_Mapping.md` - AscGraph Proto 到 MLIR 的详细映射
