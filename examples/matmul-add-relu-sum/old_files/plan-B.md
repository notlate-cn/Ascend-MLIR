# AscendCBufferPlacementPass Phase 1 开发文档

**版本**: v2.0（采用方案B：memref memory_space 标注）  
**适用编译器**: LLVM 21.1.8  
**目标平台**: Ascend 910B / 910C  
**文档状态**: 草稿，供 Agent 实现参考

---

## 目录

1. [背景](#1-背景)
2. [解决思路](#2-解决思路)
3. [实现方案](#3-实现方案)
4. [测试方案](#4-测试方案)
5. [附录：关键类型与 Op 参考](#5-附录关键类型与-op-参考)

---

## 1. 背景

### 1.1 编译流水线概览

本项目面向 AscendNPU（昇腾 AI 处理器）的 MLIR 编译流水线，目标是将高层 linalg-on-tensor 计算图自动生成符合 AscendC 编程模型的 kernel 代码。整体流水线分为以下阶段：

```
Step 1: fc_add_relu.mlir [在 examples/matmul-add-relu-sum 目录下]
        linalg-on-tensor 原始计算图
        (linalg.matmul + linalg.elementwise add/max)
        ↓
        transform_tile_and_fuse_3level.mlir [在 examples/matmul-add-relu-sum 目录下]
        Transform Dialect 调度脚本 [examples/matmul-add-relu-sum/run.sh 中的第一步，已完成，无报错]
        ↓
Step 2: TileAndFuse IR
        3层 scf.for 嵌套 + ascendc.* annotation
        (linalg-on-tensor + 循环结构)
        ↓
        One-Shot Bufferize  [examples/matmul-add-relu-sum/run.sh 中的第二步，已完成，无报错]
        ↓
Step 3: Bufferized IR                        ← Phase 1 的输入
        memref（无 memory_space）
        + scf.for + ascendc.* annotation
        ↓
        AscendCBufferPlacementPass Phase 1   ← 本文档描述的目标
        ↓
Step 4: Buffer-Placed IR                     ← Phase 1 的输出
        memref（带 #ascendc.space<*> memory_space）
        + 显式 ascendc.data_copy_* / ascendc.fixpipe op
        + 无任何 ascendc.* annotation
        ↓
        Phase 2: LinalgToAscendCPass
        读 memory_space → 转换为 ascendc.LocalTensor/GlobalTensor
        linalg.matmul → matmul 状态机 op 序列
        linalg.elementwise → ascendc Vector op
        插入 TQue EnQue/DeQue 同步
        ↓
        AscendC C++ kernel 代码
```

### 1.2 方案选型说明

**方案B 的核心思路**：用 `memref` 的标准 `memory_space` attribute 携带 TPosition 信息，类型系统不变，linalg op 无需任何修改，IR 始终合法。Phase 1 只专注于"知道数据在哪里"，Phase 2 专注于"用正确类型表达它"，两个 pass 职责严格分离。

```
Phase 1 职责：确定每个 buffer 的 TPosition，打在 memory_space 上，插入显式搬运 op
Phase 2 职责：读 memory_space，将 memref 转换为 ascendc.LocalTensor/GlobalTensor，lowering 计算 op
```

### 1.3 AscendNPU和Ascend C介绍 
#### 1.3.1 AscendNPU硬件架构模型
  https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0015.html
#### 1.3.2 Ascend C介绍
  Ascend C是CANN Kit针对算子开发场景推出的编程语言，遵循C和C++标准规范，匹配开发者开发习惯；通过多层接口抽象、自动并行计算、孪生调试等关键技术，提高算子开发效率，助力AI开发者低成本完成算子开发和模型调优部署。

  Ascend C基于MLIR开发了与Ascend C API一一对应的IR方言，项目介绍如下：
> 代码目录：externals/pyasc
> 
> 方言定义根目录：externals/pyasc/include/ascir/Dialect/Asc/IR
> 
> 框架定义：externals/pyasc/include/ascir/Dialect/Asc/IR/Fwk
> 
> 核心定义：externals/pyasc/include/ascir/Dialect/Asc/IR/Core
> 
> 基础Op定义：externals/pyasc/include/ascir/Dialect/Asc/IR/Basic
> 
> 高阶Op定义：externals/pyasc/include/ascir/Dialect/Asc/IR/Adv

### 1.3 AscendNPU 硬件内存层次

每个 AiCore 拥有独立的片上存储层次，各层对应 AscendC 的 TPosition 枚举：

```
AiCore
├── Cube 计算单元 (matmul)
├── Vector 计算单元 (add, max 等)
└── 片上存储
    ├── L1 Buffer
    │   ├── A1  (TPosition=1)  lhs 矩阵暂存，GM→L1
    │   └── B1  (TPosition=3)  rhs 矩阵暂存，GM→L1
    ├── L0A     (TPosition=2)  A2，A1→L0A，Cube lhs 输入
    ├── L0B     (TPosition=4)  B2，B1→L0B，Cube rhs 输入
    ├── L0C     (TPosition=7)  CO1，Cube 累加输出
    └── UB (Unified Buffer)
        ├── VECIN   (TPosition=9)   Vector Core 输入
        ├── VECCALC (TPosition=11)  Vector Core 中间计算
        └── VECOUT  (TPosition=10)  Vector Core 输出
```

TPosition 枚举定义于 `Core/Attributes.td` 的 `AscendC_TPositionAttr`：

```tablegen
def AscendC_TPositionAttr : I32EnumAttr<"TPosition", "queue/buffer position", [
  I32EnumAttrCase<"GM",      0, "gm">,
  I32EnumAttrCase<"A1",      1, "a1">,
  I32EnumAttrCase<"A2",      2, "a2">,
  I32EnumAttrCase<"B1",      3, "b1">,
  I32EnumAttrCase<"B2",      4, "b2">,
  I32EnumAttrCase<"CO1",     7, "co1">,
  I32EnumAttrCase<"VECIN",   9, "vecin">,
  I32EnumAttrCase<"VECOUT",  10, "vecout">,
  I32EnumAttrCase<"VECCALC", 11, "veccalc">,
]>
```

### 1.4 Transform 脚本的 Annotation 约定

Transform 脚本（v13）在 One-Shot Bufferize 之前向 IR 注入了以下 annotation，供本 pass 读取。**这些 annotation 在 Phase 1 结束时必须全部清除。**

| Annotation Key | 挂载位置 | 含义 | 示例值 |
|---|---|---|---|
| `ascendc.parallel` | scf.for | 该循环为分核循环，对应多 AiCore 并行 | `true` |
| `ascendc.prologue` | scf.for | 循环入口需执行的搬运任务列表 | `"lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"` |
| `ascendc.epilogue` | scf.for | 循环出口需执行的搬运任务列表 | `"acc:CO1->VECIN"` |
| `ascendc.unit` | linalg op | 执行该 op 的计算单元 | `"AiCore.Cube"` / `"AiCore.Vector"` |

**搬运任务格式**：`"角色:源->目标,角色:源->目标,..."`

- 角色（role）标识该 buffer 在计算中的语义，用于 def-use 链分析
- 源和目标使用 TPosition 名称或 `GM`
- 多个任务用逗号分隔

**完整的循环结构与 annotation 分布**（经 bufferize 后的输入 IR）：

```mlir
func.func @fc_relu(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                   %bias: memref<?x?xf32>, %C: memref<?x?xf32>) {
  scf.for %i_TB_M = ... {ascendc.parallel = true} {
    scf.for %i_TB_N = ... {
      ascendc.parallel = true,
      ascendc.prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN",
      ascendc.epilogue = "result:VECOUT->GM"
    } {
      %sv_A  = memref.subview %A[...]  // A 的 TB_M×K 切片
      %sv_B  = memref.subview %B[...]  // B 的 K×TB_N 切片
      %sv_C  = memref.subview %C[...]  // C 的 TB_M×TB_N 切片
      %sv_bias = memref.subview %bias[...]

      scf.for %i_Tb_M = ... {
        scf.for %i_Tb_N = ... {

          scf.for %i_K = ... {
            ascendc.prologue = "lhs:A1->A2,rhs:B1->B2",
            ascendc.epilogue = "acc:CO1->VECIN"
          } {
            %sv_A2 = memref.subview %sv_A[...]
            %sv_B2 = memref.subview %sv_B[...]
            %alloc_CO1 = memref.alloc() : memref<Tb_M x Tb_N x f32>

            linalg.matmul {ascendc.unit = "AiCore.Cube"}
                ins(%sv_A2, %sv_B2 : memref<...>, memref<...>)
                outs(%alloc_CO1 : memref<...>)
          }

          %alloc_veccalc = memref.alloc() : memref<Tb_M x Tb_N x f32>
          linalg.elementwise kind=add {ascendc.unit = "AiCore.Vector"}
              ins(%alloc_CO1_veced, %sv_bias2 : ...)
              outs(%alloc_veccalc : ...)

          %alloc_vecout = memref.alloc() : memref<Tb_M x Tb_N x f32>
          linalg.elementwise kind=max_signed {ascendc.unit = "AiCore.Vector"}
              ins(%alloc_veccalc, %zero : ...)
              outs(%alloc_vecout : ...)
        }
      }
    }
  }
}
```

### 1.5 Phase 1 的输出形态

Phase 1 输出的 IR 形态如下。**所有 annotation 已清除，所有 memref 已带 memory_space，显式搬运 op 已插入。**

```mlir
func.func @fc_relu(%A: memref<?x?xf32, #ascendc.space<gm>>,
                   %B: memref<?x?xf32, #ascendc.space<gm>>,
                   %bias: memref<?x?xf32, #ascendc.space<gm>>,
                   %C: memref<?x?xf32, #ascendc.space<gm>>) {
  scf.for %i_TB_M = ... {
    scf.for %i_TB_N = ... {

      // ── prologue: GM→A1, GM→B1, GM→VECIN ──────────────────────
      %alloc_A1   = memref.alloc() : memref<TB_M x K x f32, #ascendc.space<a1>>
      %alloc_B1   = memref.alloc() : memref<K x TB_N x f32, #ascendc.space<b1>>
      %alloc_bias_ub = memref.alloc() : memref<TB_M x TB_N x f32, #ascendc.space<vecin>>

      ascendc.data_copy_l2 %alloc_A1, %sv_A, %count_A1
          : memref<...,#ascendc.space<a1>>, memref<...,#ascendc.space<gm>>, i32
      ascendc.data_copy_l2 %alloc_B1, %sv_B, %count_B1
          : memref<...,#ascendc.space<b1>>, memref<...,#ascendc.space<gm>>, i32
      ascendc.data_copy_l2 %alloc_bias_ub, %sv_bias, %count_bias
          : memref<...,#ascendc.space<vecin>>, memref<...,#ascendc.space<gm>>, i32

      scf.for %i_Tb_M = ... {
        scf.for %i_Tb_N = ... {

          scf.for %i_K = ... {

            // ── prologue: A1→A2, B1→B2 ──────────────────────────
            %alloc_A2 = memref.alloc() : memref<Tb_M x t_K x f32, #ascendc.space<a2>>
            %alloc_B2 = memref.alloc() : memref<t_K x Tb_N x f32, #ascendc.space<b2>>

            ascendc.data_copy_l0 %alloc_A2, %sv_A1, %params_A
                : memref<...,#ascendc.space<a2>>, memref<...,#ascendc.space<a1>>, ...
            ascendc.data_copy_l0 %alloc_B2, %sv_B1, %params_B
                : memref<...,#ascendc.space<b2>>, memref<...,#ascendc.space<b1>>, ...

            %alloc_CO1 = memref.alloc() : memref<Tb_M x Tb_N x f32, #ascendc.space<co1>>

            // linalg op 保持不变，operand 类型已更新为带 memory_space 的 memref
            linalg.matmul
                ins(%alloc_A2, %alloc_B2 : memref<...,#ascendc.space<a2>>,
                                           memref<...,#ascendc.space<b2>>)
                outs(%alloc_CO1 : memref<...,#ascendc.space<co1>>)

            // ── epilogue: CO1→VECIN (Fixpipe) ───────────────────
            %alloc_vecin = memref.alloc() : memref<Tb_M x Tb_N x f32, #ascendc.space<vecin>>
            %fixpipe_params = ascendc.construct !ascendc.fixpipe_params<i32>()
            // %fixpipe_params = ascendc.data_copy_pad_ext_params : !ascendc.fixpipe_params<f32>
            ascendc.fixpipe %alloc_vecin, %alloc_CO1, %fixpipe_params
                : memref<...,#ascendc.space<vecin>>,
                  memref<...,#ascendc.space<co1>>,
                  !ascendc.fixpipe_params<f32>
          }

          // add 的 out → VECCALC（有后续 max 消费）
          %alloc_veccalc = memref.alloc() : memref<Tb_M x Tb_N x f32, #ascendc.space<veccalc>>
          linalg.elementwise kind=add
              ins(%alloc_vecin, %alloc_bias_ub : memref<...,#ascendc.space<vecin>>,
                                                  memref<...,#ascendc.space<vecin>>)
              outs(%alloc_veccalc : memref<...,#ascendc.space<veccalc>>)

          // max 的 out → VECOUT（无后续 Vector 消费，需写回 GM）
          %alloc_vecout = memref.alloc() : memref<Tb_M x Tb_N x f32, #ascendc.space<vecout>>
          linalg.elementwise kind=max_signed
              ins(%alloc_veccalc, %zero : memref<...,#ascendc.space<veccalc>>, ...)
              outs(%alloc_vecout : memref<...,#ascendc.space<vecout>>)
        }
      }

      // ── epilogue: VECOUT→GM ──────────────────────────────────
      ascendc.data_copy_l2 %sv_C, %alloc_vecout, %count_C
          : memref<...,#ascendc.space<gm>>, memref<...,#ascendc.space<vecout>>, i32
    }
  }
}
```

---

## 2. 解决思路

### 2.1 核心问题

One-Shot Bufferize 的输出中，所有 `memref.alloc` 都是普通的无 memory_space memref，没有任何 TPosition 信息。Phase 1 需要回答三个问题：

1. **哪个 alloc 应该分配到哪个 TPosition？**
2. **在哪里插入什么搬运指令？**
3. **搬运的数据量（size）如何确定？**

### 2.2 TPosition 推导规则

**不依赖循环深度**（同一层可能有多种 TPosition），通过两条规则组合推导。

**规则 A：从循环 prologue/epilogue 推导**

每层循环的 annotation 声明了该层入口/出口的搬运。搬运的目标 TPosition 即为该层新分配 buffer 的 memory_space：

```
for_TB_N.prologue = "lhs:GM->A1, rhs:GM->B1, bias:GM->VECIN"
→ 在 for_TB_N 入口为 lhs 新分配 memref<..., #ascendc.space<a1>>
  为 rhs 新分配 memref<..., #ascendc.space<b1>>
  为 bias 新分配 memref<..., #ascendc.space<vecin>>

for_K.prologue = "lhs:A1->A2, rhs:B1->B2"
→ 在 for_K 入口为 lhs 新分配 memref<..., #ascendc.space<a2>>
  为 rhs 新分配 memref<..., #ascendc.space<b2>>

for_K.epilogue = "acc:CO1->VECIN"
→ matmul 的 out alloc 标注为 #ascendc.space<co1>
  在 for_K 出口新分配 memref<..., #ascendc.space<vecin>> 并插入 Fixpipe
```

**规则 B：从 ascendc.unit + def-use 链推导 Vector op 的 out**

```
VECTOR op 的 out alloc：
  若该 alloc 的所有 use 仍在当前 AiCore 的 Vector 计算中（即有后续 linalg.elementwise 消费）
    → memory_space = #ascendc.space<veccalc>
  若该 alloc 无后续 Vector 计算消费者（即是最终输出，对应 epilogue result:VECOUT->GM）
    → memory_space = #ascendc.space<vecout>
```

对于 fc_add_relu 计算图：
- `add` 的 out：被 `max` 消费 → `veccalc`
- `max` 的 out：无后续 Vector 消费者 → `vecout`

**规则 C：matmul 的 out 固定为 CO1**

```
linalg.matmul {ascendc.unit = "AiCore.Cube"} 的 outs operand 对应的 alloc
→ memory_space = #ascendc.space<co1>（固定，不需要推导）
```

### 2.3 搬运指令选择
AscendC OP定义在 externals/pyasc/include/ascir/Dialect/Asc/IR/Basic 目录下，比如：
* externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpDataCopy.td
* externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpFixpipe.td

| 路径 | AscendC 通道 | 使用的 Op |
|---|---|---|
| GM → A1 / B1 | MTE2 | `ascendc.data_copy_l2` |
| GM → VECIN (bias) | MTE2 | `ascendc.data_copy_l2` |
| A1 → A2 | MTE1 | `ascendc.data_copy_l0` |
| B1 → B2 | MTE1 | `ascendc.data_copy_l0` |
| CO1 → VECIN | FIX (Fixpipe) | `ascendc.fixpipe` |
| VECOUT → GM | MTE3 | `ascendc.data_copy_l2` |

### 2.4 func 参数的 memory_space 处理

func 的参数（A、B、bias、C）来自 GM，需要将其类型从 `memref<?x?xf32>` 更新为 `memref<?x?xf32, #ascendc.space<gm>>`。

`memref.subview` 自动继承源 memref 的 memory_space，因此 subview 链下的所有切片均自动携带正确的 memory_space，无需逐一处理。

### 2.5 与业界实践的对比

方案B 与 TVM 的 `cache_read/cache_write` 思路完全对齐：

| 维度 | TVM | 本 Pass（方案B）|
|---|---|---|
| 内存位置表达 | buffer `scope` 参数 | `memref` 的 `memory_space` attribute |
| 搬运插入时机 | `compute_at` 指定循环层 | scf.for 的 prologue/epilogue annotation |
| 搬运表达方式 | `cache_read` op（一等公民）| 显式 `ascendc.data_copy_*` / `ascendc.fixpipe` op |
| 类型转换时机 | 同步完成 | 推迟到 Phase 2，Phase 1 只打 memory_space |

---

## 3. 实现方案
### 3.1 复用AscendC_TPositionAttr

**直接复用 `AscendC_TPositionAttr`**（定义于 `Core/Attributes.td`，cppNamespace `::mlir::ascendc`）作为 `memref`的 `memory_space` attribute 值。`TPositionAttr` 精确覆盖 GM/A1/A2/B1/B2/CO1/VECIN/VECCALC/VECOUT 所有需要区分的位置，且与 Phase 2 的 `TBuf<pos>` / `TQue<pos>` 参数类型完全一致，Phase 2 可直接读取无需任何转换。

### 3.2 Pass 定义

```cpp
// AscendCBufferPlacementPass.h
#pragma once
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::ascendc {

class AscendCBufferPlacementPass
    : public PassWrapper> {
public:
  StringRef getName() const override {
    return "AscendCBufferPlacementPass";
  }
  StringRef getArgument() const override {
    return "ascendc-buffer-placement";
  }
  StringRef getDescription() const override {
    return "Annotate memref allocs with TPosition memory_space and insert "
           "explicit DataCopy/Fixpipe ops based on ascendc.prologue/"
           "epilogue/unit annotations. All ascendc.* annotations are "
           "removed after this pass.";
  }
  void runOnOperation() override;
};

void registerAscendCBufferPlacementPass();

} // namespace mlir::ascendc
```

### 3.3 数据结构

```cpp
// 搬运任务：解析 prologue/epilogue 一条记录
struct CopyTask {
  StringRef role;    // "lhs", "rhs", "bias", "acc", "result"
  TPosition srcPos;  // 源位置（GM 或片上存储）
  TPosition dstPos;  // 目标位置
};

// 每层循环的搬运任务集合
struct LoopAnnotation {
  SmallVector prologue;
  SmallVector epilogue;
  bool isParallel = false;
};

// 每个 memref Value 对应的 TPosition
using BufferPosMap = DenseMap;
```

### 3.4 实现步骤

#### Step A：解析 Annotation

遍历所有 `scf::ForOp`，解析 annotation 字符串，构建 `loopAnnotations` 映射。

```cpp
// 解析 "lhs:GM->A1,rhs:GM->B1" 为 CopyTask 列表
SmallVector parseCopyTasks(StringRef s) {
  SmallVector tasks;
  SmallVector entries;
  s.split(entries, ',');
  for (auto entry : entries) {
    auto [role, path] = entry.trim().split(':');
    auto [src, dst]   = path.split("->");
    tasks.push_back({role.trim(),
                     parseTPosition(src.trim()),
                     parseTPosition(dst.trim())});
  }
  return tasks;
}

TPosition parseTPosition(StringRef name) {
  return StringSwitch(name)
    .Case("GM",      TPosition::GM)
    .Case("A1",      TPosition::A1)
    .Case("A2",      TPosition::A2)
    .Case("B1",      TPosition::B1)
    .Case("B2",      TPosition::B2)
    .Case("CO1",     TPosition::CO1)
    .Case("VECIN",   TPosition::VECIN)
    .Case("VECOUT",  TPosition::VECOUT)
    .Case("VECCALC", TPosition::VECCALC)
    .Default(TPosition::GM);
}
```

#### Step B：推导 BufferPosMap

遍历所有 `memref::AllocOp`，结合 loopAnnotations 和 `ascendc.unit`，确定每个 alloc 的 TPosition。

```cpp
void inferBufferPositions(func::FuncOp funcOp,
                          const DenseMap &loopAnns,
                          BufferPosMap &posMap) {

  // 规则 C：matmul 的 outs alloc → CO1
  funcOp.walk([&](linalg::MatmulOp matmulOp) {
    Value out = matmulOp.getDpsInitOperand(0)->get();
    if (auto alloc = out.getDefiningOp())
      posMap[alloc.getResult()] = TPosition::CO1;
  });

  // 规则 B：Vector op 的 outs alloc → VECCALC 或 VECOUT
  funcOp.walk([&](linalg::GenericOp genericOp) {
    auto unitAttr = genericOp->getAttrOfType("ascendc.unit");
    if (!unitAttr || unitAttr.getValue() != "AiCore.Vector") return;

    Value out = genericOp.getDpsInitOperand(0)->get();
    auto alloc = out.getDefiningOp();
    if (!alloc) return;

    // 检查是否有后续 Vector 计算消费者
    bool hasVectorConsumer = llvm::any_of(out.getUsers(), [](Operation *user) {
      if (auto g = dyn_cast(user))
        return g->hasAttrOfType("ascendc.unit") &&
               g->getAttrOfType("ascendc.unit").getValue()
                   == "AiCore.Vector";
      return false;
    });

    posMap[alloc.getResult()] = hasVectorConsumer ? TPosition::VECCALC
                                                  : TPosition::VECOUT;
  });

  // 规则 A：从 prologue/epilogue 推导 L1/L0/VECIN 的 alloc 位置
  // (这些 alloc 在 Step D 中新建，此处记录到 posMap 供后续引用)
  // 已在 Step D 中直接指定 memory_space，无需在此预推导
}
```

#### Step C：更新 func 参数类型（GM memory_space）

将函数所有 memref 类型的参数更新为带 `#ascendc.space<gm>` 的类型：

```cpp
void annotateGMArgs(func::FuncOp funcOp, OpBuilder &builder) {
  auto *ctx = funcOp.getContext();
  auto gmSpace = TPositionMemSpaceAttr::get(ctx, TPosition::GM);

  // 更新 function type
  SmallVector newArgTypes;
  for (auto argType : funcOp.getArgumentTypes()) {
    if (auto memrefType = dyn_cast(argType)) {
      newArgTypes.push_back(MemRefType::get(
          memrefType.getShape(), memrefType.getElementType(),
          memrefType.getLayout(), gmSpace));
    } else {
      newArgTypes.push_back(argType);
    }
  }
  // 更新 block argument 类型
  for (auto [arg, newType] : llvm::zip(funcOp.getArguments(), newArgTypes))
    arg.setType(newType);

  // 更新 FunctionType
  auto newFuncType = builder.getFunctionType(
      newArgTypes, funcOp.getFunctionType().getResults());
  funcOp.setFunctionType(newFuncType);

  // memref.subview 会自动继承 memory_space，无需手动更新
}
```

#### Step D：在循环边界插入搬运 op 并新建带 memory_space 的 alloc

这是 Phase 1 的核心步骤。对每个有 annotation 的 `scf::ForOp`，在其 prologue（循环体开头）和 epilogue（循环体末尾）分别插入搬运：

```cpp
void insertCopiesForLoop(scf::ForOp forOp,
                         const LoopAnnotation &ann,
                         const BufferPosMap &posMap,
                         OpBuilder &builder) {
  auto *ctx = forOp.getContext();
  Block &body = forOp.getRegion().front();

  // ── Prologue：在循环体第一条 op 前插入 ──────────────────────────
  builder.setInsertionPointToStart(&body);

  for (const auto &task : ann.prologue) {
    // 确定 src Value（GM memref）和 dst shape
    Value srcMemref = findSourceMemref(task.role, forOp);
    MemRefType dstType = computeLocalMemrefType(
        ctx, srcMemref, task.dstPos);

    // 新建带 memory_space 的 alloc
    Value localAlloc = builder.create(
        forOp.getLoc(), dstType).getResult();

    // 插入搬运 op
    insertDataCopy(builder, forOp.getLoc(),
                   localAlloc, srcMemref, task.srcPos, task.dstPos);

    // 记录到 posMap（供后续 linalg op 的 operand 替换使用）
    // posMap[localAlloc] = task.dstPos;  // 已编码在 memory_space 中
  }

  // ── Epilogue：在循环体最后一条 op 前插入 ────────────────────────
  builder.setInsertionPoint(body.getTerminator());

  for (const auto &task : ann.epilogue) {
    if (task.srcPos == TPosition::CO1 && task.dstPos == TPosition::VECIN) {
      // CO1 → VECIN: Fixpipe
      Value co1Buf   = findAllocWithSpace(forOp, TPosition::CO1);
      MemRefType vecinType = computeLocalMemrefType(
          ctx, co1Buf, TPosition::VECIN);
      Value vecinAlloc = builder.create(
          forOp.getLoc(), vecinType).getResult();
      insertFixpipe(builder, forOp.getLoc(), vecinAlloc, co1Buf);

    } else if (task.dstPos == TPosition::GM) {
      // VECOUT → GM: DataCopy
      Value vecoutBuf = findAllocWithSpace(forOp, TPosition::VECOUT);
      Value gmDst     = findSourceMemref("result", forOp);
      insertDataCopy(builder, forOp.getLoc(),
                     gmDst, vecoutBuf, TPosition::VECOUT, TPosition::GM);
    }
  }
}

// 根据搬运方向选择正确的 DataCopy op
void insertDataCopy(OpBuilder &builder, Location loc,
                    Value dst, Value src,
                    TPosition srcPos, TPosition dstPos) {
  auto *ctx = builder.getContext();

  if ((srcPos == TPosition::GM) ||
      (srcPos == TPosition::VECOUT && dstPos == TPosition::GM)) {
    // GM ↔ UB/L1：DataCopyL2（MTE2/MTE3 通道）
    Value count = computeElementCount(builder, loc, dst);
    builder.create(loc, dst, src, count);

  } else if (srcPos == TPosition::A1 || srcPos == TPosition::B1) {
    // L1 → L0：DataCopyL0（MTE1 通道）
    auto params = buildDataCopyParams(builder, loc, dst);
    builder.create(loc, dst, src, params);
  }
}

// 插入 Fixpipe（CO1 → VECIN）
void insertFixpipe(OpBuilder &builder, Location loc,
                   Value vecinDst, Value co1Src) {
  auto *ctx = builder.getContext();
  //auto paramsType = FixpipeParamsType::get(ctx,
      //cast(co1Src.getType()).getElementType());
  //Value params = builder.create(loc, paramsType);
  auto paramsType = FixpipeParamsType::get(ctx,
    cast<MemRefType>(co1Src.getType()).getElementType());
	Value params = builder.create<ConstructOp>(loc, paramsType);
  // workspace 为空（无量化）
  builder.create(loc, vecinDst, co1Src,
                             /*workspace=*/Value{}, params);
}
```

#### Step E：更新现有 memref.alloc 的 memory_space

对 `posMap` 中已推导出 TPosition 的 alloc（CO1、VECCALC、VECOUT），在原 alloc 上直接修改返回类型（或替换为新的带 memory_space 的 alloc）：

```cpp
void updateAllocMemorySpace(const BufferPosMap &posMap, OpBuilder &builder) {
  for (auto [allocResult, pos] : posMap) {
    auto allocOp = allocResult.getDefiningOp();
    if (!allocOp) continue;

    auto oldType = cast(allocResult.getType());
    auto newSpace = TPositionMemSpaceAttr::get(
        allocOp.getContext(), pos);
    auto newType = MemRefType::get(
        oldType.getShape(), oldType.getElementType(),
        oldType.getLayout(), newSpace);

    // 在原位置创建新 alloc，替换旧 alloc 的所有 use
    builder.setInsertionPoint(allocOp);
    auto newAlloc = builder.create(
        allocOp.getLoc(), newType);
    allocResult.replaceAllUsesWith(newAlloc.getResult());
    allocOp.erase();
  }
}
```

#### Step F：清除所有 ascendc.* Annotation

```cpp
void clearAnnotations(func::FuncOp funcOp) {
  funcOp.walk([](Operation *op) {
    SmallVector toRemove;
    for (NamedAttribute attr : op->getAttrs())
      if (attr.getName().getValue().starts_with("ascendc."))
        toRemove.push_back(attr.getName());
    for (StringAttr name : toRemove)
      op->removeAttr(name);
  });
}
```

#### Step G：runOnOperation 主流程

```cpp
void AscendCBufferPlacementPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  auto *ctx = &getContext();
  OpBuilder builder(ctx);

  // A: 解析所有循环的 annotation
  DenseMap loopAnns;
  funcOp.walk([&](scf::ForOp forOp) {
    LoopAnnotation ann;
    if (auto a = forOp->getAttrOfType("ascendc.prologue"))
      ann.prologue = parseCopyTasks(a.getValue());
    if (auto a = forOp->getAttrOfType("ascendc.epilogue"))
      ann.epilogue = parseCopyTasks(a.getValue());
    ann.isParallel = forOp->hasAttr("ascendc.parallel");
    if (!ann.prologue.empty() || !ann.epilogue.empty() || ann.isParallel)
      loopAnns[forOp] = std::move(ann);
  });

  // B: 推导现有 alloc 的 TPosition（CO1, VECCALC, VECOUT）
  BufferPosMap posMap;
  inferBufferPositions(funcOp, loopAnns, posMap);

  // C: 更新 func 参数为 GM memory_space
  annotateGMArgs(funcOp, builder);

  // D: 插入搬运 op，新建带 memory_space 的 L1/L0/VECIN alloc
  // 按从外到内的顺序处理，保证 alloc 在正确的循环层创建
  funcOp.walk([&](scf::ForOp forOp) {
    auto it = loopAnns.find(forOp);
    if (it != loopAnns.end())
      insertCopiesForLoop(forOp, it->second, posMap, builder);
  });

  // E: 更新现有 alloc 的 memory_space（CO1, VECCALC, VECOUT）
  updateAllocMemorySpace(posMap, builder);

  // F: 清除所有 ascendc.* annotation
  clearAnnotations(funcOp);

  // G: 验证 IR 合法性
  if (failed(verify(funcOp)))
    signalPassFailure();
}
```

### 3.5 文件结构

```
lib/Transforms/AscendCBufferPlacement/
├── CMakeLists.txt
├── AscendCBufferPlacementPass.h          // Pass 声明
├── AscendCBufferPlacementPass.cpp        // runOnOperation 主流程（Step G）
├── AnnotationParser.h/.cpp               // Step A: annotation 解析
├── TPositionInference.h/.cpp             // Step B: posMap 推导
├── GMArgAnnotator.h/.cpp                 // Step C: func 参数 memory_space
├── CopyInserter.h/.cpp                   // Step D: 搬运 op 插入
└── AllocUpdater.h/.cpp                   // Step E: alloc memory_space 更新

include/ascendc/Core/
└── MemorySpace.td                        // 新增 TPositionMemSpaceAttr 定义
```

### 3.6 依赖的 dialect 和 op

| 依赖                                | 用途                                           |
|-----------------------------------|----------------------------------------------|
| `mlir/Dialect/SCF/IR/SCF.h`       | `scf::ForOp`                                 |
| `mlir/Dialect/MemRef/IR/MemRef.h` | `memref::AllocOp`, `memref::SubViewOp`       |
| `mlir/Dialect/Linalg/IR/Linalg.h` | `linalg::MatmulOp`, `linalg::GenericOp`      |
| `mlir/Dialect/Func/IR/FuncOps.h`  | `func::FuncOp`                               |
| `Basic/OpDataCopy.td`和`OpFixpipe.td` | `DataCopyL2Op`, `DataCopyL0Op`, `FixpipeOp`等 |
| `Core/Attributes.td`              | `TPosition`, `TPositionMemSpaceAttr`（新增）     |

---

## 4. 测试方案

### 4.1 单元测试（FileCheck 格式）

位置：`test/Transforms/AscendCBufferPlacement/`

运行命令：
```bash
mlir-opt %s --ascendc-buffer-placement | FileCheck %s
```

#### 测试 1：GM 参数类型更新

```mlir
// test_gm_args.mlir
// CHECK: func.func @test(%{{.*}}: memref<?x?xf32, #ascendc.space<gm>>
func.func @test(%A: memref<?x?xf32>, %B: memref<?x?xf32>) {
  return
}
```

#### 测试 2：prologue GM→L1 搬运（for_TB_N）

```mlir
// test_prologue_gm_l1.mlir
func.func @test(%A: memref<128x256xf32>, %B: memref<256x128xf32>,
                %bias: memref<128x128xf32>, %C: memref<128x128xf32>) {
  scf.for %i = %c0 to %c128 step %c64 {
    scf.for %j = %c0 to %c128 step %c64 {
      // CHECK: memref.alloc() : memref<{{.*}}, #ascendc.space<a1>>
      // CHECK: memref.alloc() : memref<{{.*}}, #ascendc.space<b1>>
      // CHECK: memref.alloc() : memref<{{.*}}, #ascendc.space<vecin>>
      // CHECK: ascendc.data_copy_l2
      // CHECK: ascendc.data_copy_l2
      // CHECK: ascendc.data_copy_l2
      // CHECK-NOT: ascendc.prologue
    } { ascendc.parallel = true,
        ascendc.prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN",
        ascendc.epilogue = "result:VECOUT->GM" }
  } { ascendc.parallel = true }
  return
}
```

#### 测试 3：prologue L1→L0 搬运 + epilogue Fixpipe（for_K）

```mlir
// test_for_k.mlir
// CHECK: memref.alloc() : memref<{{.*}}, #ascendc.space<a2>>
// CHECK: memref.alloc() : memref<{{.*}}, #ascendc.space<b2>>
// CHECK: ascendc.data_copy_l0  ← A1→A2
// CHECK: ascendc.data_copy_l0  ← B1→B2
// ... linalg.matmul ...
// CHECK: memref.alloc() : memref<{{.*}}, #ascendc.space<vecin>>
// CHECK: ascendc.fixpipe        ← CO1→VECIN
// CHECK-NOT: ascendc.prologue
// CHECK-NOT: ascendc.epilogue
```

#### 测试 4：matmul out 分配为 CO1

```mlir
// test_matmul_co1.mlir
// CHECK: memref.alloc() : memref<{{.*}}, #ascendc.space<co1>>
// CHECK: linalg.matmul
// CHECK-SAME: outs(%{{.*}} : memref<{{.*}}, #ascendc.space<co1>>)
```

#### 测试 5：Vector op out 推导（VECCALC vs VECOUT）

```mlir
// test_vector_tposition.mlir
// add 的 out 有 max 消费者 → VECCALC
// CHECK: memref.alloc() : memref<{{.*}}, #ascendc.space<veccalc>>
// CHECK: linalg.elementwise{{.*}}kind{{.*}}add
// CHECK-SAME: outs(%{{.*}} : memref<{{.*}}, #ascendc.space<veccalc>>)

// max 的 out 无 Vector 消费者 → VECOUT
// CHECK: memref.alloc() : memref<{{.*}}, #ascendc.space<vecout>>
// CHECK: linalg.elementwise{{.*}}kind{{.*}}max_signed
// CHECK-SAME: outs(%{{.*}} : memref<{{.*}}, #ascendc.space<vecout>>)
```

#### 测试 6：epilogue VECOUT→GM 写回

```mlir
// test_epilogue_vecout_gm.mlir
// CHECK: ascendc.data_copy_l2 %{{.*}}#ascendc.space<gm>
// CHECK-SAME:                  %{{.*}}#ascendc.space<vecout>
```

#### 测试 7：annotation 全部清除

```mlir
// test_annotation_cleanup.mlir
// CHECK-NOT: ascendc.prologue
// CHECK-NOT: ascendc.epilogue
// CHECK-NOT: ascendc.parallel
// CHECK-NOT: ascendc.unit
```

#### 测试 8：linalg op 保持不变（Phase 1 不 lower 计算 op）

```mlir
// test_linalg_preserved.mlir
// CHECK: linalg.matmul
// CHECK: linalg.elementwise{{.*}}add
// CHECK: linalg.elementwise{{.*}}max_signed
```

### 4.2 集成测试

端到端验证完整 fc_add_relu 计算图：

```bash
# 完整流水线
mlir-opt fc_add_relu.mlir \
  --transform-interpreter=transform-library=transform_tile_and_fuse_3level.mlir \
  -o step3_annotated.mlir

mlir-opt step3_annotated.mlir \
  --one-shot-bufferize="bufferize-function-boundaries=true \
                         allow-return-allocs-from-loops=true" \
  --buffer-deallocation-pipeline \
  -o step3_bufferized.mlir

mlir-opt step3_bufferized.mlir \
  --ascendc-buffer-placement \
  --verify-each \
  -o step3_buffer_placed.mlir

# 验证输出
echo "=== 搬运 op 数量 ==="
grep -c "data_copy_l2" step3_buffer_placed.mlir   # 期望 >= 4（GM→A1/B1/VECIN + VECOUT→GM）
grep -c "data_copy_l0" step3_buffer_placed.mlir   # 期望 >= 2（A1→A2, B1→B2）
grep -c "ascendc.fixpipe" step3_buffer_placed.mlir # 期望 >= 1（CO1→VECIN）

echo "=== memory_space 标注 ==="
grep -c "#ascendc.space"     step3_buffer_placed.mlir  # 期望 >= 1
grep -c "#ascendc.space"     step3_buffer_placed.mlir  # 期望 >= 1
grep -c "#ascendc.space"    step3_buffer_placed.mlir  # 期望 >= 1
grep -c "#ascendc.space"  step3_buffer_placed.mlir  # 期望 >= 2
grep -c "#ascendc.space" step3_buffer_placed.mlir # 期望 >= 1
grep -c "#ascendc.space" step3_buffer_placed.mlir  # 期望 >= 1
grep -c "#ascendc.space"     step3_buffer_placed.mlir  # 期望 >= 4（func 参数）

echo "=== annotation 清除 ==="
grep "ascendc.prologue" step3_buffer_placed.mlir  # 期望无输出
grep "ascendc.epilogue" step3_buffer_placed.mlir  # 期望无输出
grep "ascendc.unit"     step3_buffer_placed.mlir  # 期望无输出

echo "=== 计算 op 保持 ==="
grep -c "linalg.matmul"      step3_buffer_placed.mlir  # 期望 >= 1
grep -c "linalg.elementwise" step3_buffer_placed.mlir  # 期望 >= 2
```

### 4.3 验证 Checklist

| 检查项 | 验证方法 | 期望结果 |
|---|---|---|
| func 参数有 `#ascendc.space<gm>` | FileCheck | 所有 memref 参数带 gm space |
| for_TB_N prologue 产生 3 条 DataCopyL2 | FileCheck | GM→A1, GM→B1, GM→VECIN |
| for_TB_N epilogue 产生 1 条 DataCopyL2 | FileCheck | VECOUT→GM |
| for_K prologue 产生 2 条 DataCopyL0 | FileCheck | A1→A2, B1→B2 |
| for_K epilogue 产生 1 条 Fixpipe | FileCheck | CO1→VECIN |
| matmul outs 有 `#ascendc.space<co1>` | FileCheck | `memref<..., #ascendc.space<co1>>` |
| add outs 有 `#ascendc.space<veccalc>` | FileCheck | `memref<..., #ascendc.space<veccalc>>` |
| max outs 有 `#ascendc.space<vecout>` | FileCheck | `memref<..., #ascendc.space<vecout>>` |
| 无 annotation 残留 | grep | 0 条匹配 |
| linalg op 保持完整 | FileCheck | matmul 和 elementwise 仍存在 |
| `--verify-each` 无报错 | mlir-opt | 0 errors |
| subview 继承正确 memory_space | FileCheck | subview 的 source/result 类型一致 |

### 4.4 边界情况测试

| 场景 | 测试目的 |
|---|---|
| tile size 为动态值（`?`）| calCount 计算正确处理动态 shape |
| t_K = 整个 K 轴（for_K 只迭代一次）| for_K 退化时 prologue/epilogue 仍正确插入 |
| bias 的 shape 与输出 C 不同（广播场景）| GM→VECIN 的 count 基于 bias 实际 shape 计算 |
| max 的 ins[1] 是 zero tensor（linalg.fill 产生）| zero tensor 不分配 GM 搬运，由 Phase 2 处理为标量 |
| for_TB_M/TB_N 被折叠为单次迭代 | parallel 标注的循环优化后 pass 仍鲁棒 |
| 多个 func（multi-function module）| pass 以 func 为粒度运行，互不影响 |

---

## 5. 附录：关键类型与 Op 参考

### 5.1 新增：TPositionMemSpaceAttr

```tablegen
// 新增于 Core/MemorySpace.td
def AscendC_TPositionMemSpaceAttr
    : AscendC_Attr<"TPositionMemSpace", "space"> {
  let parameters = (ins "TPositionAttr":$position);
  let assemblyFormat = "`<` custom<PrettyTPosition>($position) `>`";
}
// 使用: memref<128x64xf32, #ascendc.space<a2>>
```

### 5.2 DataCopyL2（GM ↔ UB/L1）

```tablegen
// OpDataCopy.td
def AscendC_DataCopyL2Op : DataCopyOp<"data_copy_l2", "DataCopy", [AscFunc]> {
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AnyType:$calCount);
}
```

用于路径：GM→A1, GM→B1, GM→VECIN（bias），VECOUT→GM。

### 5.3 DataCopyL0（L1 ↔ L0）

```tablegen
def AscendC_DataCopyL0Op : DataCopyOp<"data_copy_l0", "DataCopy", [AscFunc]> {
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AscendC_DataCopyParams:$repeatParams);
}
```

用于路径：A1→A2，B1→B2。

### 5.4 Fixpipe（CO1 → VECIN）

```tablegen
// OpFixpipe.td
def AscendC_FixpipeOp : APIOp<"fixpipe", "Fixpipe", [AscFunc]> {
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_LocalTensor:$src,
                       Optional<AscendC_LocalTensor>:$workspace,
                       AnyType:$intriParams);
}
// Types.td
def AscendC_FixpipeParams : AscendC_Type<"FixpipeParams", "fixpipe_params"> {
  let parameters = (ins "Type":$type);
}
```

用法：
```mlir
// %params = ascendc.data_copy_pad_ext_params : !ascendc.fixpipe_params<f32>
%params = ascendc.construct !ascendc.fixpipe_params<f32>()
ascendc.fixpipe %vecin_buf, %co1_buf, %params
    : memref<..., #ascendc.space<vecin>>,
      memref<..., #ascendc.space<co1>>,
      !ascendc.fixpipe_params<f32>
```

### 5.5 Phase 2 接口约定

Phase 1 输出给 Phase 2 的 IR 约定：

| IR 元素 | 约定 |
|---|---|
| func 参数 | 全部带 `#ascendc.space<gm>`，由 Phase 2 转为 `GlobalTensor` |
| 片上 alloc | 全部带 `#ascendc.space<*>`，由 Phase 2 转为 `LocalTensor`（通过 `tbuf.get_tensor`） |
| linalg.matmul | 保留，operand 带 memory_space，由 Phase 2 展开为状态机 op 序列 |
| linalg.elementwise | 保留，由 Phase 2 lower 为 ascendc Vector op |
| 搬运 op | 已为显式 ascendc op，Phase 2 只需处理其 operand 的类型转换 |
| annotation | 全部清除，Phase 2 通过读 memory_space 获取位置信息 |

---

*文档结束*