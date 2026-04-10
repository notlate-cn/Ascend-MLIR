# AscendCBufferPlacementPass Phase 1 开发文档

**版本**: v1.0
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
Step 1: fc_add_relu.mlir
        linalg-on-tensor 原始计算图
        (linalg.matmul + linalg.elementwise add/max)
        ↓
        transform_tile_and_fuse_3level.mlir  [v13-final]
        Transform Dialect 调度脚本
        ↓
Step 2: Step3 IR
        3层 scf.for 嵌套 + ascendc.* annotation
        (linalg-on-tensor + 循环结构)
        ↓
        One-Shot Bufferize  [已完成，无报错]
        ↓
Step 3: Bufferized IR                        ← 当前阶段的输入
        memref + scf.for + ascendc.* annotation
        ↓
        AscendCBufferPlacementPass Phase 1   ← 本文档描述的目标
        ↓
Step 4: Buffer-Placed IR                     ← 本文档描述的输出
        ascendc.TBuf/TQue 分配
        显式 ascendc.DataCopy / ascendc.Fixpipe op
        memref 语义消除
        ↓
        Phase 2: LinalgToAscendCPass
        (matmul 状态机展开, Vector op lowering, TQue 同步插入)
        ↓
        AscendC C++ kernel 代码
```

### 1.2 AscendNPU 硬件内存层次

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

### 1.3 Transform 脚本的 Annotation 约定

Transform 脚本（v13）在 One-Shot Bufferize 之前向 IR 中注入了以下 annotation，供本 pass 读取：

| Annotation Key | 挂载位置 | 含义 | 示例值 |
|---|---|---|---|
| `ascendc.parallel` | scf.for | 该循环为分核循环，对应多 AiCore 并行 | `true` |
| `ascendc.prologue` | scf.for | 该循环入口需执行的搬运任务列表 | `"lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"` |
| `ascendc.epilogue` | scf.for | 该循环出口需执行的搬运任务列表 | `"acc:CO1->VECIN"` |
| `ascendc.unit` | linalg op | 执行该 op 的计算单元 | `"AiCore.Cube"` / `"AiCore.Vector"` |

**搬运任务格式**：`"角色:源->目标,角色:源->目标,..."`

- 角色（role）标识该 buffer 在计算中的语义，用于 def-use 链分析
- 源和目标使用 TPosition 名称或 `GM`
- 多个任务用逗号分隔

**完整的循环结构与 annotation 分布**（经 bufferize 后）：

```mlir
scf.for %TB_M ... {ascendc.parallel = true} {
  scf.for %TB_N ... {
    ascendc.parallel = true,
    ascendc.prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN",
    ascendc.epilogue = "result:VECOUT->GM"
  } {
    // memref.subview 切片操作（tiling 产生）
    scf.for %Tb_M ... {
      scf.for %Tb_N ... {
        scf.for %K ... {
          ascendc.prologue = "lhs:A1->A2,rhs:B1->B2",
          ascendc.epilogue = "acc:CO1->VECIN"
        } {
          linalg.matmul {ascendc.unit = "AiCore.Cube"}
              ins(%sv_A, %sv_B : memref<...>, memref<...>)
              outs(%sv_C : memref<...>)
        }
        linalg.elementwise kind=add {ascendc.unit = "AiCore.Vector"}
        linalg.elementwise kind=max_signed {ascendc.unit = "AiCore.Vector"}
      }
    }
  }
}
```

### 1.4 Phase 1 的目标范围

Phase 1（本文档）只负责 **buffer 分配与搬运 op 插入**，不负责计算 op 的 lowering。

**Phase 1 输出的 IR 特征**：
- 所有 `memref.alloc` 被替换为带 TPosition 的 `ascendc.tbuf` 或 `ascendc.queue` 分配
- 在正确位置插入显式的 `ascendc.data_copy_l2` / `ascendc.fixpipe` op
- `linalg.matmul` 和 `linalg.elementwise` 保持原样（留给 Phase 2 处理）
- 所有 `ascendc.*` annotation 被清除
- `memref` 操作数类型保持不变（类型转换留给 Phase 2）

---

## 2. 解决思路

### 2.1 核心问题

One-Shot Bufferize 的输出中，所有 `memref.alloc` 都是普通的无类型 memref，没有任何 TPosition 信息。Phase 1 需要回答三个问题：

1. **哪个 alloc 应该分配到哪个 TPosition？**
2. **在哪里插入什么搬运指令？**
3. **搬运的数据量（size）如何确定？**

### 2.2 推导策略

**问题1：TPosition 推导**

不依赖循环深度（同一层可能有多种 TPosition），而是通过两条规则的组合推导：

**规则 A（循环 prologue/epilogue）**：每层循环的 annotation 声明了该层入口/出口需要的搬运。搬运的目标 TPosition 即为该层 buffer 的分配位置。

```
for_TB_N.prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"
→ 在 for_TB_N 内为 lhs 分配 TBuf<A1>，为 rhs 分配 TBuf<B1>，为 bias 分配 TBuf<VECIN>

for_K.prologue = "lhs:A1->A2,rhs:B1->B2"
→ 在 for_K 内为 lhs 分配 TBuf<A2>，为 rhs 分配 TBuf<B2>

for_K.epilogue = "acc:CO1->VECIN"
→ CO1 由 matmul out 产生（固定），目标 VECIN 已在 for_TB_N 层分配
```

**规则 B（ascendc.unit + def-use 链）**：对于 Vector op 的输出，通过分析 def-use 链确定是 VECCALC 还是 VECOUT：

```
VECTOR op 的 out buffer：
  若该 buffer 的所有 use 仍在当前 AiCore 的 Vector 计算中
    → 分配为 VECCALC（UB 内中转，不需要写回）
  若该 buffer 的 use 包含写回 GM（epilogue 中的 result:VECOUT->GM）
    → 分配为 VECOUT（最终输出，需要 DataCopy 写回 GM）
```

**问题2：搬运指令选择**

| 路径 | 指令 | Op |
|---|---|---|
| GM → A1 / B1 | DataCopy (MTE2) | `ascendc.data_copy_l2` |
| GM → VECIN (bias) | DataCopy (MTE2) | `ascendc.data_copy_l2` |
| A1 → A2 | DataCopy (MTE1) | `ascendc.data_copy_l0` |
| B1 → B2 | DataCopy (MTE1) | `ascendc.data_copy_l0` |
| CO1 → VECIN | FixpipeOp (MTE3/FIX) | `ascendc.fixpipe` |
| VECOUT → GM | DataCopy (MTE3) | `ascendc.data_copy_l2` |

**问题3：搬运 size 确定**

通过分析 `memref.subview` 操作的 sizes 参数，在 bufferize 后的 IR 中直接读取每次循环迭代的 tile size，作为 DataCopy 的 `calCount` 参数。

### 2.3 与业界实践的对比

本 pass 的职责与 TVM 的 `cache_read/cache_write` 等价：

| 对比维度 | TVM | 本 Pass |
|---|---|---|
| 内存位置来源 | `scope` 参数 | `ascendc.prologue/epilogue` + `ascendc.unit` |
| 搬运插入时机 | `compute_at` 指定 | scf.for 的循环入口/出口 |
| 搬运表达方式 | `cache_read` op | 显式 `ascendc.data_copy_*` / `ascendc.fixpipe` op |

---

## 3. 实现方案

### 3.1 Pass 定义

在 MLIR 的 Pass 框架中注册该 pass：

```cpp
// AscendCBufferPlacementPass.h
class AscendCBufferPlacementPass
    : public PassWrapper<AscendCBufferPlacementPass, OperationPass<func::FuncOp>> {
public:
  StringRef getName() const override { return "AscendCBufferPlacementPass"; }
  StringRef getArgument() const override { return "ascendc-buffer-placement"; }
  StringRef getDescription() const override {
    return "Assign TPosition to memref allocs and insert explicit DataCopy/Fixpipe ops "
           "based on ascendc.prologue/epilogue/unit annotations.";
  }
  void runOnOperation() override;
};
```

注册：

```cpp
void registerAscendCBufferPlacementPass() {
  PassRegistration<AscendCBufferPlacementPass>();
}
```

### 3.2 数据结构

```cpp
// 搬运任务：解析 prologue/epilogue annotation 的一条记录
struct CopyTask {
  StringRef role;        // "lhs", "rhs", "bias", "acc", "result"
  TPosition srcPos;      // 源 TPosition（GM 用 TPosition::GM）
  TPosition dstPos;      // 目标 TPosition
};

// 每层循环的搬运任务集合
struct LoopAnnotation {
  SmallVector<CopyTask> prologue;  // 循环入口搬运
  SmallVector<CopyTask> epilogue;  // 循环出口搬运
  bool isParallel = false;         // 是否为分核循环
};

// Buffer 分配上下文：记录每个 memref Value 对应的 TPosition
using BufferTPositionMap = DenseMap<Value, TPosition>;
```

### 3.3 实现步骤

#### Step A：解析所有循环的 Annotation

遍历 `func::FuncOp` 内所有 `scf::ForOp`，解析其 `ascendc.prologue`、`ascendc.epilogue`、`ascendc.parallel` attribute，构建 `LoopAnnotation` 映射表。

```cpp
// 解析单条搬运描述字符串，如 "lhs:GM->A1,rhs:GM->B1"
SmallVector<CopyTask> parseCopyTasks(StringRef annotation) {
  SmallVector<CopyTask> tasks;
  SmallVector<StringRef> entries;
  annotation.split(entries, ',');
  for (auto entry : entries) {
    // 格式: "role:src->dst"
    auto [role, path] = entry.split(':');
    auto [src, dst] = path.split("->");
    tasks.push_back({role.trim(), parseTPosition(src.trim()), parseTPosition(dst.trim())});
  }
  return tasks;
}

// TPosition 字符串到枚举的映射
TPosition parseTPosition(StringRef name) {
  return StringSwitch<TPosition>(name)
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

#### Step B：推导每个 memref 的 TPosition

遍历每层循环内的 `memref.subview` 和 `memref.alloc`，结合 Step A 的 annotation 信息，建立 `BufferTPositionMap`。

推导规则实现：

```cpp
void inferBufferTPositions(scf::ForOp forOp,
                           const LoopAnnotation &ann,
                           BufferTPositionMap &posMap) {
  // 找到该循环内所有 linalg op
  forOp.walk([&](linalg::LinalgOp linalgOp) {
    auto unitAttr = linalgOp->getAttrOfType<StringAttr>("ascendc.unit");
    if (!unitAttr) return;

    StringRef unit = unitAttr.getValue();

    if (unit == "AiCore.Cube") {
      // matmul: operand(0)=lhs, operand(1)=rhs, operand(2)=out
      // 查找包含该 op 的最近 for 的 prologue
      Value lhs = linalgOp.getDpsInputOperand(0)->get();
      Value rhs = linalgOp.getDpsInputOperand(1)->get();
      Value out = linalgOp.getDpsInitOperand(0)->get();

      // lhs/rhs 的 TPosition 从其所在最近 for 的 prologue 中匹配 role
      posMap[lhs] = findTPositionForRole(lhs, "lhs", forOp);
      posMap[rhs] = findTPositionForRole(rhs, "rhs", forOp);
      posMap[out] = TPosition::CO1;  // matmul out 固定为 CO1

    } else if (unit == "AiCore.Vector") {
      // add/max: 通过 def-use 链判断 out 是 VECCALC 还是 VECOUT
      Value out = linalgOp.getDpsInitOperand(0)->get();
      posMap[out] = isTerminalVectorOutput(out) ? TPosition::VECOUT
                                                 : TPosition::VECCALC;

      // ins 的 TPosition：来自 CO1 epilogue 的是 VECIN，来自 GM prologue 的也是 VECIN
      for (auto *opOperand : linalgOp.getDpsInputOperands()) {
        Value ins = opOperand->get();
        posMap[ins] = TPosition::VECIN;
      }
    }
  });
}

// 判断一个 Vector op 的 out 是否为最终输出（需要写回 GM）
bool isTerminalVectorOutput(Value v) {
  // 检查该 Value 是否被某个 for 的 epilogue "result:VECOUT->GM" 引用
  // 实现：向上查找最近的 scf.for，检查其 epilogue annotation 是否包含 "result:*->GM"
  // 且 v 的 def-use 链中无其他 Vector 计算的消费者
  for (auto *user : v.getUsers()) {
    if (isa<linalg::LinalgOp>(user)) return false;  // 还有后续 Vector 消费者
  }
  return true;  // 无 Vector 消费者，是最终输出
}
```

#### Step C：分配 ascendc.TBuf

将 `memref.alloc` 替换为对应 TPosition 的 `ascendc.tbuf`，并通过 `ascendc.tbuf.get_tensor` 获取 `LocalTensor`。

```cpp
void replaceMmemrefWithTBuf(memref::AllocOp allocOp,
                             TPosition pos,
                             OpBuilder &builder) {
  builder.setInsertionPoint(allocOp);
  MLIRContext *ctx = allocOp.getContext();

  // 创建 TBuf 类型：!ascendc.tbuf<pos>
  auto tBufType = TBufType::get(ctx, TPositionAttr::get(ctx, pos));

  // 实例化 TBuf
  auto tBufOp = builder.create<TBufOp>(allocOp.getLoc(), tBufType);

  // 从 TBuf 获取 LocalTensor（长度由 allocOp 的 shape 推导）
  Value len = computeElementCount(builder, allocOp.getLoc(), allocOp.getType());
  auto getTensorOp = builder.create<TBufGetTensorOp>(
      allocOp.getLoc(),
      LocalTensorType::get(ctx, allocOp.getType().getElementType()),
      tBufOp.getBuffer(), len);

  // 替换所有 allocOp 的 use
  allocOp.getResult().replaceAllUsesWith(getTensorOp.getTensor());
  allocOp.erase();
}
```

> **注意**：`ascendc.LocalTensor` 和 `memref` 是不同的类型系统。在 Phase 1 中，为了最小化变更范围，可以先只为 alloc 打 TPosition 标注（通过 `memref.memory_space` attribute），不做实际类型替换，Phase 2 统一做类型转换。详见 3.4 节的两种实现策略。

#### Step D：在正确位置插入搬运 op

遍历每层 `scf::ForOp`，在循环体入口（prologue）和出口（epilogue）插入搬运 op。

```cpp
void insertCopyOps(scf::ForOp forOp,
                   const LoopAnnotation &ann,
                   const BufferTPositionMap &posMap,
                   OpBuilder &builder) {

  // === 插入 prologue（循环体第一条 op 之前）===
  Block &body = forOp.getRegion().front();
  builder.setInsertionPointToStart(&body);

  for (auto &task : ann.prologue) {
    if (task.srcPos == TPosition::GM && task.dstPos == TPosition::A1) {
      // GM → A1: DataCopyL2（MTE2 通道）
      insertDataCopyGM2L1(builder, forOp, task, posMap);
    } else if (task.srcPos == TPosition::GM && task.dstPos == TPosition::B1) {
      insertDataCopyGM2L1(builder, forOp, task, posMap);
    } else if (task.srcPos == TPosition::GM && task.dstPos == TPosition::VECIN) {
      // GM → VECIN: DataCopyL2（bias 搬运）
      insertDataCopyGM2UB(builder, forOp, task, posMap);
    } else if (task.srcPos == TPosition::A1 && task.dstPos == TPosition::A2) {
      // A1 → A2: DataCopyL0（MTE1 通道）
      insertDataCopyL1ToL0(builder, forOp, task, posMap);
    } else if (task.srcPos == TPosition::B1 && task.dstPos == TPosition::B2) {
      insertDataCopyL1ToL0(builder, forOp, task, posMap);
    }
  }

  // === 插入 epilogue（循环体最后一条 op 之后）===
  builder.setInsertionPoint(body.getTerminator());

  for (auto &task : ann.epilogue) {
    if (task.srcPos == TPosition::CO1 && task.dstPos == TPosition::VECIN) {
      // CO1 → VECIN: Fixpipe（FIX 通道）
      insertFixpipe(builder, forOp, task, posMap);
    } else if (task.dstPos == TPosition::GM) {
      // VECOUT → GM: DataCopyL2（MTE3 通道）
      insertDataCopyUB2GM(builder, forOp, task, posMap);
    }
  }
}
```

具体的 DataCopy 插入示例（GM → A1）：

```cpp
void insertDataCopyGM2L1(OpBuilder &builder, scf::ForOp forOp,
                          const CopyTask &task, const BufferTPositionMap &posMap) {
  Location loc = forOp.getLoc();

  // 找到对应 role 的 GlobalTensor（func 参数）和 LocalTensor（A1 alloc）
  Value globalSrc = findGlobalTensorForRole(task.role, forOp);
  Value localDst  = findLocalTensorForRole(task.role, task.dstPos, posMap, forOp);

  // 计算搬运元素数（从 subview sizes 推导）
  Value calCount = computeCopyCount(builder, loc, localDst);

  // 插入 DataCopy（L2 API，GM↔UB/L1）
  // 对应 AscendC API: DataCopy(dst, src, calCount)
  builder.create<DataCopyL2Op>(loc, localDst, globalSrc, calCount);
}
```

Fixpipe 插入示例（CO1 → VECIN）：

```cpp
void insertFixpipe(OpBuilder &builder, scf::ForOp forOp,
                   const CopyTask &task, const BufferTPositionMap &posMap) {
  Location loc = forOp.getLoc();

  // CO1 的 LocalTensor（matmul out）
  Value co1Tensor   = findLocalTensorForRole("acc", TPosition::CO1, posMap, forOp);
  // VECIN 的 LocalTensor（add ins[0]）
  Value vecinTensor = findLocalTensorForRole("acc", TPosition::VECIN, posMap, forOp);

  // FixpipeParams（最简配置，无量化）
  auto paramsType = FixpipeParamsType::get(builder.getContext(),
                                            builder.getF32Type());
  Value params = builder.create<DataCopyPadExtParamsOp>(loc, paramsType);

  // 插入 Fixpipe op
  // 对应 AscendC API: Fixpipe(dst, src, workspace, params)
  builder.create<FixpipeOp>(loc, vecinTensor, co1Tensor,
                             /*workspace=*/Value{}, params);
}
```

#### Step E：清除所有 ascendc.* Annotation

```cpp
void clearAnnotations(func::FuncOp funcOp) {
  funcOp.walk([](Operation *op) {
    SmallVector<StringAttr> toRemove;
    for (auto attr : op->getAttrs()) {
      if (attr.getName().getValue().startswith("ascendc."))
        toRemove.push_back(attr.getName());
    }
    for (auto name : toRemove)
      op->removeAttr(name);
  });
}
```

#### Step F：runOnOperation 主流程

```cpp
void AscendCBufferPlacementPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // A: 解析所有循环的 annotation
  DenseMap<scf::ForOp, LoopAnnotation> loopAnnotations;
  funcOp.walk([&](scf::ForOp forOp) {
    LoopAnnotation ann;
    if (auto attr = forOp->getAttrOfType<StringAttr>("ascendc.prologue"))
      ann.prologue = parseCopyTasks(attr.getValue());
    if (auto attr = forOp->getAttrOfType<StringAttr>("ascendc.epilogue"))
      ann.epilogue = parseCopyTasks(attr.getValue());
    if (forOp->hasAttr("ascendc.parallel"))
      ann.isParallel = true;
    if (!ann.prologue.empty() || !ann.epilogue.empty() || ann.isParallel)
      loopAnnotations[forOp] = std::move(ann);
  });

  // B: 推导每个 memref 的 TPosition
  BufferTPositionMap posMap;
  for (auto &[forOp, ann] : loopAnnotations)
    inferBufferTPositions(forOp, ann, posMap);

  // C: 替换 memref.alloc 为 ascendc.tbuf
  SmallVector<memref::AllocOp> allocs;
  funcOp.walk([&](memref::AllocOp alloc) { allocs.push_back(alloc); });
  for (auto alloc : allocs) {
    if (auto it = posMap.find(alloc.getResult()); it != posMap.end())
      replaceMmemrefWithTBuf(alloc, it->second, builder);
  }

  // D: 插入搬运 op
  for (auto &[forOp, ann] : loopAnnotations)
    insertCopyOps(forOp, ann, posMap, builder);

  // E: 清除所有 annotation
  clearAnnotations(funcOp);
}
```

### 3.4 两种实现策略（二选一）

在与 Phase 2 的接口设计上，有两种策略：

**策略 A（保守型，推荐 Phase 1 采用）**

Phase 1 只做两件事：
1. 在 `memref.alloc` 上打 `memref.memory_space` attribute，标注 TPosition
2. 在循环入口/出口插入显式搬运 op（但 operand 仍然是普通 memref）

优点：改动最小，Phase 1 不需要引入 ascendc dialect 的类型系统，风险低，测试简单。

```mlir
// 策略 A 的输出示例
%alloc_A1 = memref.alloc() {ascendc.tposition = "A1"} : memref<128x256xf32>
ascendc.data_copy_l2 %alloc_A1, %global_A, %count : memref<...>, memref<...>, i32
```

**策略 B（激进型）**

Phase 1 完成真正的类型转换，将 `memref` 替换为 `ascendc.LocalTensor`，插入 `ascendc.tbuf`/`ascendc.tbuf.get_tensor`。

优点：Phase 2 接收到的 IR 已经是完整的 ascendc 类型，lowering 更干净。
缺点：需要同时处理 linalg op 的 operand 类型不匹配问题（linalg 接受 memref 不接受 LocalTensor），可能需要插入 cast op 过渡。

**建议**：Phase 1 采用策略 A，Phase 2 统一完成类型转换。本文档的代码示例以策略 B 为参考，实现时根据选择调整。

### 3.5 文件结构

```
lib/Transforms/
└── AscendCBufferPlacement/
    ├── AscendCBufferPlacementPass.h       // Pass 声明
    ├── AscendCBufferPlacementPass.cpp     // Pass 实现（主流程）
    ├── AnnotationParser.h/.cpp            // Step A: annotation 解析
    ├── TPositionInference.h/.cpp          // Step B: TPosition 推导
    ├── TBufAllocator.h/.cpp               // Step C: TBuf 分配
    ├── CopyInserter.h/.cpp                // Step D: 搬运 op 插入
    └── CMakeLists.txt
```

---

## 4. 测试方案

### 4.1 单元测试（FileCheck 格式）

位于 `test/Transforms/AscendCBufferPlacement/`。

#### 测试 1：Annotation 解析正确性

验证 prologue/epilogue 字符串被正确解析为 CopyTask。

```
// test_annotation_parse.mlir
// RUN: mlir-opt %s --ascendc-buffer-placement | FileCheck %s

func.func @test(%A: memref<?x?xf32>, %B: memref<?x?xf32>) {
  scf.for %i = ... {
    ascendc.prologue = "lhs:GM->A1,rhs:GM->B1"
  } {
    // CHECK: ascendc.data_copy_l2
    // CHECK: ascendc.data_copy_l2
    // CHECK-NOT: ascendc.prologue
  }
}
```

#### 测试 2：TPosition 推导正确性（matmul）

验证 linalg.matmul 的三个 operand 正确分配到 A2、B2、CO1。

```
// test_tposition_matmul.mlir
// RUN: mlir-opt %s --ascendc-buffer-placement | FileCheck %s

func.func @test(...) {
  scf.for %K ... {ascendc.prologue = "lhs:A1->A2,rhs:B1->B2",
                   ascendc.epilogue = "acc:CO1->VECIN"} {
    // CHECK: ascendc.tbuf : !ascendc.tbuf<a2>
    // CHECK: ascendc.tbuf : !ascendc.tbuf<b2>
    // CHECK: ascendc.tbuf : !ascendc.tbuf<co1>
    linalg.matmul {ascendc.unit = "AiCore.Cube"} ...
    // CHECK: ascendc.fixpipe
    // CHECK-NOT: ascendc.epilogue
  }
}
```

#### 测试 3：TPosition 推导正确性（Vector op，VECCALC vs VECOUT）

验证 add 的 out 推导为 VECCALC（有后续消费者），max 的 out 推导为 VECOUT（无后续 Vector 消费者）。

```
// test_tposition_vector.mlir
// RUN: mlir-opt %s --ascendc-buffer-placement | FileCheck %s

func.func @test(...) {
  scf.for %Tb_N ... {
    scf.for %K ... {ascendc.epilogue = "acc:CO1->VECIN"} {
      linalg.matmul {ascendc.unit = "AiCore.Cube"} ...
    }
    // add 的 out 有后续 max 消费 → VECCALC
    // CHECK: ascendc.tbuf : !ascendc.tbuf<veccalc>
    linalg.elementwise kind=add {ascendc.unit = "AiCore.Vector"} ...

    // max 的 out 无后续 Vector 消费 → VECOUT
    // CHECK: ascendc.tbuf : !ascendc.tbuf<vecout>
    linalg.elementwise kind=max_signed {ascendc.unit = "AiCore.Vector"} ...
  }
}
```

#### 测试 4：搬运 op 插入位置与顺序

验证 prologue 的搬运在循环体开头，epilogue 的搬运在循环体结尾，且顺序与 annotation 一致。

```
// test_copy_order.mlir
// RUN: mlir-opt %s --ascendc-buffer-placement | FileCheck %s

func.func @test(...) {
  scf.for %TB_N ... {
    ascendc.prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN",
    ascendc.epilogue = "result:VECOUT->GM"
  } {
    // CHECK: ascendc.data_copy_l2  ← lhs:GM->A1（prologue 第1条）
    // CHECK: ascendc.data_copy_l2  ← rhs:GM->B1（prologue 第2条）
    // CHECK: ascendc.data_copy_l2  ← bias:GM->VECIN（prologue 第3条）
    // ... 计算 op ...
    // CHECK: ascendc.data_copy_l2  ← result:VECOUT->GM（epilogue）
    // CHECK-NOT: ascendc.prologue
    // CHECK-NOT: ascendc.epilogue
  }
}
```

#### 测试 5：annotation 清除完整性

验证 pass 运行后所有 `ascendc.*` attribute 被清除。

```
// test_annotation_cleanup.mlir
// RUN: mlir-opt %s --ascendc-buffer-placement | FileCheck %s

// CHECK-NOT: ascendc.prologue
// CHECK-NOT: ascendc.epilogue
// CHECK-NOT: ascendc.parallel
// CHECK-NOT: ascendc.unit
```

### 4.2 集成测试

使用完整的 fc_add_relu 计算图，端到端验证 Phase 1 的输出：

```bash
# Step 1: 生成带 annotation 的 Step3 IR
mlir-opt fc_add_relu.mlir \
  --transform-interpreter=transform-library=transform_tile_and_fuse_3level.mlir \
  -o step3_annotated.mlir

# Step 2: One-Shot Bufferize
mlir-opt step3_annotated.mlir \
  --one-shot-bufferize="bufferize-function-boundaries=true allow-return-allocs-from-loops=true" \
  --buffer-deallocation-pipeline \
  -o step3_bufferized.mlir

# Step 3: 运行 Phase 1
mlir-opt step3_bufferized.mlir \
  --ascendc-buffer-placement \
  -o step3_buffer_placed.mlir

# Step 4: 验证输出
# 检查项：
# 1. 无 memref.alloc 残留（或全部有 tposition 标注）
# 2. 存在正确数量的 data_copy_l2 / data_copy_l0 / fixpipe op
# 3. 无任何 ascendc.* annotation 残留
# 4. linalg.matmul / linalg.elementwise 保持完整（Phase 1 不处理）
grep -c "ascendc.data_copy_l2" step3_buffer_placed.mlir  # 期望 >= 4
grep -c "ascendc.data_copy_l0" step3_buffer_placed.mlir  # 期望 >= 2
grep -c "ascendc.fixpipe"      step3_buffer_placed.mlir  # 期望 >= 1
grep    "ascendc.prologue"     step3_buffer_placed.mlir  # 期望无输出
```

### 4.3 验证 Checklist

| 检查项 | 验证方法 | 期望结果 |
|---|---|---|
| for_TB_N.prologue 产生 3 条 DataCopy | FileCheck | GM->A1, GM->B1, GM->VECIN 各一条 |
| for_TB_N.epilogue 产生 1 条 DataCopy | FileCheck | VECOUT->GM 一条 |
| for_K.prologue 产生 2 条 DataCopy | FileCheck | A1->A2, B1->B2 各一条 |
| for_K.epilogue 产生 1 条 Fixpipe | FileCheck | CO1->VECIN 一条 |
| matmul out 分配到 CO1 | FileCheck | `!ascendc.tbuf<co1>` |
| add out 分配到 VECCALC | FileCheck | `!ascendc.tbuf<veccalc>` |
| max out 分配到 VECOUT | FileCheck | `!ascendc.tbuf<vecout>` |
| 无 annotation 残留 | grep/FileCheck | 0 条匹配 |
| linalg op 保持不变 | FileCheck | `linalg.matmul` 仍存在 |
| IR 仍合法（verify pass） | `--verify-each` | 无报错 |

### 4.4 边界情况测试

| 场景 | 测试目的 |
|---|---|
| tile size 为动态值（`?`）| 验证 calCount 计算正确处理动态 shape |
| K 轴为 1（无实际 for_K 循环）| 验证 for_K 退化时搬运仍正确插入 |
| bias 形状与 C 形状不同（广播）| 验证 bias:GM->VECIN 的 count 计算 |
| for_TB_M 循环只有1次迭代（被优化掉）| 验证 parallel 标注的循环被优化后 pass 的鲁棒性 |

---

## 5. 附录：关键类型与 Op 参考

### 5.1 TBuf 相关

```tablegen
// Types.td
def AscendC_TBuf : AscendC_BaseQueueType<"TBuf", "tbuf"> {
  let parameters = (ins "TPositionAttr":$tPositionAttr);
  let assemblyFormat = "`<` custom<PrettyTPosition>($tPositionAttr) `>`";
}

// TBuf.td
def AscendC_TBufOp : AscendC_Op<"tbuf", [AscConstructor]> {
  let results = (outs AscendC_TBuf:$buffer);
}
def AscendC_TBufGetTensorOp : APIOp<"tbuf.get_tensor", "Get"> {
  let arguments = (ins AscendC_TBuf:$buffer, Optional<AnyType>:$len);
  let results = (outs AscendC_LocalTensor:$tensor);
}
```

用法示例：
```mlir
%tbuf_A2 = ascendc.tbuf : !ascendc.tbuf<a2>
%tensor_A2 = ascendc.tbuf.get_tensor %tbuf_A2, %len : !ascendc.tbuf<a2>, !ascendc.local_tensor<f32>
```

### 5.2 DataCopy 相关

```tablegen
// OpDataCopy.td
def AscendC_DataCopyL2Op : DataCopyOp<"data_copy_l2", "DataCopy", [AscFunc]> {
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AnyType:$calCount);
}
def AscendC_DataCopyL0Op : DataCopyOp<"data_copy_l0", "DataCopy", [AscFunc]> {
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AscendC_DataCopyParams:$repeatParams);
}
```

搬运通道对应关系：
- `data_copy_l2`：MTE2（GM↔UB）或 MTE3（UB→GM），对应 GM→L1/UB 和 VECOUT→GM
- `data_copy_l0`：MTE1（L1↔L0），对应 A1→A2 / B1→B2

### 5.3 Fixpipe 相关

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

用法示例（CO1→VECIN，无量化）：
```mlir
%params = ascendc.data_copy_pad_ext_params : !ascendc.fixpipe_params<f32>
ascendc.fixpipe %vecin_tensor, %co1_tensor, %params
    : !ascendc.local_tensor<f32>, !ascendc.local_tensor<f32>, !ascendc.fixpipe_params<f32>
```

### 5.4 同步相关（Phase 1 暂不实现，供参考）

```tablegen
// OpBlockSync.td
def PipeBarrierOp : APIOp<"pipe_barrier", "PipeBarrier"> {
  let arguments = (ins AscendC_PipeAttr:$pipe);
}
// 主要管道：
//   PIPE_MTE2 (4): GM→L1 DataCopy 通道
//   PIPE_MTE1 (3): L1→L0 DataCopy 通道
//   PIPE_M    (2): Cube 计算通道
//   PIPE_V    (1): Vector 计算通道
//   PIPE_MTE3 (5): L0C→UB Fixpipe 通道 / UB→GM DataCopy 通道
```

TQue 的 EnQue/DeQue 同步机制在 Phase 2 实现，Phase 1 不涉及。

---

*文档结束*