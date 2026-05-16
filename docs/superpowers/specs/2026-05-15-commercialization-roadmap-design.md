# Ascend-MLIR V2 商用化路线图设计

**日期**：2026-05-15  
**状态**：已批准，待实现  
**范围**：五阶段商用化改造，目标覆盖主流 LLM 推理（Transformer decoder）和视觉 backbone（ResNet/ViT）

---

## 背景与约束

当前 V2 pipeline（Normalize→Kernelize→Schedule→Realize→Backend）架构骨架已就位，但存在四个商用阻塞：

1. **算子覆盖不足**：`classifyElementwiseBodyOp` 只认 3 个 arith op；Reduction 只支持 AddF；无 f16/bf16/i8 dtype 路径。任何真实 LLM 模型在 Kernelize 后必然 unsupported。
2. **Attention pattern 硬编码**：`collectAttentionLikeHandwrittenPattern` 只匹配教科书 SDPA 拓扑，FlashAttention/GQA/带 mask 变体无法识别，新增变体必须改主干代码。
3. **Realize 是 plan-only**：Schedule 决策未驱动 IR 变换，tiling 和 buffer placement 由旧 Phase5 隐式完成，两套逻辑不一致。
4. **无真实模型 e2e 测试**：所有 full-pipeline test 是手写小片段，无数值正确性验证。

硬性架构原则（不得违反）：上层不允许出现按算子名/模型名分支的硬编码；新算子/新模式靠注册扩展，不改主干。

---

## 阶段总览

```
Phase 1: 算子覆盖扩展          → 解锁 softmax/GELU/LayerNorm/RMSNorm/量化
Phase 2: Attention 扩展性       → pattern recognizer 注册化，支持 FlashAttention/GQA
Phase 3: Realize IR 实例化      → Schedule 决策驱动 scf.for 生成 + bufferize
Phase 4: Transformer e2e        → decoder layer 端到端 + CPU 仿真数值对拍
Phase 5: Vision e2e             → Conv/BN/ReLU + ViT block 跑通
```

依赖链：Phase 1 是所有阶段前提；Phase 2 和 Phase 3 依赖 Phase 1，互相独立可并行；Phase 4 依赖 Phase 1+2+3；Phase 5 依赖 Phase 4。

---

## Phase 1：算子覆盖扩展

### 问题定位

三处形成"三层锁"，必须同步扩展：

| 层 | 文件 | 当前限制 |
|---|---|---|
| Classifier | `LinalgBodyClassifier.cpp:68-76` | elementwise 只认 AddF/MulF/MaximumF |
| Classifier | `LinalgBodyClassifier.cpp:330-353` | reduction 只认 AddF |
| BackendSupportMatrix | `BackendSupportMatrix.h:20-34` | ComputeKind 枚举缺少新种类 |
| ComputeConversion | `ComputeLoweringPass.cpp` → `ComputeConversion.cpp:2801` | emit 只处理 add/mul/max |

### 1A：arith/math elementwise 全集扩展

**新增 op 集合**（分优先级）：

优先级 1（解锁 softmax/GELU/LayerNorm）：
- `arith.subf`、`arith.divf`、`arith.negf`、`arith.maxf`
- `math.exp`、`math.exp2`、`math.log`、`math.sqrt`、`math.rsqrt`
- `math.tanh`、`math.erf`

优先级 2（解锁更多激活函数和数值算子）：
- `arith.remf`、`arith.ceilf`、`arith.floorf`
- `math.sin`、`math.cos`、`math.abs`、`math.copysign`、`math.round`、`math.fma`
- `arith.cmpf` + `arith.select`（ReLU、masked attention 的 `-inf` 填充）

**设计：`ElementwiseBodyOpRegistry`**

把 arith/math op 类型 → `ComputeKind` + AscendC intrinsic 的映射做成注册表，`LinalgBodyClassifier`、`BackendSupportMatrix`、`ComputeConversion` 三层消费同一张表。新增一个 op 只需一处注册。

```cpp
// lib/Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.h
struct ElementwiseBodyOpEntry {
  // op 类型标识（用 TypeID）
  llvm::TypeID opTypeId;
  ComputeKind kind;
  unsigned arity;  // 1=unary（exp/sqrt/neg），2=binary（add/sub/mul）
  // AscendC 发射函数指针（供 ComputeConversion 调用）
  // 参数顺序：builder, loc, dst, src0, src1（unary 时 src1 为 nullptr Value）, count
  std::function<void(OpBuilder&, Location, Value, Value, Value, Value)> emitter;
};

void registerElementwiseBodyOp(ElementwiseBodyOpEntry entry);
void registerBuiltinElementwiseBodyOps();  // 注册全部内置 op
const ElementwiseBodyOpEntry* lookupElementwiseBodyOp(llvm::TypeID opTypeId);
```

`classifyElementwiseBodyOp` 改为查表：
```cpp
ComputeKind classifyElementwiseBodyOp(Operation &bodyOp) {
  auto *entry = lookupElementwiseBodyOp(bodyOp.getRegisteredInfo()->getTypeID());
  return entry ? entry->kind : ComputeKind::Unknown;
}
```

`ComputeKind` 枚举新增：`ElementwiseSub`、`ElementwiseDiv`、`ElementwiseNeg`、`ElementwiseExp`、`ElementwiseSqrt`、`ElementwiseRsqrt`、`ElementwiseTanh`、`ElementwiseErf`、`ElementwiseLog`、`ElementwiseAbs`、`ElementwiseSin`、`ElementwiseCos`、`ElementwiseFma`、`ElementwiseSelect`（配合 cmpf+select 的 fused 模式）。

**FusedElementwise 的泛化**：当前 `isSupportedFusedElementwiseBody` 把 body 里的 op 列表线性遍历并要求每个都在支持集里。扩展后逻辑不变，支持集由注册表决定，不再硬编码。

### 1B：Reduction 种类扩展

新增 `ComputeKind`：`ReductionMax`、`ReductionMin`、`ReductionMul`。

`isSupportedPhase5ReductionBody` 改为：识别 body 里的单个 arith/math binary op（AddF/MaximumF/MinimumF/MulF），返回对应的 `ComputeKind`，不再写死 AddF。

同时支持**多输出 reduction**（body yield 两个值，如同时输出 sum 和 max），用于 online softmax。

### 1C：dtype 扩展

**目标 dtype**：
- `f16`、`bf16`（LLM 推理标配，优先级最高）
- `i8`、`ui8`（量化推理）
- 混合 dtype matmul：输入 f16/bf16，累加 f32，输出 f16/bf16

**设计**：

`populateLinalgSemanticInfo`（`KernelizeSemanticUtils.cpp:262`）新增 dtype 字段到 `KernelizeOpSemanticInfo`：
```cpp
struct KernelizeOpSemanticInfo {
  // ... 现有字段 ...
  SmallVector<Type, 4> inputElementTypes;
  SmallVector<Type, 2> outputElementTypes;
};
```

`BackendSupportMatrix` 新增 dtype 校验接口：
```cpp
bool isSupportedDtype(ComputeKind kind, ArrayRef<Type> inputTypes,
                      ArrayRef<Type> outputTypes) const;
UnsupportedReason explainDtype(ComputeKind kind, ArrayRef<Type> inputTypes,
                               ArrayRef<Type> outputTypes) const;
```

AscendC 后端（`ComputeConversion.cpp`）的 matmul/elementwise emit 路径根据 dtype 选择对应的 AscendC API（`Mmad`/`Mmad16`/`MmadInt8` 等）。

### Phase 1 验收标准

- [ ] `linalg.generic` with softmax body（subf + exp + reduce-add + divf）通过 Kernelize 分类，生成正确 AscendC
- [ ] `linalg.generic` with GELU body（fma + erf + mul）通过完整 pipeline
- [ ] `linalg.generic` with RMSNorm body（mul + reduce-add + rsqrt）通过完整 pipeline
- [ ] f16 matmul + f16 elementwise 通过 pipeline，dtype 不匹配时给出明确诊断
- [ ] 所有现有 full-pipeline LIT 测试无回归
- [ ] 新增 LIT 测试：`ascend-full-pipeline-softmax.mlir`、`ascend-full-pipeline-gelu.mlir`、`ascend-full-pipeline-rmsnorm.mlir`、`ascend-full-pipeline-f16-matmul.mlir`

---

## Phase 2：Attention Pattern 扩展性

### 问题定位

`collectAttentionLikeHandwrittenPattern`（`FusionCandidateAnalysis.cpp:349-415`）承担了两件不同的事：图拓扑识别 + 角色分配。两件事混在一个函数里，新增变体必须改主干。

### 设计：`HandwrittenPatternRecognizer` 接口 + 注册表

**接口定义**（新文件 `lib/Conversion/Ascend/Kernelize/HandwrittenPatternRecognizer.h`）：

```cpp
namespace mlir::afir::ascend::kernelize {

struct PatternMatch {
  SmallVector<Operation *, 8> ops;  // 匹配到的 op 群（按 opId 排序）
  llvm::StringRef kindId;            // 对应的 HandwrittenContract kind
};

class HandwrittenPatternRecognizer {
public:
  virtual ~HandwrittenPatternRecognizer() = default;
  virtual llvm::StringRef kindId() const = 0;
  virtual std::optional<PatternMatch>
  tryMatch(Operation *seed,
           const DependencyAnalysisResult &deps,
           const OpRoleMap &roleMap,
           const KernelizeConfig &config) const = 0;
};

void registerHandwrittenPatternRecognizer(
    std::unique_ptr<HandwrittenPatternRecognizer> recognizer);
void registerBuiltinHandwrittenPatternRecognizers();
ArrayRef<HandwrittenPatternRecognizer *> getHandwrittenPatternRecognizers();

} // namespace
```

**注册表**（新文件 `HandwrittenPatternRecognizerRegistry.cpp`）：与 `HandwrittenContractRegistry` 同模式，`llvm::ManagedStatic` + `std::mutex`，线程安全，幂等注册。

**内置 recognizer 迁移**：

`AttentionSdpaRecognizer`：把 `collectAttentionLikeHandwrittenPattern` 现有逻辑原封不动迁入 `tryMatch`，`kindId()` 返回 `kKernelizeHandwrittenKindAttentionSdpa`。主干函数删除。

`FlashAttentionRecognizer`（新增）：识别条件：
- seed 有 Cube role
- 存在多轮"Cube→Reduction→Vector（含 exp/mul 特征）→Cube"链式结构
- Vector op 集合中有 `math.exp` 类型的 body op（通过 Phase 1 的 `ElementwiseBodyOpRegistry` 查询）
- `kindId()` 返回 `kKernelizeHandwrittenKindFlashAttention`（Phase 1C 后加入，需同步在 `HandwrittenContractRegistry` 注册对应 contract）

**`FusionCandidateAnalysis.cpp` 主循环改造**：

```cpp
// 旧代码（删除）
if (auto ops = collectAttentionLikeHandwrittenPattern(seed, deps, roleMap, config))
  return buildHandwrittenPatternCandidate(handwrittenKind, *ops, ...);

// 新代码
registerBuiltinHandwrittenPatternRecognizers();
for (HandwrittenPatternRecognizer *recognizer : getHandwrittenPatternRecognizers()) {
  if (auto match = recognizer->tryMatch(seed, deps, roleMap, config))
    return buildHandwrittenPatternCandidate(match->kindId, match->ops, ...);
}
```

**与 HandwrittenContractRegistry 的关系**：recognizer 只负责"这是什么 pattern"（返回 kindId），contract 负责"这个 pattern 怎么调度"。两个 registry 职责正交，kindId 是连接键。

### Phase 2 验收标准

- [ ] `AttentionSdpaRecognizer` 迁移后，现有 attention LIT 测试无回归
- [ ] `FlashAttentionRecognizer` 能识别 2 轮迭代的 FlashAttention-style 拓扑（新增 LIT 测试）
- [ ] 新增自定义 recognizer 的单元测试：注册 → 识别 → contract lookup 全链路
- [ ] `FusionCandidateAnalysis.cpp` 中不再有任何硬编码的 handwritten kind 字符串比较

---

## Phase 3：Realize IR 实例化

### 问题定位

`RealizePass` 默认 `plan-only` 不改 IR。`ScheduleDecision` 中的 tile sizes 未驱动任何 IR 变换。旧 Phase5 用自己的内部推断隐式完成 tiling，两套逻辑可能不一致。

### 设计：两步走

**Step A：TilingRealizationDriver（scf.for 生成）**

新增 `materialization-mode=tiled-linalg`，新增 `TilingRealizationDriver`（`lib/Conversion/Ascend/Realize/TilingRealizationDriver.cpp`）。

职责：
1. 读取每个 kernel 的 `ScheduleDecision`（tile sizes、dominant role）
2. 对 primary op 调用 MLIR 上游 `linalg::TileUsingForOp`，生成 `scf.for` nest
3. 对 fused ops 调用 `linalg::fuseProducerOfSlice`，把 vector/reduction epilogue 融入循环体
4. 输出：`scf.for` + `linalg.generic`（仍是 tensor IR）

```cpp
class TilingRealizationDriver {
public:
  LogicalResult materialize(ModuleOp module,
                            const RealizePlanBundle &plan,
                            const ScheduleDecision &decision);
private:
  LogicalResult tilePrimaryOp(Operation *primaryOp,
                               ArrayRef<int64_t> tileSizes,
                               SmallVectorImpl<Operation *> &tiledOps);
  LogicalResult fuseEpilogueOps(ArrayRef<Operation *> fusableOps,
                                 scf::ForOp outerLoop);
};
```

**Step B：Buffer Placement（bufferize + memory space 标注）**

扩展为 `materialization-mode=full-realize`（Step A + Step B）。

Step B 在 Step A 输出的 tiled tensor IR 上：
1. 调用 One-Shot Bufferize（现有 `BufferizationDriver`）
2. 消费 `MemoryRealizationPlan` 的 placement 决策，对每个 buffer 打 memory space 属性
3. 输出：带 memory space 标注的 memref IR，可直接交给 Phase5 `ComputeLoweringPass`

**与 Phase5 的分工**：

| 职责 | V2 Realize（full-realize 模式） | Phase5 ComputeLoweringPass |
|---|---|---|
| Tiling（scf.for 生成） | ✓ | 不再自己推断 |
| Buffer placement | ✓ | 不再自己推断 |
| AscendC intrinsic 发射 | ✗ | ✓（保持不动） |
| Movement op 生成 | ✗ | ✓（memref.copy → DMA） |

Phase5 接收 memref IR（已 tile、已 place），直接做 `classifyLinalgComputeKind` → emit AscendC，不再需要自己的 tiling 推断路径。

**兼容性**：原有 `plan-only`、`one-shot-bufferize`、`memory-space-annotate` 三种模式保持不变，不删除。`full-realize` 是新增模式。

### Phase 3 验收标准

- [ ] `materialization-mode=tiled-linalg`：matmul 的 tile sizes 来自 ScheduleDecision，生成的 `scf.for` 循环边界与 tile sizes 一致（LIT FileCheck）
- [ ] `materialization-mode=full-realize`：输出 memref IR 带正确 memory space，Phase5 可直接消费（LIT FileCheck）
- [ ] 现有所有 full-pipeline LIT 测试在 `full-realize` 模式下无回归
- [ ] `TilingRealizationDriver` 单元测试：给定 plan + decision，验证 scf.for 结构

---

## Phase 4：Transformer Decoder Layer e2e

### 目标

一个完整 GPT-style decoder layer 从 linalg IR 跑到 AscendC 代码，数值与 CPU reference 对拍。

**算子组成**：
- QKV projection：`linalg.matmul` (f16)
- Scaled dot-product attention：`linalg.batch_matmul` + softmax（subf + exp + reduce-add + divf）
- Output projection：`linalg.matmul` (f16)
- FFN：`linalg.matmul` + GELU（fma + erf + mul）+ `linalg.matmul`

### 验证链

```
test/Conversion/ascend-full-pipeline-transformer-decoder-layer.mlir
  → ascend-normalize
  → ascend-kernelize
  → ascend-schedule
  → ascend-realize（materialization-mode=full-realize）
  → ascend-compute-lower
  → AscendC 代码（FileCheck 验证 IR 变换正确性）

test/e2e/transformer-decoder-layer-numerical.py
  → 用 CPU 仿真器运行生成的 AscendC
  → 与 numpy reference 对拍（f16: atol=1e-2, rtol=1e-3；f32: atol=1e-5）
```

### Phase 4 验收标准

- [ ] LIT 测试通过：IR 各阶段变换正确（Kernelize 分配角色、Schedule 选模板、Realize 生成 scf.for、Backend emit AscendC）
- [ ] 数值对拍通过（CPU 仿真）：f16 精度 atol ≤ 1e-2
- [ ] Attention pattern 由 `AttentionSdpaRecognizer`（或 `FlashAttentionRecognizer`）识别，无 hard-coded kind 分支

---

## Phase 5：Vision Backbone e2e

### 目标

ResNet block（Conv2d + BatchNorm + ReLU）和 ViT block（LayerNorm + MHSA + FFN）跑通。

### 新增工作

**Conv Kernelize 语义模型注册**：
- 在 `KernelizeExternalModels.cpp` 注册 `linalg::Conv2DNhwcHwcfOp`、`linalg::PoolingNhwcMaxOp` 等的外部模型
- `populateLinalgSemanticInfo` 识别 Conv 的 iteratorKinds（含 window 维度），分配 `AccessPatternKind::Contraction`

**Cube schedule 4D tensor 支持**：
- `TemplateRegistry.cpp:102` 的 `cube_static_matmul` 模板 `maxRank` 从 3 改为 6
- `ScheduleSearch.cpp` 的 `getRoleDrivenCubeTile` 扩展到 4D+（NHWC 的 N/H/W parallel 轴 + C reduction 轴）

**BatchNorm/LayerNorm lowering**：
- 由 Phase 1C 的 math op 扩展覆盖（sqrt + reduce-add + sub + divf）

### Phase 5 验收标准

- [ ] ResNet basic block（Conv2d + BN + ReLU）端到端 LIT 测试通过
- [ ] ViT block（LayerNorm + MHSA + FFN）端到端 LIT 测试通过
- [ ] CPU 仿真数值对拍通过（f16 atol ≤ 1e-2）
- [ ] 无新增 hard-coded op 名分支（Conv 通过注册表接入）

---

## 文件变更总览

### 新增文件

| 文件 | 阶段 | 职责 |
|---|---|---|
| `lib/Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.h` | P1 | elementwise op → ComputeKind + emitter 注册表接口 |
| `lib/Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.cpp` | P1 | 注册表实现 + builtin op 注册 |
| `lib/Conversion/Ascend/Kernelize/HandwrittenPatternRecognizer.h` | P2 | PatternMatch 结构体 + Recognizer 接口 |
| `lib/Conversion/Ascend/Kernelize/HandwrittenPatternRecognizerRegistry.cpp` | P2 | 注册表实现 |
| `lib/Conversion/Ascend/Kernelize/AttentionSdpaRecognizer.cpp` | P2 | 现有 SDPA 逻辑迁入 |
| `lib/Conversion/Ascend/Kernelize/FlashAttentionRecognizer.cpp` | P2 | FlashAttention 拓扑识别 |
| `lib/Conversion/Ascend/Realize/TilingRealizationDriver.cpp` | P3 | Schedule 决策 → scf.for 生成 |
| `lib/Conversion/Ascend/Realize/TilingRealizationDriver.h` | P3 | 接口声明 |
| `test/Conversion/ascend-full-pipeline-softmax.mlir` | P1 | softmax e2e LIT |
| `test/Conversion/ascend-full-pipeline-gelu.mlir` | P1 | GELU e2e LIT |
| `test/Conversion/ascend-full-pipeline-rmsnorm.mlir` | P1 | RMSNorm e2e LIT |
| `test/Conversion/ascend-full-pipeline-f16-matmul.mlir` | P1 | f16 matmul e2e LIT |
| `test/Conversion/ascend-full-pipeline-transformer-decoder-layer.mlir` | P4 | decoder layer e2e LIT |
| `test/Conversion/ascend-full-pipeline-resnet-block.mlir` | P5 | ResNet block e2e LIT |
| `test/Conversion/ascend-full-pipeline-vit-block.mlir` | P5 | ViT block e2e LIT |
| `test/e2e/transformer-decoder-layer-numerical.py` | P4 | 数值对拍脚本 |
| `test/e2e/resnet-block-numerical.py` | P5 | 数值对拍脚本 |

### 修改文件

| 文件 | 阶段 | 改动 |
|---|---|---|
| `BackendSupportMatrix.h/.cpp` | P1 | 新增 ComputeKind 枚举值；新增 dtype 校验接口 |
| `LinalgBodyClassifier.cpp` | P1 | elementwise/reduction 分类改为查表 |
| `ComputeConversion.cpp` | P1 | elementwise emit 路径按注册表分发 |
| `KernelizeSemanticUtils.cpp` | P1C | 新增 dtype 字段到 KernelizeOpSemanticInfo |
| `KernelizeExternalModels.cpp` | P1C/P5 | 注册 dtype 感知模型；P5 注册 Conv |
| `FusionCandidateAnalysis.cpp` | P2 | 主循环改为遍历 recognizer 注册表 |
| `HandwrittenContractRegistry.cpp` | P2 | 注册 FlashAttention contract |
| `TemplateRegistry.cpp` | P2/P5 | P2: 从 contract 自动加载模板；P5: cube maxRank 扩展 |
| `RealizePass.cpp` | P3 | 新增 tiled-linalg 和 full-realize 模式 |
| `ScheduleSearch.cpp` | P5 | getRoleDrivenCubeTile 支持 4D+ |

---

## 不在本次范围内

- Autotuner / device 实测反馈回路（tuning DB 目前只做缓存）
- Diagnostic 全面改造（静默 return false 改为 emitError）
- Movement path 泛化（BackendSupportMatrix 6 条硬编码路径）
- Dynamic shape 全面支持
- Ring Attention / MLA 等高级变体（第二阶段）

这些不是商用阻塞，可在 Phase 5 之后按需规划。
