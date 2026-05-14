# P3b — RCore reduce 多核模板实施计划

> 父计划:`docs/superpowers/plans/2026-05-11-port-af-scheduler-to-vector-plan.zh.md` §4 + P3b 行。
>
> 本文把父计划里 `P3b` 那个大格子拆成 4 个可独立 commit 的子阶段(P3b-1 → P3b-4),并细化文件清单、接口、验收门。
>
> **执行约定:** P3b-1 / P3b-2 可以由 sub-agent 按本计划直接执行;**P3b-3 必须先单独走一次设计讨论再动**(group-split + 双 kernel 是结构性变化,本文只给出方向,不给最终代码)。

**目标:** 让 vector-plan 支持 AF 的 `kRCore` reduce 模板 —— R 轴切多核 + 两阶段 partial→combine —— 解决 full-reduce-to-scalar(softmax / LayerNorm / loss)在 vector-plan-codegen 下的两个问题:(a)R1 的 IR 非法 bug,(b)真正的 R 轴并行(不只是 block_dim=1 的退化兜底)。

**架构思路:** 沿 AF `reduce_schedule_case_generator.cpp::GeneratorRCoreTask` 的做法 —— 在 RCore 被选中时,把一个 group 拆成两个 kernel func(partial + combine),用 GM workspace tensor 把两段串起来,coordinator 表达 kernel 间依赖。我们这边比 AF 多一层 host-launch 调度(`AclnnBackend` / `hostLaunchAscendCKernel`),所以 host code gen 也要配套改。

**Tech stack:** MLIR (linalg / scf / tensor / bufferization 方言)、vector-plan TilePlanGen / LoopNestBuilder / GroupEmitter / GroupOutline、`AclnnBackend` host-cpp 生成器、`hostLaunchAscendCKernel` runtime。

**当前依赖:** AF port P1 / P3a / P3b-i(RBLOCK 单 kernel split,`51a4195`)/ P4 / P5a–d / B-1 / B-3 / c1 全部已完成。R3 已修(`28c8ea6`)。R1 仍 open。Memory `[[af-scheduler-port]]` 有完整 commit 时间线。

**重要术语区分(避免混淆):**
- **RBLOCK split**(P3b-i 已做):**单 kernel** 内 R 轴切大块,VECCALC accumulator 累加(`reduce_sum_2d_l2` 一次喂一块)。解决"R 轴 tile 装不进 UB"。
- **RCore**(本计划):**双 kernel** partial→combine,R 轴切到不同核。解决"P 小、R 大,想用 R 轴提供多核并行"。
- 两条路径 codegen 不重叠:RBLOCK 走 VECCALC accumulator + scf.for inner loop;RCore 走 partial GM workspace + 第二阶段 combine kernel。

---

## 0. 端到端目标形态(完成 P3b 后 IR 长什么样)

**输入(用户 linalg):**
```mlir
func.func @network(%x: tensor<1024xf32>, %init: tensor<f32>) -> tensor<f32> {
  %r = linalg.generic {iterator_types = ["reduction"]}
       ins(%x) outs(%init) { ... } -> tensor<f32>
  return %r
}
```

**Outline 之后(P3b-3 完成):**
```mlir
// network.mlir
func.func @network(%x: tensor<1024xf32>, %init: tensor<f32>) -> tensor<f32> {
  %ws = tensor.empty() : tensor<32xf32>             // 32 = block_dim
  %ws1 = call @kernel_group0_partial(%x, %ws)
         : (tensor<1024xf32>, tensor<32xf32>) -> tensor<32xf32>
  %r = call @kernel_group0_combine(%ws1, %init)
       : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
  return %r
}

func.func private @kernel_group0_partial(...) -> tensor<32xf32>   // block_dim=32, each block partial-reduces 1024/32=32 elements
func.func private @kernel_group0_combine(...) -> tensor<f32>       // block_dim=1, reduces 32 partials → 1 scalar
```

**network.json 加段:**
```json
{
  "kernels": [
    { "id": "kernel_group0_partial", "kind": "ascendc", ... },
    { "id": "kernel_group0_combine", "kind": "ascendc", "depends_on": ["kernel_group0_partial"], ... }
  ]
}
```

**host code:** AclnnBackend 顺序 emit 两次 `hostLaunchAscendCKernel`,workspace tensor 由 coordinator 中间产物的 host alloc 承担。

R1 reproducer 输出非零、精度 ≤ 1e-6。

---

## 1. 子阶段速览

| 阶段 | 内容 | 行为变化 | 工作量 | 风险 |
|---|---|---|---|---|
| **P3b-1** ✅ | RCore 作为枚举候选,`pickBest` 不选 | 无 | 小(~150 行) | 低 |
| **P3b-2** ◀ 起手 | RCore 的 blockSplit/ubSplit/loop 物化,仅 emit partial 单 kernel | 新选项,只能跑 partial-only(数值不对) | 中(~300 行) | 中 |
| **P3b-3** ⭐ | Group 拆分成 partial + combine 双 kernel,coord/JSON/host gen 串起来 | R1 reproducer 跑通,精度过 | **大**(~600 行 + 设计讨论) | **高** —— 先单独 design review |
| **P3b-4** | `costEstimate` 真正选 RCore,处理 reduce-in-middle/多 reduce | 性能优化路径打通 | 中(~200 行) | 中 |

**P3b-1 状态(已 done):** 落在 AF port 的 P5b/P5c/P5d 这批 commit 里(`1beaa2d` / `17962e3` / `aad7ab2`),不在 RCore 专属 commit 下:
- `TilePlanDraft::reduceIsBlock` 字段存在(`include/Conversion/VectorPlan/TilePlan.h`)。
- `enumerateTilingCases` 在每个 `ubR != -1` 的 draft 后追加一个 `reduceIsBlock=true` 的 RCore variant(`TilePlanGen.cpp:284-291`)。
- `costEstimate` 在 `draft.reduceIsBlock` 时返回 `kInfeasible`(line 324-325),并标 TODO "no two-stage codegen yet"。
- `buildPlan` 入口 assert RCore draft 不会到这里(line 359-360)。

所以本计划**起手直接 P3b-2**,P3b-1 当作 prerequisite 验证(读上面 4 处代码确认还在,再起步)。

每阶段 1 个 commit,标题 `feat(vector-plan): P3b-N — <内容>`。

---

## 2. 现状摸底(动手前必读 ~30 分钟)

执行 P3b-1 前必须读完下面这些文件并能口述它们的当前职责。这是 P1 已经铺好的骨架,P3b 完全是往里加,不重写。

| 文件 | 当前职责 | P3b 怎么用 |
|---|---|---|
| `include/Conversion/VectorPlan/TilePlan.h` | `TilePlan` / `TileParam` / `AxisGrouping` / `AxisClass` / `TilePlanDraft` 数据结构 | P3b-1 加 `enum ReduceTemplate { None, Common, FullLoad, RCore }`、`reduceTemplate` 字段、`reduceIsBlock` flag |
| `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` | `classifyAxes` / `enumerateTilingCases` / `blockSplit` / `ubSplit` / `pickBest` 编排 | P3b-1/2/4 加 RCore 候选 + 决策 |
| `lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.cpp` | 物化 outer/inner scf.for | P3b-2 加 "R 轴当 block 轴" 物化 |
| `lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp` | clone linalg op + emit init/yield + reduction-split | P3b-3 拆分成两阶段 emit |
| `lib/Conversion/VectorPlan/GroupOutline/GroupOutlinePass.cpp` | 1 group → 1 private kernel func | P3b-3 改成 1 group → ≤2 kernel funcs(RCore 时双 kernel) |
| `lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.cpp` | tensor 层 coord → network.json | P3b-3 加 kernel 依赖字段(`depends_on` 或 `groups_relations_in`) |
| `lib/Runtime/AclnnBackend/AclnnBackend.cpp` | tensor coord → host C++ | P3b-3 顺序 emit 多次 launch 调用、分配 workspace |
| `lib/Runtime/Execution/HostLaunchHelper.cpp` | 单次 kernel launch | P3b-3 检查是否需要 per-kernel 不同 block_dim 支持(应该已经支持,需 verify) |
| AF 源码 `reduce_schedule_case_generator.cpp::GeneratorRCoreTask` (`/home/gser/code/ge-eco/.../autofuse/optimize/task_generator/`) | partial + combine 两阶段拆分模板 | **只读对标**;P3b-3 设计时参考其 group 拆分逻辑 |
| AF doc `reduce全流程说明.md` | RCore 三模板的设计说明 | **只读对标** |

**摸底验收:** 能口述清楚下面 3 个问题:
1. P1 commit `98030e6` 把 `AxisGrouping` / `TilePlanDraft` 数据结构放在哪了?`TilePlan` 里有哪些字段是 P1 加的、哪些是预留的?
2. 现在 `enumerateTilingCases` 怎么枚举?返回什么?
3. `GroupOutline` 现在怎么把一个 group 变成一个 kernel func?它怎么决定 boundary in / out?

---

## 3. P3b-1 — RCore 作为枚举候选(不选中)— ✅ 已完成

**状态:** P5b/P5c/P5d 已经把这块做掉了。RCore variant 在 `enumerateTilingCases` 里枚举,`costEstimate` 用 `kInfeasible` 拒选,`buildPlan` 有 assert 保护。

**起手前 verify(~5 分钟):**
1. `grep -n "reduceIsBlock" include/Conversion/VectorPlan/TilePlan.h` — 字段存在。
2. `grep -n "reduceIsBlock\|RCore" lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` — 看到 RCore variant 枚举 + `kInfeasible` 拒选 + buildPlan assert,4 处。
3. R3 reproducer 现在仍 PASS、R1 reproducer 现在仍 fail —— 跟 plan 描述一致。

如果以上 verify 全通过,跳到 §4 起手 P3b-2。

---

## 4. P3b-2 — RCore blockSplit/ubSplit 物化(单 kernel partial-only)

**Commit message:** `feat(vector-plan): P3b-2 — materialize R-axis-as-block tiling, emit partial-only kernel`

**目标:** 让 `pickBest` 在 full-reduce 场景下**允许选 RCore**,但只 emit **partial kernel 的 IR**(不做 combine,不拆 group)。输出是 `tensor<block_dim×...>` 的 partial 张量,**数值上还不对**(没 combine),但 IR 合法、编译过、跑得起来。这一步主要验 LoopNestBuilder / GroupEmitter 对 "R 轴当 block 轴" 的物化是否正确。

### 4.1 文件清单

| 动作 | 文件 | 改动 |
|---|---|---|
| 改 | `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` | `blockSplit` RCore 分支:block 轴 = `(非 R 外轴 merge) ++ (R 切块)`(对标 AF `ReduceBlockTiling`)。`ubSplit` 处理 `ubTilingAxisR`(reduce 轴的 inner-tile + tail-peel)。`pickBest` 在 full-reduce(`yAxes 全空 || P 极小`)时允许选 RCore |
| 改 | `lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.cpp` | 加 "R 轴当 block 轴" 物化路径:当 `TilePlan::reduceIsBlock` 时,把 R 轴当 outer scf.for(等价于今天 parallel 轴的 outer),原来的 inner reduction RBLOCK 仍然存在 |
| 改 | `lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp` | RCore 单 kernel partial-only:linalg op 的 outs 改成 `tensor<block_dim×...>`(rank+1),每个 block 写自己那一槽,本阶段**不**做 combine |
| 建 | `examples/reduce-rcore-partial-only/` | 输入:`tensor<1024xf32>` full reduce → `tensor<f32>`(用户视角);MLIR 输出:partial kernel + 临时 `tensor<32xf32>` workspace。**精度不验**,只验 IR 结构 + 能编译 + 能 launch 出 32 个 partial |
| 建 | `test/Conversion/VectorPlanCodegen/rcore-block-axis-on-r.mlir` | lit 断言:RCore 时 outer scf.for 的轴 == 原始 R 轴的 inner split |

### 4.2 关键设计点

- **R 轴当 block 轴的物化:** 父计划 §3.3 已说明,fuse 后的 block 轴 = `(非 R 外轴) ++ (R 块轴)`;非 R 段空(full-reduce)时 block 轴就只剩 R 块轴。LoopNestBuilder 现有的"非 R 外轴乘积外层循环 + div/mod 恢复"逻辑套用得上,只是把"非 R 外轴段"换成"R 块轴段"。
- **partial-only emit:** GroupEmitter 在 RCore + 单 kernel 模式下,把 linalg op 的 outs 从 `tensor<...>` 升维成 `tensor<block_dim × ...>`(在最外加一维),`outs(%init)` 替换成 `outs(tensor.empty<block_dim × ...>)`,每个 block 的迭代写到 `outs[%block_idx, ...]`。
- **`pickBest` 允许选 RCore 的条件:** 简单版 —— `if (full_reduce && reduceTemplate == RCore) score < Common 的 +inf`。等 P3b-4 再做真正的 cost model。
- **本阶段输出数值不对:** partial 张量是 `block_dim` 个分量,不是最终 scalar。用户看到的是 `tensor<32xf32>` 而不是 `tensor<f32>`。这是 expected。

### 4.3 验收门

1. 现有 lit + e2e 全过(回归)。
2. 新 lit `rcore-block-axis-on-r.mlir` 断言 outer scf.for 跑在 R 轴上、step 是 `R_BLOCK` tunable。
3. `examples/reduce-rcore-partial-only/` 能编译 + 能 launch + 能 dump 出 `tensor<32xf32>` 的 partial 结果(数值不验,只验形态)。
4. R1 reproducer **仍然 fail**(没拆 group → 没有 combine → coordinator 类型不匹配)。这是 expected。

### 4.4 风险

- LoopNestBuilder 的 IV 恢复(div/mod 把 fused block 轴拆回原始轴)在 R 轴上不需要(R 轴本来就单独切),逻辑可以简化。
- GroupEmitter 升维 outs 这件事要小心:tensor.empty 的 shape 要包含 block_dim,这是个 SSA tunable,需要 `tensor.empty(%bdim)` dynamic shape。验证 bufferize 处理得动。

---

## 5. P3b-3 — Group 拆分成 partial + combine 双 kernel ⭐

**⚠️ 起手前必须单独走一次 design review。本节只列方向、文件、风险,不给最终代码。**

**Commit message:** `feat(vector-plan): P3b-3 — split RCore group into partial + combine kernels`

**目标:** RCore 被选中时,把一个 reduce group 拆成两个独立的 kernel func(partial 和 combine),coordinator 串两次 call,workspace tensor 衔接。R1 reproducer 跑通,精度过。

### 5.1 三个核心未解决问题(design review 必须回答)

**Q1: 拆分发生在哪个 pass、什么 IR 阶段?**

候选 A:`GroupOutline` 之前(tensor 层,linalg.generic 还在)—— 在 tile-fuse 之后、outline 之前插一个 `SplitRCoreGroup` pass,把单个 linalg.generic(带 reduction iter)拆成 partial generic + combine generic,改 coord 加 workspace。

候选 B:`GroupOutline` 内部 —— outline pass 看见 RCore group 时自动 emit 两个 kernel func。

候选 C:`vector-plan-codegen` 出口(memref 层)—— partial / combine 都已经 lowered 到 AscendC,但要拆 host launch 序。

**初步倾向 A**(对标 AF 的做法,AF 在 ImplGraph 层拆,即等价于 tensor 层)。Q1 决定后续所有设计。

**Q2: workspace tensor 怎么表达?**

- partial 输出 `tensor<block_dim × ...>`(P3b-2 已经在做)。这个张量是不是直接当 coord 的中间 SSA value,然后 combine 把它当 input?
- 那 host code 要不要给这个 tensor 分配 GM workspace?还是直接走"coord 内部产物 → AclnnBackend 自动 alloc"现有逻辑?
- workspace 的大小取决于 block_dim,而 block_dim 是 tilings file 里的 tunable —— host 端怎么知道分配多少?

**初步思路:** workspace 直接当 coord 的中间 SSA tensor,`AclnnBackend` 已经会 alloc 中间 tensor(`emitAscendCLaunch` 里有 dynamic dim 分配逻辑),应该不需要新机制。但 block_dim 作为 shape 的动态来源要确认。

**Q3: combine kernel 的 block_dim 怎么定?**

- combine 的输入是 `tensor<block_dim × ...>`,reduce 维 = 32(假设 block_dim=32),其他维 = post-reduce shape。
- 自然 block_dim:1(单核做小 reduce)。但 R 大、post-reduce shape 又有 parallel 维时(reduce-in-middle 场景),combine 自己也可能想多核分 post-reduce 的 parallel 维。
- 本阶段先简化:**combine 永远 block_dim=1**。reduce-in-middle 留到 P3b-4 处理。

### 5.2 文件清单(初稿,design review 后确定)

| 动作 | 文件 | 改动方向 |
|---|---|---|
| 建 | `lib/Conversion/VectorPlan/GroupOutline/SplitRCoreGroup.cpp`(或类似) | 新 pass:在 outline 前,把 RCore 的 linalg op 拆成 partial+combine 两个 op + 中间 workspace tensor |
| 改 | `lib/Conversion/VectorPlan/GroupOutline/GroupOutlinePass.cpp` | 让 outline 接受 "1 group ≤ 2 kernel" 的输入(可能不需要改,如果 SplitRCoreGroup 已经在 tensor 层拆成两个独立 op) |
| 改 | `lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.cpp` | 顺序记录 kernel call,依赖关系自然由 tensor SSA def-use 表达 |
| 改 | `lib/Runtime/AclnnBackend/AclnnBackend.cpp` | 多次 `emitAscendCLaunch` 已经支持(elewise 多 kernel 已经 work),验证 workspace 中间 tensor 的 alloc 走得通 |
| 改 | `lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp` | P3b-2 的 partial-only emit 在这里被 SplitRCoreGroup 调用;combine 的 emit 路径独立 |
| 建 | `examples/reduce-rcore-e2e/` | full-reduce-to-scalar e2e:输入 `tensor<1024xf32>`,期望 scalar,精度 ≤ 1e-6 |
| 建 | `test/Conversion/Collapse/tile-fuse-vector-rcore-split.mlir` | lit 断言 SplitRCoreGroup 后 coord 里出现两个 call |

### 5.3 验收门

1. 现有所有 lit / e2e 全过(回归)。
2. R1 reproducer:`rank-1→rank-0 full reduce` 跑通,精度 ≤ 1e-6。**这是 R1 真正修复的标志。**
3. `examples/reduce-rcore-e2e/run.sh` PASS。
4. 至少一个 `tensor<rank-2→rank-1 reduce>` 也能走 RCore(虽然 `pickBest` 默认不选,可以加 flag 强制)。

### 5.4 风险

- **大件,设计未定。** 起手前必走一次单独 design review(写一份 1-2 页的 sub-spec)。
- `SplitRCoreGroup` 改 tensor 层 IR 时,要小心和 P3b-2 的 partial-only emit 配合 —— 不能"GroupEmitter 升维 outs" 和 "SplitRCoreGroup 拆 op" 各干各的。可能 P3b-2 实现完后,P3b-3 要把 P3b-2 的逻辑挪到 SplitRCoreGroup 里、GroupEmitter 回退到不升维。这种重构是预期内的。
- AclnnBackend 的中间 tensor alloc 用的是 dynamic shape,block_dim 作为 shape 来源能不能正确传过去要 verify。可能需要给 workspace tensor 一个明确的 static-by-tilings shape 而不是 dynamic SSA。

---

## 6. P3b-4 — costEstimate 真正选 RCore + 边角场景

**Commit message:** `feat(vector-plan): P3b-4 — cost-driven RCore selection and reduce-in-middle handling`

**目标:** `pickBest` 在 "R 大、P 小" 场景下真正选 RCore(而不是父计划 §3.6 那种占位规则);处理 reduce 不在 group 末尾的情况(后面还有 elewise / broadcast);处理多个 reduce。

### 6.1 文件清单

| 动作 | 文件 | 改动 |
|---|---|---|
| 改 | `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` | `costEstimate`:RCore 的 score = `(blockDim 离 #AICores 距离) + (combine kernel 单核开销) - (partial 阶段并行收益)`。Common 退化形(`vectorizedDims` 占满 + block_dim=1) score 比 RCore 差时,选 RCore。 |
| 改 | `lib/Conversion/VectorPlan/GroupOutline/SplitRCoreGroup.cpp` | reduce-in-middle:把 reduce 后的 elewise / broadcast 算子塞进 combine kernel,而不是 coordinator |
| 建 | `examples/softmax-rcore-e2e/`(可选) | nanoGPT softmax 形:`tensor<B×L×Dxf32>` 沿 D 做 softmax(D=768) —— 验 RCore 在 softmax 上能选中且 perf > Common 退化形 |

### 6.2 验收门

1. softmax-like example 选中 RCore(`--debug-only=tile-plan-gen` 看到);精度过。
2. reduce 后有 elewise 的 combo 仍然 work(可能需要新 example 或扩 `combo-elewise-reduce-e2e`)。
3. 多个独立 reduce 的 group(经 fuser 已拆开)各自独立选模板,不互相影响。

---

## 7. 阶段间依赖

```
P3b-1 ✅ ─→ P3b-2 ◀ 起手 ─→ P3b-3 ⭐(先 design review) ─→ P3b-4
                              ↑
   AF port P1/P3a/P3b-i(RBLOCK)/P4/P5a–d/B-1/B-3/c1 全部已 done
   R3 fix 已 done(28c8ea6)
   P3b-1 RCore 枚举 + ∞-scored 已 done(P5b/P5c/P5d 三个 commit 内)
```

P3b-2 是真正起手点。P3b-3 起手前要走 design review,不直接执行本计划。

**注意 P3b-i(RBLOCK 单 kernel,`51a4195`)和本计划的关系:** 它在做 R 轴 inner-tile 切大块,数据通过 VECCALC accumulator + scf.for iter_arg 累加,**始终单 kernel**。本计划的 RCore 是 R 轴切到**不同 block**(多 kernel),partial 走 GM workspace、combine 第二阶段。两条 codegen 路径在 GroupEmitter / LinalgToAscendC 里**分别独立**(`reduceIsBlock=false` 走 RBLOCK,`reduceIsBlock=true` 走 RCore),实现时不互相踩。

---

## 8. 现状 vs 目标(给执行者的导航)

- **R1 现状:** `vector-plan-codegen` 在 full-reduce-to-scalar 上吐出非法 IR(`func.return` not last + 孤立 `scf.yield`)。
- **本计划完成后 R1 现状:** R1 reproducer 跑出正确 scalar 结果,精度过;softmax / LayerNorm 走 vector-plan 不再阻塞。
- **本计划没修的:** R2(`ReduceSum<half, RA>` CANN 不支持)、R4(`block_dim_expr` 空)、R5(greedy fuser 跨 Rule 3 合并)—— 与 R1 / RCore 正交。

---

## 9. AF 源码 cross-ref(只读对标,不照搬)

| 本文 § | AF 文件 |
|---|---|
| §3 RCore 枚举 | `optimize/autoschedule/autoschedule.cpp::GenTilingCase`(`is_reduce_first_stage` 分支) |
| §4 R 轴当 block 轴 | `optimize/autoschedule/schedule.cpp::ReduceBlockTiling` |
| §5 group 拆分 | `optimize/task_generator/reduce_schedule_case_generator.cpp::GeneratorRCoreTask`(170-215 行)+ `RMulticorePhase2Graph::Construct` |
| §5 group 依赖 | `optimize/optimize.h::ScheduleTask::groups_relations_in` + `Optimizer::IsReduceFirstStage / RefreshGroupRelation` |
| §6 cost / 模板选择 | `optimize/task_generator/reduce_score_function_generator.cpp` |
| 整体设计 | `autofuse/doc/reduce全流程说明.md` |

---

## 10. 不在范围内 / 显式跳过

- **R2 / R4 / R5**:正交问题,不在本计划。
- **transpose / concat / split / gather** 的 schedule case:父计划 P4/P5/P7,与 RCore 无关。
- **真正的 score_func 调参**:本计划只做"能选中 RCore"的最简 cost,真正性能调优单独排期。
- **动态 shape + 符号约束求解**:父计划 P6 范畴。
- **支持 cube(matmul)算子参与 RCore**:RCore 在 AF 也只对 vector reduce 适用。
