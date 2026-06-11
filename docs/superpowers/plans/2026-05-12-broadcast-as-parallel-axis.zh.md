# 方案：broadcast = parallel 轴 + lowering 层复制

> 配套上一份 `2026-05-11-port-af-scheduler-to-vector-plan.zh.md`；这份只管 broadcast 这条路的正式化。
> AF 侧参考：`ge-eco/.../autofuse/doc/AF知识地图/专题-broadcast全流程.md`。

## 0. 问题

现状对 broadcast 轴的调度是个 hack 且坏的：
- "小广播轴" → `buildPlan` 的 `else if (ax.isBroadcastSplit)` 分支走 **`BCAST_n` Full step-1**（一个 `for bcast in 0..ext step 1` 循环，tile 在该维 size 1）。codegen 出来是 per-element `DataCopy(ub[i], gm[i*stride], 1)` + `pipe.init_buffer` 在 N-trip 循环里 —— Ascend DMA 是 32-byte 粒度、`count=1` 会按整 block 走 → ub 互相覆盖；InitBuffer-in-loop → UB 池溢出。→ **算错**（probe `b1_trailing` / `b3_leading` sim `max_abs_diff≈4.7`）。
- "大广播轴" → `BCAST_TILE_n` Inner tunable（一个 `for bcast in 0..ext step BCAST_TILE` 循环）。`bcast-multi-axis-e2e` 走这条且 run.sh 把 `BCAST_TILE` pin 成整 extent → 实际上 1 个 trip → 整维 tile + 片上 `broadcast_l2` 复制 —— **这条是对的**（= AF 的路子）。
- "中间轴广播 + 块轴被切" → 残留 `memref.subview`（把广播 operand 在 step-1 loop 外 hoist 的产物）→ translate 失败（probe `b2_middle`）。

AF 的态度：broadcast 是一种 view 语义（merge 前折进 Load 的 repeats/strides），广播轴往往**不成为真正的循环**（`RemoveAllZeroStrideLoopAxis`），片上靠专门的 `Broadcast` API（`BroadcastOneAxis/TwoAxis/AllCommonAxis` → `BroadcastFirstDim` 等）复制。

## 1. 核心原则

**broadcast 是一个 linalg-native 的 view：某 operand 的 indexing map 把某些迭代维 project 掉了（= 该 operand 沿那些维是常量）。** 调度器把广播轴**当普通 parallel 轴**；"把常量 operand 在片上沿被 drop 的维复制开"是**纯 lowering 的事**，由 operand 的 indexing map 驱动，不需要专门的调度。

— 我们不照搬 AF 的"折进 Load view"，而是用 linalg 的 projection map 天然表达 broadcast + 用现有的 `ascendc.broadcast_l2` 在片上复制。

## 2. 各层

### §2.1 分类（`Collapse` / `classifyAxes` / `TileFuseUtils`）—— 信息化，不影响调度

- 迭代轴 `d` 是 **broadcast-const** ⟺ 某 group member 有个 operand 的 indexing map 不引用 `d`，且 `d` 是 parallel 轴。（reduce 轴被 output map drop 是 reduce，不是 broadcast。）
- `CollapsedGroupInfo::broadcastAxes` = 这种轴的并集（已有，`Collapse` 算）。
- `AxisClass` 上的 flag：**把 `isBroadcastSplit` 改名 `isBroadcastConst`**（它不再意味着"调度上特殊"，只意味着"有 operand 沿它常量"），更新所有引用点。
- `classifyAxes`：broadcast 轴 `kind = Y`、`isBroadcastConst = true`、`bindMultiCore = false`（不当块轴 —— 保持 `pickBlockAxis` 行为稳定；AF 多数也不把广播轴切多核）。
- **唯一消费者是 lowering**（`ComputeConversion`/`DataMoveConversion` 用它/或直接看 operand map 来决定哪些 operand 要复制）。

### §2.2 调度（`TilePlanGen` / `buildPlan` / `enumerateTilingCases` / `costEstimate`）

- **删掉 `buildPlan` 里整个 `else if (ax.isBroadcastSplit)` 分支。** 广播轴落进 Y 轴的常规处理：块轴（不会，`bindMultiCore=false`）/ ubY（inner tunable）/ 否则 §3.4 整维全载。→ 现实里广播轴 = "整维全载"（直到真 cost model 改它）。删 `bcastCount`/`bcastTileCount` 计数器。
- `enumerateTilingCases`：`ubY` 候选纳入广播轴（去掉 5c 里 `!g.axes[y].isBroadcastConst` 的过滤）。它们是非块轴 ubY → `costEstimate` 现在 ∞ 它们 → 不被选 → 结构上完整、无害。
- `costEstimate`：**扩展 5b 的可行性检查** —— 一个 draft 如果把某根**静态超大的 parallel 或 broadcast 轴整维全载**、估计放不下（粗略：`size · elemBytes > kReductionTileBudgetBytes` 之类的阈值，先复用同一个常量）→ ∞ 那个 draft，逼它去 ub-tile（= 等价于以前的 `BCAST_TILE`，但走统一的 ubY 机制）。这条让方案"通用"而不退化。（注意：要让大广播轴被 ub-tile 真正可行，得让"非块轴 ubY" 的 codegen 工作 —— 见下方"依赖"。）
- `pickBlockAxis` 不变（仍跳过 broadcast 轴 / `bindMultiCore=false` 的轴）。

### §2.3 lowering（`ComputeConversion` / `DataMoveConversion`）—— broadcast 的"真身"

- 根据 operand 的 indexing map 检测"该 operand 沿迭代维 `d` 是常量" → 按它的 reduced shape load（连续 `DataCopy`）→ 用 `ascendc.broadcast_l2` 沿 `d` 在 tile 内复制 → compute op 在整 tile 上跑。
- **必须覆盖广播维在 tile 里的所有位置**：尾维（`[N,1]→[N,M]`，已有：probe `b1` 已经在发 `broadcast_l2`）、中间维（`[N,1,M]→[N,K,M]`）、首维（`[1,M]→[K,M]`，≈ AF `BroadcastFirstDim`）。`b2`/`b3` 可能有洞 → 这部分补全（`broadcast_l2` 的 verifier + lowering 对不同 `constRank` + 广播维位置的支持）。
- `b2` 那个残留 `memref.subview`：§2.2 删了 step-1 loop 后这条 hoist 路不触发 → 自然消失。

### §2.4 examples / 验收

- `bcast-multi-axis-e2e/run.sh`：去掉 `BCAST_TILE_0/1` params + 注释（广播轴现在整维全载）—— 功能不变、kernel 变。
- 新增 e2e 门禁：`bcast-trailing-e2e` / `bcast-middle-e2e` / `bcast-leading-e2e`（= `/tmp/rbprobe/b1,b2,b3` 转正，搬进 `examples/`），加进门禁列表（→ 11 个）。
- lit：`tile-fuse-vector-bcast-{trailing,middle,leading}.mlir` 钉调度结果（广播轴无 plan 条目、operand map 是 projection、`broadcast_l2` 出现且广播维位置对）。
- probe 脚本 `/tmp/rbprobe/runone.py` 留着，每个 commit 后跑一遍 r1/r2/r3/b1/b2/b3/c1。

## 3. 切法（commit 序列）

| # | 内容 | 验收 |
|---|---|---|
| **B-1** | §2.1（改名 `isBroadcastConst`）+ §2.2（删 BCAST 分支、ubY 纳入广播轴、`costEstimate` 扩展超大整维轴 → ∞）+ §2.4 改 `bcast-multi-axis-e2e/run.sh` | lit 不变（或微调那 1 个 bcast lit）；8 门禁过；probe `b1`/`b3` sim 应该已通（取决于 `broadcast_l2` 首/尾维是否都 OK） |
| **B-2** | §2.3 —— `broadcast_l2` lowering 支持尾/中/首维广播 + verifier；顺带 `b2` 的中间轴情形 | probe `b2` translate+sim 通；`b1`/`b3` 仍通 |
| **B-3** | §2.4 —— 3 个新 e2e example + lit，加进门禁 | 11 门禁过；新 lit 过 |
| **B-4**（可选/留后） | broadcast-const 做成 per-(axis, operand) 精确（现在 per-axis union；`b2` 是混合情形，B-2 里会碰到，按需细化） | — |

## 4. 依赖 / 风险

- **"非块轴 ubY 的 codegen"** —— `costEstimate` 要能把大广播轴选去 ub-tile，就需要 `buildPlan` 物化一个 `ubY ≠ 块轴` 的 draft。两条实现路：(a) "swap 块轴" —— 块轴改成 `draft.ubTilingAxisY`，原 `bp.axis` 改 §3.4 整维；小改但对某些 shape 会产生 rank-N strided DataCopy（`DataMoveConversion` 现在只处理 rank-2 strided）；(b) 块轴保持 `bp.axis` + ubY 轴拿独立 inner tunable + 块轴拿一个退化 Inner —— 但块轴退化 Inner 在非-innermost 时不走 tail-peel → 最后一个 outer tile 会 over-read（LoopNestBuilder 现在只 peel innermost inner）。**所以 B-1 里 `costEstimate` 的"超大整维轴 → ∞"先只对 reduce 轴生效（5b 已有），parallel/broadcast 轴的那条留到"非块轴 ubY codegen"做完再开** —— 否则会触发上面的坑。即 B-1 实际上只删 BCAST 分支 + 改名 + run.sh，不引入新的 ∞ 条件。大广播轴整维放不下这个限制，跟所有整维全载轴一样，留给 cost model。
- `broadcast_l2` 现在对首维 / 中间维广播的支持度未知 —— B-1 跑完 probe 就知道 `b3` 行不行；不行则 B-2 范围更大些。
- `bcast-multi-axis-e2e` 的广播轴 extent —— run.sh pin `BCAST_TILE=D0/D2`，说明它们小到能整维放下；改成整维全载不会爆 UB。若以后有反例，靠 B-1 风险段说的限制兜底（= 整维放不下就是已知限制）。
