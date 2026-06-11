# Multi-Plan Tiling Variants — codegen 期出 N 个 kernel + autotune 跨 variant 选

> 父背景：`docs/superpowers/notes/2026-05-14-kg1-hang-debug.md`（autotune 重构讨论）。
>
> **执行约定：** P1-P5 plumbing 可由 sub-agent 按本计划直接执行；P6 验证（人为构造多 variant 算子）需要 design review 一次再动。

**目标：** 把 TilePlanGen 的 "enumerate → cost ∞ filter → pick best (序)→ buildPlan 1 个" 改成 "enumerate → cost ∞ filter → buildPlan 全部 → codegen 全部 → autotune 跨 variant + 跨 tile size 选 best"。

**核心动因：** 当前 `costEstimate` 是 feasibility-only（注释自承），跨 feasible draft 之间用枚举顺序选，不是真比较。多 variant codegen + autotune profile 让"scheduling 选择"也由实测决定，对齐 AF `AxesReorderPgoSolver` 形态。

**Tech stack：** MLIR (vector-plan TilePlanGen / GroupOutline)、`afir-translate -mlir-to-cann`、`autotuner`、`aclnn-backend` host gen、`network_runner.py`。

**当前依赖：** P3b-2 done（RCore TilePlan + GroupEmitter + CannTranslation R1 端到端 IR 合法），P3b-3 未 done。本计划不依赖 P3b-3，但 P3b-3 落地后会产出本计划的第一批"真" multi-variant 算子（RCore vs Common reduce 模板）。

**Flag：** `--vector-plan-codegen=enable-tiling-variants=1`，**default ON**。所有现有 e2e demo 跟着改名（`kernel_group0` → `kernel_group0__v0`），plumbing 在 regression 中天然被验证。

---

## 0. 端到端目标形态

**输入（用户 linalg）：**
```mlir
func.func @kernel_group0(%a: tensor<32x64xf16>, %b: tensor<32x64xf16>,
                          %init: tensor<32x64xf16>) -> tensor<32x64xf16> {
  %r = linalg.generic ... addf
}
```

**TilePlanGen + outline 之后（P1-P2 完成）：**
```mlir
module attributes {
  vector_plan.tiling_infos = [
    {kernel_id = "kernel_group0__v0", fields = [...], block_dim_expr = "..."},
    {kernel_id = "kernel_group0__v1", fields = [...], block_dim_expr = "..."}
  ]
} {
  func.func private @kernel_group0__v0(...) attributes {afir.axis_extent_expr = "2048", ...}
  func.func private @kernel_group0__v1(...) attributes {afir.axis_extent_expr = "2048", ...}
}
```

`afir-translate -mlir-to-cann` 出：
```
build_e2e/
├─ kernel_group0__v0.cpp
├─ kernel_group0__v0_space.json
├─ kernel_group0__v1.cpp
├─ kernel_group0__v1_space.json
└─ kernel_group0_family.json     ← P2 新增，索引 variants
```

Autotune 后（P3 完成）：
```
kernel_group0_best.json:
{
  "variant": "v1",                  ← P3 新增字段
  "config": {"XBLOCK": 256, "XBLOCK_SUB": 64, ...},
  "best": {"cycles": 471349, "block_dim": 8}
}
```

Host C++ 生成（P4 完成）：
```cpp
hostLaunchAscendCKernel("kernel_group0__v1",   // ← 选中的 variant
                        kernelBinariesDir,
                        tilingsPath,
                        ...);
```

---

## 1. 子阶段速览

| 子阶段 | 内容 | 影响面 | 体量 | 风险 |
|---|---|---|---|---|
| **P1** | TilePlanGen：filter ∞ → buildPlan 全部 → 多 variant func + per-variant tiling_infos | TilePlanGen.cpp，~150 行 | 中 |
| **P2** | CannTranslation + network_runner phase 2：每 variant 各出 cpp/space.json + family.json 索引 | CannTranslation.cpp、network_runner.phase2、build_host plumbing，~150 行 | 中 |
| **P3** | Autotuner `--family` 模式：跨 variant loop + cross-variant argmin | autotuner_main.cpp，~120 行 | 中 |
| **P4** | aclnn-backend：读 best.json variant → emit 带后缀的 launch 调用 | AclnnBackend.cpp，~30 行 | 低 |
| **P5** | network_runner phase 2/3/4/5 串起来；run_manifest / dump 命名跟随 variant | network_runner.py，~80 行 | 低 |
| **P6** | 6 个 e2e regression + Q1-(a) 人为造 2-draft 算子验证 cross-variant 选择 | examples/+ docs，~50 行 | 中 |

总体 ~580 行 + 验证，约 2 周。

---

## 2. 现状摸底（动手前必读）

执行前必须读完下面这些文件并能口述当前职责：

| 文件 | 当前职责 | 本计划改动点 |
|---|---|---|
| `include/Conversion/VectorPlan/TilePlan.h` | `TilePlan` / `TilePlanDraft` 结构 | P1 不动结构，加 `feasibleDrafts` 帮助函数 |
| `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` | `enumerateTilingCases` / `costEstimate` / `genVectorTilePlan` / `emitTilingInfos` | **P1 主战场**：genVectorTilePlan 改成 loop |
| `lib/Conversion/VectorPlan/Pipeline.cpp` | vector-plan pass pipeline | P1 加 `enable-tiling-variants` 选项 |
| `lib/Conversion/VectorPlan/GroupOutline/GroupOutlinePass.cpp` | 1 group → 1 private kernel func | P1 期间需要确认它能跟得上"多 variant func"——P3b-3 改这里时再深动；本计划 P1 暂时让所有 variant 落在同一个 module |
| `lib/Target/CannKernel/CannTranslation.cpp` | walk 单 func → emit 单 cpp + emitTilingSpaceJson | **P2 主战场**：walk 多 func → 多 cpp + 多 space.json + family.json |
| `tools/autotuner/autotuner_main.cpp` | 单 kernel runSearch | **P3 主战场**：加 `--family` 模式 |
| `lib/Runtime/AclnnBackend/AclnnBackend.cpp` | tensor coord + best tilings → host C++ launch 调用 | **P4 主战场**：launch name 改成读 best.json 的 variant |
| `python/network_runner.py` | 5-phase orchestrator | **P5 主战场**：phase 2/3/4/5 都要改 |
| `python/runner_utils/build_host.py` | g++ link | P2/P5 改 cpp 列表 |
| `python/runner_utils/network_json.py` | `network.json` 读取 + Network 类的 ascendc_kernels() | 不动（network.json 仍用 family id） |

**摸底验收：** 能口述清楚 3 个问题：
1. `enumerateTilingCases` 返回的 drafts 里，`costEstimate` 把哪些标 ∞？为什么？
2. `emitTilingInfos` 在 module attribute 上写了什么？afir-translate 怎么读？
3. `aclnn-backend` 是从 network.json 还是 tilings_best.json 拿到 kernel name 的？

---

## 3. P1 — TilePlanGen 全 build feasible drafts

**目标：** `genVectorTilePlan` 从 "选 1 build 1" 改成 "filter ∞ build 全部"。同 module 内多个 sibling func，命名 `<original>__v<idx>`。

### 3.1 接口设计

- 新 helper：`SmallVector<TilePlanDraft> filterFeasibleDrafts(drafts, info, vecDims, elemBytes)` —— 用 `costEstimate < ∞` 过滤。
- `genVectorTilePlan` 改为返回 `SmallVector<TilePlan>` 而不是单 `TilePlan`。
- caller (Pipeline.cpp / 上层调用者) loop over plans, 每个 plan 独立 buildPlan + emitTilingInfos。
- buildPlan 接受一个 `variantIdx` / `variantName` 参数，决定 func name suffix。

### 3.2 命名约定

- variant idx：drafts 数组里的索引（feasibility filter 之后）。
- variant id 字符串：`v0`, `v1`, ...
- func name：`<original>__<variant_id>`，例如 `kernel_group0__v0`。
- `__` 双下划线，避免跟 AscendC kernel 内部名字冲突。

### 3.3 emitTilingInfos 改动

现在写 module-level `vector_plan.tiling_infos = [{kernel_id: ..., fields: ..., block_dim_expr: ...}]`，**一条 entry**。

P1 后：每个 variant 一条 entry，`kernel_id` 含后缀。CannTranslation 在 P2 里按 kernel_id 找对应的 fields/block_dim_expr。

### 3.4 Single-variant 时的行为

当 feasibility filter 后只剩 1 个 draft（当前所有 e2e demo 情况），仍然 emit 1 个 variant，命名 `__v0`。**不做"单 variant 退化为无后缀"的特殊路径**——保持单一路径简化调试，跟用户决定的"看影响面"对齐。

### 3.5 文件改动清单

| 文件 | 改动 |
|---|---|
| `include/Conversion/VectorPlan/TilePlan.h` | 加 `TilePlanVariant { idx; name; plan; }` 或直接让 buildPlan 接 variantName 参数 |
| `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` | `genVectorTilePlan` 返回 vector；filterFeasibleDrafts helper；buildPlan 加 variantName 形参；emitTilingInfos 改 entry 写法 |
| `lib/Conversion/VectorPlan/Pipeline.cpp` | 加 `enable-tiling-variants` pass option，default true；caller loop |
| `lib/Conversion/VectorPlan/TileFuse/Collapse.cpp` | 确认它生成的 CollapsedGroupInfo 是 read-only（多次 buildPlan 复用同一个 info） |
| `lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp` | 同上，确认 stateless |
| `lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.cpp` | 同上 |
| `lib/Conversion/VectorPlan/GroupOutline/GroupOutlinePass.cpp` | 确认能处理 module 里多个 sibling func（应该已经能，需要 verify） |

### 3.6 P1 验收门

- 跑 `examples/autotune-stress-elewise` lowered MLIR：看到 `kernel_group0__v0` + `kernel_group1__v0` 两个 sibling func（每个原 group 各 1 个 variant，因为现在只 1 个 feasible）。
- module attr `vector_plan.tiling_infos` 有 2 个 entry，kernel_id 各带 `__v0` 后缀。
- 临时改 `costEstimate` 让 elementwise 出 2 个 feasible（Q1-(a)）：能看到 `__v0` + `__v1` 两个 variant func。
- P1 不动 CannTranslation / autotuner / host，所以**这阶段还跑不通端到端**。

---

## 4. P2 — CannTranslation 多产出 + family.json

**目标：** afir-translate 处理 module 里所有 variant func，emit N 个 .cpp + N 个 space.json + 1 个 family.json 索引。

### 4.1 接口设计

`afir-translate -mlir-to-cann` 现在 `--tiling-space-out=PATH` 是单文件。改成：
- 单 func module（无 `__v` 后缀）：保持原 behavior，单 .cpp + 单 space.json。
- 多 func module（含 `__v` 后缀的 funcs）：
  - 每 func 出独立 .cpp 文件，文件名 = `<funcName>.cpp`（注意：现在 `--output` 是单文件，要扩展成 `--output-dir`）。
  - 每 func 出独立 `<funcName>_space.json`。
  - 额外 emit `<familyId>_family.json`，索引所有 variants。

family.json schema（Q3 已对齐）：
```json
{
  "kernel_id": "kernel_group0",
  "variants": [
    {
      "id": "v0",
      "func_name": "kernel_group0__v0",
      "space_file": "kernel_group0__v0_space.json",
      "cpp_file": "kernel_group0__v0.cpp",
      "draft": {"ubTilingAxisY": 0, "ubTilingAxisR": -1, "reduceIsBlock": false}
    }
  ]
}
```

`draft` 字段从 module attr `vector_plan.tiling_infos[i]` 里拷过来；运行期不消费，纯 debug。

### 4.2 文件改动清单

| 文件 | 改动 |
|---|---|
| `lib/Target/CannKernel/CannTranslation.cpp` | walk module 找所有 `__v` 后缀 func；按 family id 分组；每 func emit .cpp + space.json；每 family emit family.json |
| `tools/afir-translate/afir-translate.cpp` | CLI 加 `--output-dir` 选项（跟 `--output` 互斥），用于多产出模式 |
| `python/network_runner.py` (phase 2) | 调 afir-translate 用 `--output-dir`；调 runtime-session 对每 variant 各跑一遍编译 |

### 4.3 P2 验收门

- `examples/autotune-stress-elewise` phase 2 跑完后，工作目录下能看到：
  - `kernel_group0__v0.cpp` + `kernel_group0__v0_space.json` + `kernel_group0_family.json`
  - `kernel_group1__v0.cpp` + `kernel_group1__v0_space.json` + `kernel_group1_family.json`
  - `artifacts/kernel_group0__v0/kernel_group0__v0.bin`
  - `artifacts/kernel_group1__v0/kernel_group1__v0.bin`
- `kernel_group0_family.json` 解析后有 1 个 variant entry，id=v0。

---

## 5. P3 — Autotuner `--family` 模式

**目标：** Autotuner 接 family.json，跨 variant + 跨 tile-size argmin。

### 5.1 接口设计

新 CLI 模式：
```
autotuner --family <kid>_family.json
          --inputs ...
          --expected ...
          --shape ...
          --output <kid>_best.json
          ...
```

跟现有 `--space` 模式互斥。内部：
1. 解析 family.json，遍历 variants。
2. 对每个 variant，加载它的 space.json + 对应的 .cpp，跑现有 `runSearch`（不动，复用），出 per-variant best。
3. Cross-variant argmin（语义 II：per-variant best 先选，再跨 variant 比 cycles）。
4. Emit `<kid>_best.json` 多一个字段：
```json
{
  "variant": "v1",
  "kernel_name": "kernel_group0__v1",
  "config": {...},
  "best": {"cycles": ..., "block_dim": ...},
  "all_variants": [
    {"variant": "v0", "best_cycles": 836132, "best_config": {...}},
    {"variant": "v1", "best_cycles": 471349, "best_config": {...}}
  ]
}
```

### 5.2 文件改动清单

| 文件 | 改动 |
|---|---|
| `tools/autotuner/autotuner_main.cpp` | 加 `--family` flag；family 解析；variant loop；cross-variant argmin；新输出字段 |

### 5.3 P3 验收门

- 在 `examples/autotune-stress-elewise` 上跑 autotuner --family，确认 best.json 有 `variant=v0`，cycles 跟之前 single-mode 一致。
- 临时构造 2-variant 情况，确认 cross-variant argmin 选了较快的那个。

---

## 6. P4 — aclnn-backend 路由 variant

**目标：** aclnn-backend 读 best.json 里的 variant id，emit 含后缀的 hostLaunch 调用。

### 6.1 改动点

`AclnnBackend` 现在用 `--tilings <tilings_default.json>` 拿 `{kernel_group0: {XBLOCK: 16, ...}}`，根据 kernel_id 在 network.json 里找 launch 信息。

P4 后：
- `--tilings` 仍传同样的 JSON，但 key 改成 variant kernel name（`kernel_group0__v0`）。
- 或者：`--tilings` 不变，额外 `--variants <family-mapping.json>` 提供 `{kernel_group0: "v0"}` 映射。
- aclnn-backend 内部把 kernel_id 拼上后缀，emit `hostLaunchAscendCKernel("kernel_group0__v0", ...)`。

**选哪个**：第一种简单，但 phase 3 (default build) 和 phase 5 (best build) 都要造 keyed-by-variant-name 的 tilings JSON。第二种更明确分离 family → variant 的选择。**倾向第一种**，phase 3/5 反正都要重 build tilings JSON。

### 6.2 文件改动清单

| 文件 | 改动 |
|---|---|
| `lib/Runtime/AclnnBackend/AclnnBackend.cpp` | 接受 variant-suffixed kernel name；拼 launch 调用 |
| `python/network_runner.py` (phase 3/5) | tilings_default.json / tilings_best.json 的 key 改成 variant-suffixed name |

### 6.3 P4 验收门

- `examples/autotune-stress-elewise` 生成的 `network_host_default.cpp` 里 hostLaunch 调用是 `kernel_group0__v0`。
- 跑 phase 3 不 crash，输出正确。

---

## 7. P5 — network_runner 编排

**目标：** 5-phase 全链路串起来。

### 7.1 各 phase 改动

**Phase 1（emit-network-json）**：不动。network.json 仍 family id。

**Phase 2（codegen）**：
- 调 afir-translate 用 `--output-dir`。
- 解析 family.json，每 variant 调 runtime-session 编译 .cpp → .bin。
- artifacts 目录每 variant 一份子目录。

**Phase 3（default build + dump）**：
- 对每 family，挑 v0 作为 default variant（暂时简单选 v0，等 P1 之后可以根据 cost ∞ 阈值做更聪明的 default）。
- 为 v0 算 default tilings（按现有 extent cap 算）。
- tilings_default.json 的 key 是 `kernel_group0__v0`（不是 `kernel_group0`）。
- 生成 host C++、链接、跑、dump 中间件。

**Phase 4（autotune）**：
- 对每 family 调 `autotuner --family`。
- 收集 best.json，写 tilings_best.json（key 是选中的 variant 名）。

**Phase 5（final build + verify）**：
- 用 tilings_best.json 重 link，run，verify。

### 7.2 文件改动清单

| 文件 | 改动 |
|---|---|
| `python/network_runner.py` | 所有 5 个 phase 都要适配 family / variant 命名 |
| `python/runner_utils/network_json.py` | Network 类加 `family_files()` 列举 family.json 路径 |
| `python/runner_utils/build_host.py` | 编译列表跟随 variant |

### 7.3 P5 验收门

- 6 个 e2e demo 全部 regression PASS：
  - `examples/mixed-attn-e2e`：max_diff 跟之前持平
  - `examples/two-elewise-e2e`：PASS
  - `examples/kg1-hang-repro`：PASS
  - `examples/autotune-stress-elewise`：PASS
  - `examples/reduce-sum-3d-f16-e2e`：跟之前持平
  - `examples/aclnn-attn-e2e`：跟之前持平（如适用）

---

## 8. P6 — 多 variant 验证

**目标：** 构造一个真有 ≥2 feasible draft 的场景，证明 cross-variant 选择真在工作。

### 8.1 Q1-(a) 临时降级 cost model

把 `costEstimate` 的两条 ∞ 规则之一拆开：
- "non-block ub-Y axis → ∞" 改成 "score = 1.5 × baseline"（仍倾向 block axis，但不再排除）。

对 2D elementwise（`autotune-stress-elewise` 32×64）：`enumerateTilingCases` 会产生：
- `(ubY=axis0, ubR=-1)` —— block 轴 = axis0
- `(ubY=axis1, ubR=-1)` —— block 轴 = axis0，ubY 用 axis1（即"行 vs 列"的两种 tile 方向）

之前 ∞ 选第一个；改完后两个都进 variants，autotuner 实测谁快。

### 8.2 期望观察

- `examples/autotune-stress-elewise` 跑下来：
  - `kernel_group0_family.json` 有 2 variants（v0, v1）。
  - `kernel_group0_best.json` 的 `variant` 字段可能是 v0 也可能是 v1（取决于实测 cycles），`all_variants` 字段含两个 score 对比。
- 端到端 max_diff 仍 0。

### 8.3 撤回 Q1-(a)

P6 验证完后**不**保留临时降级——把 `costEstimate` 改回 feasibility-only。P3b-3 落地后再让 cost model 真比较。或者把降级藏在另一个 flag 后面。

### 8.4 文件改动清单

| 文件 | 改动 |
|---|---|
| `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` | 临时降级 `costEstimate`（带 TODO 注释 / 或藏 flag） |
| `docs/superpowers/notes/2026-05-14-multi-plan-verification.md` | 新建：验证日志 + 实测 score 对比 |

### 8.5 P6 验收门

- `examples/autotune-stress-elewise` phase 4 输出 family best.json，含 `variant` + `all_variants` 字段。
- 撤回 Q1-(a) 后，所有 demo 仍 PASS。

---

## 9. 风险与回退

### 9.1 主要风险

1. **GroupOutline 跟不上多 variant func**：现在 outline 是 1 group → 1 private kernel func。多 variant func 在同 module 里，outline 是否能正确处理？P1 摸底必须 verify。
2. **Tilings JSON key 名改动 ripple**：从 `kernel_group0` 改成 `kernel_group0__v0`，aclnn-backend、autotuner、network_runner 都要同步。漏一处导致 phase 3/4/5 串不起来。**单进 PR**，整链路一次性改干净。
3. **CI / pytest 硬编码 artifact 路径**：P5 验收时要扫一遍 `python/tests/`。
4. **artifacts 目录爆炸**：多 variant + autotune profile 子目录 → 编译期 disk usage ×N。Q1 把 default 改成 on 后，所有 demo 都受影响。监控 build_e2e/ 总大小。

### 9.2 回退方案

每个子阶段独立 commit，可独立 revert。最坏情况下：
- revert P1 + P2 + P3 + P4 + P5 + P6 五个 commit
- 或：把 `--enable-tiling-variants` 默认改成 false（保留代码，禁用 default）

### 9.3 跟 P3b-3 的协调

P3b-3 触碰 `GroupOutlinePass`（1 group → 2 kernel funcs partial/combine）。本计划 P1-P5 不动 outline，但 outline 在多 variant 场景下是否能 fan-out（1 group × N variants × 2 kernel funcs = 2N output funcs）需要 P3b-3 设计 review 时一并考虑。

建议把 P3b-3 的 design review 推迟到本计划 P5 完成后——届时多 variant plumbing 已稳，P3b-3 在它上面继续盖。

---

## 10. 验收门汇总

| 阶段 | 验收 |
|---|---|
| P1 | autotune-stress-elewise lowered MLIR 含 `__v0` 后缀 sibling func；module attr 每 variant 一条 |
| P2 | 工作目录有 family.json + per-variant .cpp/space.json/.bin |
| P3 | autotuner --family 输出 best.json 含 variant 字段 |
| P4 | network_host_default.cpp 调用 `kernel_group0__v0` |
| P5 | 6 个 e2e demo regression 全过 |
| P6 | 构造 2-variant demo，best.json `all_variants` 字段有 2 个 score；撤回临时降级后仍 PASS |

---

## 11. 后续衔接

P5 完成后，下一步候选：

- **P3b-3 设计 review**：基于 multi-variant plumbing 设计 double-kernel partial+combine。
- **真 cost model（P6 of AF port plan）**：UB-peak / vectorized bytes 解析模型，用作 multi-variant 的预排序，autotuner 只 profile top-K。
- **Autotuner 候选生成器解耦**：把 `CannTranslation.cpp:2035` 的硬编码 `{16,...,256}` 拆出来，按 param role 生成（XBLOCK 跟核数、SUB 跟 UB、RBLOCK 跟 reduce budget）。
- **Multi-group reuse**（AF `PGOProfileReuseGroup`）：等多段 reduce 落地后再做。

---

参考：
- AF `AxesReorderPgoSolver` 对照：`/home/gser/code/ge-eco/.../autofuse/doc/AF知识地图/专题-AutotilingPGO.md`
- AF port 总计划：`docs/superpowers/plans/2026-05-11-port-af-scheduler-to-vector-plan.zh.md`
- P3b RCore 计划：`docs/superpowers/plans/2026-05-14-p3b-rcore-reduce-multicore.zh.md`
- Memory `[[network-runner-v1]]`、`[[af-scheduler-port]]`

---

## 12. 实施结果（2026-05-14）

**已完成 commit 链**：

| 阶段 | commit | 内容 |
|---|---|---|
| P1a | d102b36 | TileFuse 进入时改名 `__v0`；22 个 legacy 脚本批量改 `--name` |
| P2  | 10c53c1 | CannTranslation per-variant space.json + family.json |
| P3  | b18e2f8 | autotuner `--family` 模式 + cross-variant argmin |
| P4  | c9ca838 | aclnn-backend hostLaunch 加 `__v0` 后缀 |
| P5  | be80c8a | network_runner 5 phase 全 family/variant aware |
| P1b | 5b42cf4 | TileFuse ModuleOp pass + discovery + clone N + relaxNonBlockUbY |

**Regression matrix**（N=1，单 variant，每个 demo 走完整 multi-variant 链路）：

| Demo | 结果 |
|---|---|
| `mixed-attn-e2e` | `max_diff=0.0001221` PASS |
| `two-elewise-e2e` | `max_diff=0` PASS |
| `kg1-hang-repro` | `max_diff=0` PASS |
| `autotune-stress-elewise` | `max_diff=0` PASS |
| `relu-e2e`（legacy 路径采样）| `session.validation=pass` |

每个 demo 的 autotuner 日志显示 `Best variant: v0 kernel=<name>__v0 ...`，证明 family 模式整链路活跃。

**P6 多 variant 验证（部分完成）**：

新建 `examples/autotune-multivar-bcast/`（256×64 + dim-0 broadcast）。codegen 级 N=2 已确认：
- Lowered MLIR：`kernel_group0__v0`（block axis = N, axis_size=64）+ `kernel_group0__v1`（block axis = M, axis_size=256）
- `kernel_group0_family.json` 含 2 个 variant entry
- Phase 2 编译 2 个 .bin（`artifacts/kernel_group0__v0/...bin` + `__v1`）

**P6 e2e 阻塞**：phase 3 `_resolve_kernel_input_shape` 抛 `IndexError`，因 PackTilingData 把 scf.for iter_arg 的 argNumber 烤进 `dim_arg5_1` shape_key —— `arg5` 在 func 签名里是 TilingData struct，不是 tensor 输入。这是**预先存在**的 PackTilingData bug，跟 P1b 多 variant 无关，触发条件是"含 bcast + 多 axis 不全 collapse"。

修复方向（后续 follow-up）：PackTilingData 区分 func-arg 与 region-arg；canonicalize 时只接受 func-arg 的 argNumber 作为 shape_key。

## 13. 后续路线

| 任务 | 触发条件 | 优先级 |
|---|---|---|
| PackTilingData shape_key fix | 想真正 e2e 验证 P6 多 variant | 高（解锁 P6 实测） |
| P3b-3 设计 review | 完成 RCore double-kernel partial+combine | 中 |
| Cost model 真打分（P3b-4 / AF port P6） | 想用静态 cost 而不是 autotune 真测决定 variant | 低 |
| Multi-group reuse（AF `PGOProfileReuseGroup`）| 多段 reduce / 大融合图 | 低 |
