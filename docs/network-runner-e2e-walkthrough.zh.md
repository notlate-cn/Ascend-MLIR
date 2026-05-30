# network_runner 端到端流程 + 调试工具走查

> 配套阅读：`docs/network-runner-walkthrough.zh.md`（v1 内部细节，two-elewise 深挖）
>
> 适用范围：`develop @ 55ed59a2` 之后。包含本文初版（debug 工具 P0-1 / P0-2 / P1-3 / P1-4 / P1-5 / P2）以及 2026-05-28 后追加的：
> - `tensor.pad` 在 coordinator 的 whitelist（`9cc53ab7`）
> - Conv2D → aclnn fallback（`b80b68b4`）+ BatchNorm / Pool2D aclnn（`f755480b`）
> - 统一 Python logger + subprocess tee + run-id contextvars（`683de3e3`）
> - logger `context()` contextmanager + 跨 run append + PID-后缀 run-id（`55ed59a2`）
> - OpRoleClassifier lit 覆盖 4 case（`a19a40e3`）
> - ResNet-18 / GPT-2 small bring-up 全 e2e PASS on sim

本文以 BERT-tiny 为骨架例子（28 节点 DAG，混合 aclnn + AscendC 真实拓扑），讲清一个网络从 torch 到真机的五阶段流，以及每阶段你能看到哪些产物、用哪个工具看。

---

## 1. 总图

```
torch model ──┐
              │ examples/bert-e2e/export_bert.py
              │ (torch.export.export → torch_mlir.compile → linalg-on-tensors)
              ▼
   step0_linalg.mlir          ←【Phase 0】外部，不属 network_runner
              │
              │【Phase 1】outline
              │   --recognize-attention / --recognize-layernorm / --aclnn-finalize-decl
              │   --linalg-fold-unit-extent-dims --canonicalize
              │   --auto-fuse-group-analysis --auto-fuse-group-outline
              ▼
   model_recognized.mlir → model_unit_folded.mlir → _outlined_combined.mlir
   groups/network.mlir / network.json / network.provenance.json / kernel_group*.mlir
              │
              │【Phase 2】codegen + compile per AscendC kernel
              │   afir-opt --auto-fuse-codegen
              │   afir-translate --emit-cann
              │   runtime-session --kernel ... --output ...
              ▼
   kernel_groupN_lowered.mlir / kernel_groupN_kernel.cpp
   artifacts/kernel_groupN__v0/{out/manifest.txt, *.o, *.bin}
   kernel_groupN__v*_space.json
              │
              │【Phase 3】default-tile + dump intermediates
              │   aclnn-backend --tilings tilings_default.json
              │   network_test_default --dump-intermediates ...
              ▼
   tilings_default.json
   intermediates_default/<kid>__v0_{in,out}_<i>.npy
              │
              │【Phase 4】autotune (per kernel family，可跳过)
              ▼
   tilings_best.json
              │
              │【Phase 5】final-run + verify
              │   aclnn-backend --tilings tilings_best.json → network_test
              │   network_test --input ... --output ... [--profile-dir ...]
              ▼
   outputs/out<i>.npy + L0 PASS/FAIL
   profiles/<kid>__v0.timing.json
```

任一阶段产物都被 **`<workdir>/manifest.json`** 收录，并按 `<workdir>/stages/NN-<phase>-<file>` 编号 symlink 提供"按时间线浏览"视图（P0-1）。

---

## 2. 一行启动 + 全套 instrumentation

```bash
cd /home/gser/code/Ascend-MLIR
export ASCEND_HOME_PATH=/home/gser/Ascend/cann
source examples/env.sh

# A) 从 torch 模型导出（首次或参数变更时）
conda run -n torch-mlir python examples/bert-e2e/export_bert.py \
  --hidden 64 --heads 1 --seq 8 --dtype fp32 --outdir ~/bert-viz

# B) 跑 5 阶段 + 所有 debug 产物
python3 python/network_runner.py \
  --input-linalg ~/bert-viz/step0_linalg.mlir \
  --inputs       ~/bert-viz/input_0.npy \
  --expected     ~/bert-viz/expected_0.npy \
  --workdir      ~/bert-viz/work \
  --max-phase 5 --backend sim --atol 1e-2 --rtol 1e-2 \
  --debug-out --profile-dir profiles      # ← 新增

# C) 生成 DAG（任一 phase >= 1 后都可跑）
python3 python/tools/ascend_kernel_dag_viz/gen_dag_json.py ~/bert-viz/work

# D) 浏览器开
xdg-open ~/bert-viz/work/dag.html
```

`--debug-out` 写 `manifest.json` + `stages/`；`--profile-dir profiles` 在 phase-5 dump 每 kernel wall_us。两者都是可选 opt-in，不影响默认产物。

---

## 3. 逐阶段细节

### Phase 0：torch → linalg（外部）

工具：`examples/bert-e2e/export_bert.py`（用 torch-mlir compile pipeline）。

**重要事实**：torch-mlir 当前导出**不保留 FX node 名作 MLIR loc**——所有 linalg op 在 step0 里只有 `loc(unknown)` 或纯文件位置。这是 P2 走 sim-as-reference 而不是 torch-hook 自动 mapping 的根本原因（loc 没锚点）。

**产物**：
- `step0_linalg.mlir` — linalg + tensor + arith dialect
- `input_0.npy` / `expected_0.npy` — 随机种子固定的输入 + torch ref 输出

### Phase 1：outline

```bash
# step a0: 识别 attention / layernorm / transpose 等高级 pattern → 折成 aclnn 调用
afir-opt --recognize-attention --recognize-layernorm --aclnn-finalize-decl \
         step0_linalg.mlir -o model_recognized.mlir

# step a: unit-dim 折叠 + canonicalize（去 identity copy）
afir-opt --linalg-fold-unit-extent-dims --canonicalize \
         model_recognized.mlir -o model_unit_folded.mlir

# step b: group analysis + outline
afir-opt --auto-fuse-group-analysis=disable-cube-fusion=true \
         --auto-fuse-group-outline=output-dir=groups/ \
         model_unit_folded.mlir -o _outlined_combined.mlir
```

**核心智能 = group analysis**：
- 已被 `recognize-*` 折成 aclnn 的（attention/LN/transpose/matmul）→ **单 op 一个 kernel**，标 `aclnn.op` attr
- 剩下的纯 linalg elementwise / reduce → 按 axis 兼容性 + producer-consumer 局部性合 **AscendC fused vector kernel**
- Cube ops（matmul / batch_matmul）默认走 aclnn fallback（`disable-cube-fusion=true`）

**产物**（在 `<work>/groups/`）：

| 文件 | 用途 | 谁读 |
|---|---|---|
| `network.mlir` | 剥离了 kernel body 的 coordinator IR | phase 2 / aclnn-backend |
| `network.json` | 编译/运行契约：kernel id / kind / args / results / 拓扑 | 主链路 + DAG viz |
| `network.provenance.json` | ★ debug-only：每 kernel 的 source_ops + op_role + loc + fused_ops_summary | DAG viz / ascend_diff locate |
| `kernel_group<N>.mlir` | 每个 AscendC fused kernel 的孤立 IR | phase 2 codegen |

**怎么看（P1-3 + P1-4）**：

```bash
# 高层概览：DAG 可视化
xdg-open <work>/dag.html
# 点节点 → 右栏出 source_ops + op_role + loc + result shape

# 文本概要：fused_ops_summary 一眼读
jq -r '.kernels[] | "\(.kernel_id) [\(.kind)] \(.fused_ops_summary)"' \
   <work>/groups/network.provenance.json

# 找具体源 op：
jq -r '.kernels[] | select(.kernel_id == "kernel_group20") | .source_ops' \
   <work>/groups/network.provenance.json
# 输出包含 loc — 直接 vim +<line> step0_linalg.mlir 跳转
```

### Phase 2：per-kernel codegen + compile

对每个 `kernel_group<N>.mlir`：

```
afir-opt   --auto-fuse-codegen        # linalg-on-tensors → AscendC IR
afir-translate --emit-cann            # AscendC MLIR → C++ kernel source
runtime-session --kernel ...          # 编译 .o + manifest，落 artifacts/<kid>__v0/
```

变体（`__v0` / `__v1` / ...）来自 **multi-variant TilePlanGen**：同一 kernel 的不同 tile 计划生成多个 func，供 autotune 选最佳。

`*_space.json` 是 tile 参数搜索空间 schema：每参数的候选 + shape 等式约束。

**产物**：
- `kernel_group<N>_lowered.mlir` — AscendC dialect 中间态
- `kernel_group<N>_kernel.cpp` — 真生成的 AscendC 内核代码
- `artifacts/kernel_group<N>__v<i>/` — 编译产物（`.o` + `.bin` + manifest.txt）
- `kernel_group<N>__v<i>_space.json` — tile 搜索空间

### Phase 3：default-tile + dump intermediates

每 AscendC kernel 用 schema 的"默认 tile"配置（通常是最保守的能跑通的），先跑一遍 sim 把每 kernel 的输入输出 dump 成 npy。这步**为 autotune 提供 golden**，也是 **P2 locate 的 reference 数据源**。

```bash
aclnn-backend --input network.mlir --output network_host_default.cpp \
              --tilings tilings_default.json --kernel-binaries artifacts/
g++ network_host_default.cpp harness.cpp ... -o network_test_default
network_test_default --input <inputs> --output <outputs> \
                     --dump-intermediates intermediates_default/
```

**产物**：
- `tilings_default.json` — `{kernel_name: {param: value, _block_dim: N}, ...}`
- `intermediates_default/<kid>__v0_{in,out}_<i>.npy`

### Phase 4：autotune（可跳）

```bash
autotuner --schema <kernel_space.json>
          --tilings <candidate>
          --golden  intermediates_default/<kid>_out_<i>.npy   # phase-3 dump 当 golden
```

对每个 AscendC kernel family，遍历 tile 候选 → 跑 sim → 与 golden 比对（精度门控）→ 选时间最短的通过候选。跨 kernel 不联合优化。

**跳过开关**：`NETWORK_RUNNER_SKIP_AUTOTUNE=1` 直接拿 default tilings 走 phase 5（accuracy-only）。

**产物**：`tilings_best.json`

### Phase 5：final-run + verify

```bash
aclnn-backend --tilings tilings_best.json → network_host.cpp → network_test
network_test --input ... --output outputs/out<i>.npy \
             [--profile-dir profiles/]   # ← P1-5：per-kernel wall_us
```

phase-5 内嵌的 `np.allclose` 跟 `--expected` 比对，**给出最终 PASS/FAIL** 与退出码。

**产物**：
- `outputs/out<i>.npy`
- `profiles/<kid>__v0.timing.json` — `{kernel, wall_us, block_dim, num_inputs, num_outputs, backend}`

---

## 4. Debug 工具与典型工作流

### 4.1 工具一览

| 工具 | commit | 何时用 |
|---|---|---|
| `<work>/manifest.json` + `stages/` | P0-1 | 按时间线浏览产物，自动 |
| `python/tools/ascend_diff.py` (default = L0) | P0-2 | 最终输出 PASS/FAIL 判决 |
| `<work>/groups/network.provenance.json` | P1-3 | kernel ↔ source op 映射，自动生成 |
| `python/tools/ascend_kernel_dag_viz/` | P1-4 + viz fix | 拓扑可视化（HTML+JS） |
| `profiles/<kid>.timing.json` | P1-5 | per-kernel wall_us，opt-in `--profile-dir` |
| `ascend_diff.py locate` | P2 | first-bad-kernel 拓扑定位 |

### 4.2 工作流 A：最终输出挂了

L0 diff 是最快的判决：

```bash
python3 python/tools/ascend_diff.py \
  --actual   ~/bert-viz/work/outputs/out0.npy \
  --expected ~/bert-viz/expected_0.npy \
  --atol 1e-3 --rtol 1e-3
# rc=0 → PASS；rc=1 → FAIL，输出 max_diff / mean_diff / first_mismatch_idx
```

### 4.3 工作流 B：sim PASS / NPU FAIL（最常见 bring-up 场景）

跑两遍——一遍 sim、一遍真机——各自 dump intermediates，喂给 `locate`：

```bash
# 1) sim run（trusted reference）
python3 python/network_runner.py --backend sim ... --workdir sim_run --debug-out

# 2) NPU run（suspect）
python3 python/network_runner.py --backend npu ... --workdir npu_run --debug-out

# 3) 拓扑扫 first-bad-kernel
python3 python/tools/ascend_diff.py locate \
  --network    sim_run/groups/network.json \
  --reference  sim_run/intermediates_default \
  --actual     npu_run/intermediates_default \
  --provenance sim_run/groups/network.provenance.json \
  --profiles   npu_run/profiles \
  --atol 1e-3 --rtol 1e-3

# 输出形如：
#   kernel_group2 [add] PASS  (max_diff=0)
#   kernel_group6 [add] FAIL  (max_diff=0.08)  [FIRST-BAD]
#   ---
#   first-bad-kernel: kernel_group6
#     source op op_007: add  loc("step0_linalg.mlir":42:11)
```

**aclnn kernel 在 locate 里被跳过**——HostLaunchHelper 不 dump aclnn 中间值。aclnn-heavy 网络（如 BERT-tiny 26 kernel 中 20 个 aclnn）的 locate 覆盖率有限，这是 P2 plan 里就明说的局限。

### 4.4 工作流 C：性能分析

```bash
# 按耗时倒序
jq -s 'sort_by(.wall_us) | reverse' ~/bert-viz/work/profiles/*.timing.json

# 提示：sim 上的 wall_us = camodel CPU 模拟时间，不代表 NPU 真实性能。
# NPU 上的 wall_us 包含 dispatch overhead，是上界，不是纯 kernel time。
# 真要做精细 perf 分析需要 aclrtRecordEvent，目前是 deferred。
```

### 4.5 工作流 D：按时间线读 IR

`--debug-out` 后：

```bash
ls ~/bert-viz/work/stages/
# 10-outline-model_recognized.mlir
# 10-outline-model_unit_folded.mlir
# 10-outline-_outlined_combined.mlir
# 10-outline-network.json
# 10-outline-kernel_group0.mlir
# 20-codegen-kernel_group0_lowered.mlir
# 20-codegen-kernel_group0_kernel.cpp
# 30-default-build-dump-tilings_default.json
# 30-default-build-dump-intermediates_default
# 40-autotune-tilings_best.json
# 50-runtime-network_host.cpp
# 50-runtime-network_test
# 50-runtime-outputs
# 50-runtime-profiles
```

每条都是 symlink 指向真文件，编号自然排序，对应 `manifest.json` 的 phase 顺序。

### 4.6 工作流 E：拓扑可视化交叉对照

`xdg-open <work>/dag.html` 打开后：

- **颜色编码**：绿 input / 粉 output / 橙 aclnn / 蓝 ascendc
- **节点标题**：`kernel_id [fused_ops_summary]` —— BERT-tiny 上可以一眼看到 `kernel_group20 [add+elementwise_chain]` 这种
- **点击节点 → 右栏侧 panel**：source_ops 列表（每个含 op_role + loc）、output checkpoint hint、result shape

dag.html 是 **单文件内联**（dag.json 已 embed），直接 `file://` 打开，不需要 http server，不受代理影响。CDN（cytoscape.js / dagre）仍走 unpkg，需联网。

---

## 5. BERT-tiny 拓扑示例

`xdg-open ~/bert-viz/work/dag.html` 实际看到的 kernel 分布：

| kind | 数量 | op_role 分布 |
|---|---|---|
| **aclnn** | 20 | 11× Transpose · 6× Matmul · 2× LayerNorm · 1× FlashAttentionScore |
| **ascendc fused** | 6 | 3× `add` · 2× `add+add` · 1× `add+elementwise_chain` |

input → out 的关键路径：

```
arg0 ──→ Transpose(group7) ──┬──→ Matmul(group5) ──→ add(group6) ──→ Transpose(group11)──┐
                              │                                                            ├──→ FlashAttentionScore
                              ├──→ Matmul(group8) ──→ add(group9) ──→ Transpose(group10)─┤
                              └──→ Matmul(group1) ──→ add(group2) ──→ Transpose(group3)──┘
                                                                                          ↓
                                                                                  Transpose(group13)
                                                                                          ↓
                                                                                    Matmul(group15)
                                                                                          ↓
                                                                                   add+add(group16)
                                                                                          ↓
                                                                                       LayerNorm
                                                                                          ↓
                                                                                ...继续 FFN + 第二 LN...
                                                                                          ↓
                                                                                       out0
```

注意所有 Q/K/V 都是 Transpose → Matmul → add 的并行三股，对应 attention 三投影 + bias 加和。fused kernel 主要是 bias add（`add` / `add+add`）和 FFN 内的 GELU 链（`add+elementwise_chain`）。

---

## 6. 常见坑 + 排查路径

| 现象 | 第一步看 |
|---|---|
| phase 1 SIGABRT | `_outlined_combined.mlir` 有没有；group analysis 报错从 stderr 拿 |
| phase 2 编译失败 | `kernel_group<N>_lowered.mlir` 看 IR；`artifacts/<kid>__v<i>/out/manifest.txt` |
| phase 3 sim 卡死 | 多半 tile XBLOCK > extent，回退 default tilings 看 schema；MEMORY 里"multi-input dyn reduce bug" 类 |
| phase 4 autotune 全 miss | 用 `NETWORK_RUNNER_SKIP_AUTOTUNE=1` 走默认 tile；查 `kernel_<kid>_space.json` 候选数 |
| phase 5 final PASS 但 max_diff 大 | 容差太松，先 `ascend_diff --atol 1e-5` 精确判 |
| sim PASS 但 NPU FAIL | **工作流 B**（4.3）—— sim 当 reference 找 first-bad-kernel |
| 性能不达标 | 工作流 C；先看 wall_us 分布，再决定要不要换 tile / 重 autotune |

---

## 7. 当前局限（明确暂缓）

- **L2/L3 semantic group diff**：依赖 provenance + 重编译能力，第一版未做
- **aclnn kernel 无 L1 ref**：HostLaunchHelper 不 dump aclnn 中间值，locate 覆盖不到
- **真 NPU per-kernel perf**：当前 wall_us 含 dispatch overhead，纯 kernel time 需要 `aclrtRecordEvent`，deferred
- **四层 report**（Kernelize / Schedule / Realize / Translate 决策结构化）尚无
- **自动跑两遍**：sim + npu 的成对运行目前要手工触发，未来可在 network_runner 加 `--ref-from sim` 一键模式

---

## 8. 关键文件路径速查

```
python/network_runner.py                                   # 主驱动
python/tools/ascend_diff.py                                # L0 diff + L1 locate
python/tools/ascend_kernel_dag_viz/gen_dag_json.py         # DAG viz 生成器
python/tools/ascend_kernel_dag_viz/viewer.html             # 模板
python/runner_utils/harness.cpp                            # phase 5 host binary
python/runner_utils/build_host.py                          # g++ link

lib/Conversion/AutoFuse/GroupOutline/NetworkJsonEmitter.cpp    # network.json + provenance
lib/Conversion/AutoFuse/GroupOutline/OpRoleClassifier.cpp      # op_role 分类
lib/Conversion/AutoFuse/GroupOutline/GroupOutlinePass.cpp      # outline 编排
lib/Dialect/AFIR/Transforms/EmitNetworkJsonPass.cpp            # 独立 emit-network-json pass
lib/Runtime/Execution/HostLaunchHelper.cpp                     # phase 5 kernel launcher + timing + dump

examples/bert-e2e/run.sh / export_bert.py                  # BERT-tiny e2e
examples/two-elewise-e2e/run.sh                            # 最小 e2e
```

---

## 9. 已验证通过的网络（截至 2026-05-29）

| 网络 | sim 全 e2e | 真 NPU 全 e2e | 关键修复 commit |
|---|---|---|---|
| two-elewise | ✓ | — | — |
| autotune-stress-elewise | ✓ | — | — |
| **encoder**（attention 块）| ✓ max_diff=7.15e-7 | ✓ | MEMORY [GroupAnalysis fusion bugs] 4 commit |
| **BERT-tiny** | ✓ | ✓ max_diff=1.4e-5（910C）| MEMORY [BERT bring-up] `652088a2` |
| **ResNet-18**（CNN）| ✓ | — | `9cc53ab7` pad + `b80b68b4` Conv→aclnn + `f755480b` BN/Pool aclnn |
| **GPT-2 small**（12 层 124M）| 通到 phase 3，phase 4-5 未跑 | — | `b7f222cc` |

## 10. 关键已知改进机会（按价值排）

按 BERT-tiny / GPT-2 实测得出的融合分析结论：

| 改进 | 当前痛点 | 影响 | MEMORY 锚 |
|---|---|---|---|
| **CV fusion**（matmul + bias 进 cube epilogue）| GPT-2 13 退化 add + BERT 3 退化 add | -13 dispatch / GPT-2，aclnn matmul 后的 bias 都能融 | [AF CV fusion port plan]，~750 LOC MVP stub 在 `GroupAnalysisPass.cpp:147-150` |
| **Transpose-eliminate**（perm 传 consumer）| 12 个 transpose 链 / GPT-2 | -12~30 dispatch | [GroupAnalysis fusion bugs] future 段，AF doc `transpose全流程说明.md` 有原型 |
| **aclrt event per-kernel timing** | wall_us 含 dispatch overhead | 真 NPU perf 信号 | [debug.md §7.6 L1-perf]，本文 §7 deferred |
| **C++ 侧 report 框架**（Realize / Schedule decision 结构化）| 当前末端只有 IR + stderr | 替代第四类调试维度 | [debug.md §7.9] |

## 11. 进一步阅读

- `docs/network-runner-walkthrough.zh.md` — v1 内部细节，two-elewise 深挖（编号、ID 体系、phase 4 autotune 内部算法）
- `docs/auto-fuse/debug.md` §7 — debug 规范的原始设计文档（本调试工具链的设计依据 + 与提议的差距说明）
- `MEMORY.md` → `[BERT bring-up]` / `[Encoder PASSES]` / `[transpose+glue B2 crash]` / `[AF CV fusion port plan]` 等条目 — 真机 bring-up 已发现并修复的具体 bug 案例及未来改进 plan
