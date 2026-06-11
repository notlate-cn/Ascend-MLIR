# network-runner v1 端到端流水线讲解

以 `examples/two-elewise-e2e/` 为例,串讲 `python/network_runner.py` 五个阶段从 MLIR 到 camodel CPU sim 验证的全过程。

---

## 1. 背景与目标

### 1.1 解决的问题

我们手里有一个 MLIR 网络（`func.func @model`），里面是 linalg / aclnn 的混合算子。要把它在 camodel CPU sim 上跑通端到端，最少做四件事：

1. 把网络切成若干 AscendC kernel + aclnn 调用。
2. 为每个 AscendC kernel 走 codegen → -mlir-to-cann → 编译，拿到 `.bin`。
3. 拼一份 host C++（依次调用 kernel + aclnn），用 g++ 链出可执行文件。
4. 跑出来，对账 numpy 期望值。

`network_runner.py` 把这四件事拆成 5 个 phase，加了一个 autotune 环节。

### 1.2 v1 的边界

- **不拉硬件**：默认强制 aclnn 走 host-mode CPU 参考实现，AscendC 部分走 camodel sim。真硬件路径靠 `NETWORK_RUNNER_REAL_ACLNN=1` 切回去，目前没在卡上验过。
- **不做图优化**：phase 1 之前的 IR 我们认账，不重排算子、不做 layout 选择。
- **不做调度**：autotuner 是一个 kernel 内部的笛卡尔积穷举 + 剪枝，不跨 kernel。

### 1.3 为什么挑 two-elewise

- IR 只有 29 行（4×4 fp16 两个独立 elementwise），所有产物都能逐个看完。
- 真正走 `--vector-plan-group-outline` 自动切 kernel，跟 `examples/mixed-attn-e2e/` 的手写 `network.mlir` 路径形成对照。
- 4×4 = 16 个元素 = 最小的 XBLOCK 候选，避开 `block_dim_expr=""` 这条 latent issue（见 §5.4）。

---

## 2. 示例代码与运行

### 2.1 `model.mlir`

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @model(%a: tensor<4x4xf16>, %b: tensor<4x4xf16>,
                  %c: tensor<4x4xf16>, %d: tensor<4x4xf16>,
                  %i0: tensor<4x4xf16>, %i1: tensor<4x4xf16>)
    -> (tensor<4x4xf16>, tensor<4x4xf16>) {
  %x = linalg.generic { ... } ins(%a, %b) outs(%i0) {
    ^bb0(%p, %q, %o): %v = arith.addf %p, %q : f16
                       linalg.yield %v : f16
  } -> tensor<4x4xf16>
  %y = linalg.generic { ... } ins(%c, %d) outs(%i1) {
    ^bb0(%p, %q, %o): %v = arith.mulf %p, %q : f16
                       linalg.yield %v : f16
  } -> tensor<4x4xf16>
  return %x, %y : tensor<4x4xf16>, tensor<4x4xf16>
}
```

### 2.2 为什么必然切成两个 kernel

`lib/Conversion/VectorPlan/GroupAnalysis/CanFuse.cpp` 里 `getFusionKind` 的判定：

```cpp
if (hasSSAEdge(g1, g2)) return FusionKind::Vertical;     // 有 SSA 边
for (auto v : g1.boundaryIn)
  if (g2.boundaryIn.contains(v)) return FusionKind::Horizontal;  // 共享输入
return FusionKind::None;
```

- `%x = a + b` 与 `%y = c * d`：没有 SSA 边（`%y` 不消费 `%x`）。
- 输入 `{%a,%b,%i0}` 与 `{%c,%d,%i1}` 完全不相交，没有共享 boundaryIn。
- 结论：`FusionKind::None`，outline pass 必然切成两个 kernel。

### 2.3 调用形态

```bash
bash examples/two-elewise-e2e/run.sh
```

`run.sh` 干的事：source CANN 环境 + 注入 sim 库的 `LD_LIBRARY_PATH` + 跑 `gen_inputs.py` 造 npy + 启动 `network_runner.py`。

### 2.4 期望尾部输出

```
[harness] wrote output[0] → .../outputs/out0.npy
[harness] wrote output[1] → .../outputs/out1.npy
network.output[0]: max_diff=0  PASS
network.output[1]: max_diff=0  PASS
```

---

## 3. 五阶段流水线总览

```
                 model.mlir  (linalg)
                     │
   ┌─────────── Phase 1: outline ────────────┐
   │  --vector-plan-group-analysis           │
   │  --vector-plan-group-outline            │
   └─────────────────┬───────────────────────┘
                     ▼
       groups/{network.mlir, network.json,
               kernel_group0.mlir, kernel_group1.mlir}
                     │
   ┌─────── Phase 2: 每个 ascendc kernel ────┐
   │  --vector-plan-codegen                  │
   │  -mlir-to-cann   (产 .cpp + _space.json)│
   │  runtime-session (产 .bin + manifest)    │
   └─────────────────┬───────────────────────┘
                     ▼
       artifacts/<kid>/{<kid>.bin, out/manifest.txt}
                     │
   ┌─── Phase 3: default tilings + dump ─────┐
   │  挑最大 XBLOCK → tilings_default.json    │
   │  aclnn-backend → network_host_default.cpp│
   │  g++ link → network_test_default         │
   │  跑一遍并 --dump-intermediates           │
   └─────────────────┬───────────────────────┘
                     ▼
       intermediates_default/<kid>_in_*.npy
                                 _out_*.npy   ← phase 4 oracle
                     │
   ┌────────── Phase 4: autotune ────────────┐
   │  per-kernel 笛卡尔积 + block_dim>32 剪枝 │
   │  → kernel_group{0,1}_best.json           │
   │  → tilings_best.json                     │
   └─────────────────┬───────────────────────┘
                     ▼
   ┌─── Phase 5: best tilings + 终验 ────────┐
   │  aclnn-backend → network_host.cpp        │
   │  g++ link → network_test                 │
   │  跑一遍 → outputs/out{0,1}.npy           │
   │  np.allclose vs --expected               │
   └─────────────────────────────────────────┘
```

**关键点**：phase 3 跑一次拍下中间 tensor 给 phase 4 当 oracle —— 这是循环依赖（"想知道 tile 是否正确，得先有正确结果"）的破法。我们承认 phase 3 的 default tiling 一定算得对（因为它是 schema 默认），用它 dump 出每个 kernel 的输入和输出 npy，autotune 阶段就有黄金参考了。

---

## 4. 逐阶段精讲

`build_e2e/` 是已经跑过一次的产物，下文路径都基于它。

### 4.1 Phase 1 — outline / emit-network-json

**输入**：`examples/two-elewise-e2e/model.mlir`

**两条路径**（`network_runner.py:39 phase1_outline_or_emit_json`）：

| 入口 | 触发 | 用途 |
|------|------|------|
| `--input-linalg model.mlir` | `--vector-plan-group-analysis` + `--vector-plan-group-outline` | 自动切（本例） |
| `--input-network DIR/` | 拷贝 `*.mlir` + `--emit-network-json` | 手写 network（mixed-attn-e2e） |

**命令**（自动路径）：

```bash
afir-opt --linalg-fold-unit-extent-dims model.mlir -o model_unit_folded.mlir
afir-opt --vector-plan-group-analysis \
         --vector-plan-group-outline=output-dir=$WORK/groups \
         model_unit_folded.mlir -o _outlined_combined.mlir
```

第一步把 unit 维折掉。第二步是核心：group-analysis 给每个 linalg op 打 `vector_plan.group_id`；outline pass 把同组算子提成 private func，原 func 改成依次 call。

**产出**：`build_e2e/groups/`

- `network.mlir`：协调器（callable 框架）
  ```mlir
  func.func private @kernel_group0(t<4x4xf16>, t<4x4xf16>, t<4x4xf16>) -> t<4x4xf16>
  func.func private @kernel_group1(t<4x4xf16>, t<4x4xf16>, t<4x4xf16>) -> t<4x4xf16>
  func.func @model(%arg0..%arg5) -> (t<4x4xf16>, t<4x4xf16>) {
    %0 = call @kernel_group0(%arg0, %arg1, %arg4) : ...
    %1 = call @kernel_group1(%arg2, %arg3, %arg5) : ...
    return %0, %1
  }
  ```
- `kernel_group0.mlir`、`kernel_group1.mlir`：每个 kernel 一份独立文件。
- `network.json`：下游所有阶段的 single source of truth。`kernels[]` 关键字段：
  ```json
  {"id":"kernel_group0","kind":"ascendc","file":"kernel_group0.mlir",
   "args":[{"from":"input","name":"arg0"},
           {"from":"input","name":"arg1"},
           {"from":"input","name":"arg4"}],
   "results":[{"name":"kernel_group0_r0","dtype":"f16","shape":[4,4]}]}
  ```
  `from` 可以是 `input`（取自 network 输入）或 `kernel`（消费另一个 kernel 的 result）。`outputs[]` 同理指向 `kernel.result`。

**易踩坑**：outline 出来的 kernel 默认全是 `kind: ascendc`。aclnn 节点只可能从手写 network.mlir + `--emit-network-json` 路径产生（看 op 名前缀 `__aclnn_`）。

### 4.2 Phase 2 — codegen + compile

**输入**：`groups/<kid>.mlir`，每个 ascendc kernel 一份。

**命令**（per kernel；见 `network_runner.py:152`）：

```bash
afir-opt --vector-plan-codegen kernel_group0.mlir -o kernel_group0_lowered.mlir
afir-translate -mlir-to-cann kernel_group0_lowered.mlir \
               -o kernel_group0.cpp \
               --tiling-space-out=kernel_group0_space.json
runtime-session --kernel kernel_group0.cpp --kernel-kind vec \
                --output artifacts/kernel_group0 --name kernel_group0
```

三段式：MLIR codegen → AscendC C++ + tiling space → 驱动 ascendc kernel 编译器，最终落到 `.bin`。

**产出**：

- `kernel_group0_lowered.mlir`（≈8 KB AscendC dialect）
- `kernel_group0.cpp`（生成的 device kernel C++）
- `kernel_group0_space.json`：
  ```json
  {
    "block_dim_expr": "",
    "kernel": "kernel_group0",
    "soc": "Ascend910B1",
    "tiling_params": [
      {"name":"XBLOCK",     "fixed":false, "values":[16]},
      {"name":"XBLOCK_SUB", "fixed":false, "values":[16]}
    ]
  }
  ```
- `artifacts/kernel_group0/`：
  - `kernel_group0.bin` — device 二进制
  - `out/manifest.txt` — 启动元数据（kernel_name / soc_version / kernel_kind / device_binary_path）

**关键设计点**：

- `_space.json` 的 `block_dim_expr` 是 phase 3/4 计算 `_block_dim` 的表达式（grammar 见 `eval_block_dim`，支持 `+−*/` + `ceil(a/b)`）。本 case 是空字符串，走静态 fallback `_block_dim=1`。这是已知 latent issue：outline 出来的 elementwise kernel 这一栏目前都是空的，导致大 shape 时只算第一个 tile 而其余位置全为 0。本例 4×4=16 ≤ 最小 XBLOCK 16，正好能在一个 tile 里算完，所以不暴露。
- `tiling_params[].values` 是搜索空间。本例每个参数只有一个候选，所以 phase 4 搜索空间 = 1×1 = 1。

### 4.3 Phase 3 — default tilings + dump intermediates

**目的**：选一份保证算对的 tiling，跑出每个 kernel 的真实 in/out npy，喂给 phase 4。

**default tiling 怎么取**（`network_runner.py:230`）：

```python
for p in space.get("tiling_params", []):
    if not p.get("fixed", False):
        vals = p.get("values", [])
        params[p["name"]] = vals[-1] if vals else 16   # ← 最大 XBLOCK
```

挑 `values[-1]`（最大值）是有意为之：XBLOCK 越大 → 每 block 处理元素越多 → `_block_dim = ceil(N/XBLOCK)` 越小。camodel sim 只有 ~32 核，超过就报错或挂；小一点更安全。

**aclnn-backend 干什么**（`lib/Runtime/AclnnBackend/AclnnBackend.cpp`）：

读 `network.mlir` + `tilings_default.json` + `--kernel-binaries DIR`，产出 `network_host.cpp`。核心片段（`build_e2e/network_host.cpp:15-51`）：

```cpp
static void network_impl(TensorInfo inputs[], int, TensorInfo outputs[], int,
                         aclrtStream stream) {
  TensorInfo t0[3] = {inputs[0], inputs[1], inputs[4]};
  TensorInfo t1[1] = {};
  t1[0].rank = 2; t1[0].shape[0] = 4; t1[0].shape[1] = 4; t1[0].dtype = 1;
  { size_t _n = 2; for (int _d = 0; _d < t1[0].rank; ++_d) _n *= t1[0].shape[_d];
    t1[0].data = ::operator new(_n); }
  if (mlir::runtime::hostLaunchAscendCKernel(
        "kernel_group0",
        /*kernelBinariesDir=*/".../artifacts",
        /*tilingsPath=*/".../tilings_best.json",
        t0, 3, t1, 1) != 0) { /*err*/ return; }
  // ... 同样的 pattern 调 kernel_group1 → t3 ...
  outputs[0] = t1[0];
  outputs[1] = t3[0];
}
```

每个 kernel 的 `kernelBinariesDir`、`tilingsPath` 直接 hardcode 到生成的 .cpp 里。`hostLaunchAscendCKernel` 是 `lib/Runtime/Execution/HostLaunchHelper.cpp` 提供的 `extern "C"` 入口（详见 §5.1）。

**g++ link**（`python/runner_utils/build_host.py`）：

把 `harness.cpp` + `network_host_default.cpp` 一起编，链 `libAscendCRuntime` + CANN 的 `ascendcl` / `ascend_hal`。

**跑**：

```bash
$WORK/network_test_default \
    --input a.npy --input b.npy --input c.npy --input d.npy \
    --input init0.npy --input init1.npy \
    --output output_default_0.npy --output output_default_1.npy \
    --dump-intermediates $WORK/intermediates_default
```

**产出**：

- `network_host_default.cpp` / `network_test_default`
- `output_default_{0,1}.npy`
- `intermediates_default/`：
  ```
  kernel_group0_in_{0,1,2}.npy   ← phase 4 inputs
  kernel_group0_out_0.npy         ← phase 4 expected
  kernel_group1_in_{0,1,2}.npy
  kernel_group1_out_0.npy
  ```

**易踩坑**：`--output` 数量必须等于 network outputs 数。harness 的 `outputs` vector 是按 `--output` 数量预分配的，少了 `network_impl` 写越界、收尾 SIGSEGV（见 `network_runner.py:286` 的注释和 `harness.cpp:266`）。

### 4.4 Phase 4 — autotune

**目的**：在 phase 2 给的 tiling 搜索空间里，找出 cycle 最少 + 数值正确的那一组。

**调用**（per kernel；`network_runner.py:335`）：

```bash
autotuner --space   kernel_group0_space.json \
          --kernel  kernel_group0.cpp \
          --inputs  kernel_group0_in_0.npy,kernel_group0_in_1.npy,kernel_group0_in_2.npy \
          --expected kernel_group0_out_0.npy \
          --shape   arg0_dim0=4,arg0_dim1=4,...   (从 dumped npy 推) \
          --output  kernel_group0_best.json \
          --profile-out kernel_group0_autotune_profile/
```

autotuner 内部：穷举 `tiling_params[].values` 笛卡尔积 → 用 `block_dim_expr` 算 `block_dim`，>32 直接剪枝 → 编译 + 跑 sim → 跟 expected npy 对，记录 cycle_count → 选 cycle 最少且通过对账的。

**产出**（`kernel_group0_best.json`）：

```json
{
  "best": {
    "block_dim": 1,
    "cycle_count": 646552,
    "max_abs_diff": 0,
    "score": 646552,
    "profile_path": ".../candidate.json"
  },
  "config": {"XBLOCK": 16, "XBLOCK_SUB": 16},
  "device_binary_path": "/tmp/autotuner_build-.../kernel_group0.bin",
  "manifest_path": "/tmp/autotuner_build-.../out/manifest.txt"
}
```

聚合后：`tilings_best.json`：

```json
{
  "kernel_group0": {"XBLOCK":16,"XBLOCK_SUB":16,"_block_dim":1},
  "kernel_group1": {"XBLOCK":16,"XBLOCK_SUB":16,"_block_dim":1}
}
```

**易踩坑**：autotuner `--expected` 是单数。多输出 AscendC kernel 在 v1 不支持。

**本 case 特殊**：搜索空间 1×1=1 → best 必然 == default。但 `_best.json` 的字段（cycle / max_abs_diff / score）展示了机制本身。

### 4.5 Phase 5 — best tilings + 终验

跟 phase 3 同一条链路，只是：

- `--tilings tilings_best.json`（不是 default）
- 产出叫 `network_host.cpp` / `network_test`（不带 `_default`）
- 跑完不 dump intermediates，而是把 `outputs/out{0,1}.npy` 拿来 `np.allclose(atol=rtol=1e-2)` 对账 `--expected`

最后逐个 print：

```
network.output[0]: max_diff=0  PASS
network.output[1]: max_diff=0  PASS
```

任一 FAIL → exit code 非零。

---

## 5. 关键设计决策与权衡

### 5.1 HostLaunchHelper 为什么走 ExecutionSession::run(TaskGraph)

`hostLaunchAscendCKernel` 内部不直接调 `NativeExecutionRunner::runFile`，而是把每次 launch 当成一个 TaskGraph 节点喂给 `ExecutionSession::run`，并且为每个 kernel 在临时目录里 dump `.npy`，按 `runtime-session` 的 manifest 契约组织 I/O。

原因：早期版本曾有 multi-block kernel "时灵时不灵"的非确定性 bug。根因是 tiling 参数在 `unordered_map<string,int64_t>` 里按迭代顺序打包，libstdc++ 每个进程随机化字符串 hash → 字节序不固定。修法是按参数名字母排序（`HostLaunchHelper.cpp` 的 no-schema fallback 路径），并且整体改走 session 路径以匹配 `runtime-session` 的 deterministic 行为。

### 5.2 aclnn host-mode 强制

`harness.cpp:250` 默认强制 `setHostMode(true)`：

```cpp
if (std::getenv("NETWORK_RUNNER_FORCE_HOST_MODE") ||
    !std::getenv("NETWORK_RUNNER_REAL_ACLNN")) {
  mlir::runtime::aclnn::setHostMode(true);
}
```

理由：camodel sim 没有真 device。如果让 aclnn 走真 API，它会成功调用并把结果写到所谓"device buffer"，但下游 AscendC kernel 拿到的是没法解引用的指针，下一次 `dumpTensorIfEnabled` 一 memcpy 就崩。所以在 sim 模式下 aclnn 一律走 `lib/Runtime/AclnnOps.cpp` 的 CPU 参考实现。

切真硬件用 `NETWORK_RUNNER_REAL_ACLNN=1`。

### 5.3 manifest schema fallback

理想路径：每个 kernel 的 `tiling_space.json` 装在 `<kernelBinariesDir>/<kernelName>/` 下，HostLaunchHelper 读它拿到 schema-declared 顺序，按顺序打包 tiling 参数。

现实：runner 的 artifact 布局只装了 `.bin` + `manifest.txt`，`_space.json` 留在 `$WORK/<name>_space.json`。HostLaunchHelper 找不到 schema → 走 no-schema fallback → 按字母序排参数名打包。

本 case 参数名 `XBLOCK` / `XBLOCK_SUB`，字母序恰好等于 schema 序，所以稳。重命名成 `XBLOCK_A` / `XBLOCK_B`（与 schema 序相反）就能复现 silent-wrong。这是已记 caveat，长期修法是把 `tiling_space.json` 也安装到 artifact 旁边，或者让 helper 回 `<workdir>` 找。

### 5.4 block_dim_expr 为空

outline 出来的 elementwise kernel 在 `afir-translate -mlir-to-cann` 时不写 `block_dim_expr`。fallback `_block_dim=1` 意味着 sim 上只跑一个 block。tile 之外的元素会保留 init 值（一般是 0）。本 case shape 4×4 ≤ 最小 XBLOCK 16，正好不暴露。任何 e2e demo 想要大 shape 都得先解决这个 latent issue。

---

## 6. 当前局限与下一步

已记 issues（来自 `project_network_runner_v1.md` + `docs/superpowers/notes/2026-05-14-reduce-codegen-status.md`）：

- **block_dim_expr 静态空**（§5.4）：限制 e2e demo 只能用小 shape。
- **同进程第二次 AscendC launch 在 aclnn host-mode 间断后挂 camodel >20min**：mixed-attn-e2e 当初规划的第二个 AscendC kernel 因此丢掉，目前 v1 mixed demo 是 1 个 AscendC + 1 个 aclnn。原因不明，疑似 camodel state interference。本 two-elewise 例子两个连续 AscendC launch 没有遇到，因为中间没有 aclnn。
- **tiling_space.json 安装位置**（§5.3）：no-schema fallback 字母序碰巧在当前 kernel 上 OK，重命名能复现 silent-wrong。
- **autotuner 不支持多输出 kernel**：`--expected` 是单数，多输出 v1 不收。
- **reduce 路径还没好**：见 `docs/superpowers/notes/2026-05-14-reduce-codegen-status.md`。

真硬件路径：`NETWORK_RUNNER_REAL_ACLNN=1` 可以让 aclnn 走真 API + 让 generated host 的 aclInit fallback 不强制 host-mode；目前没在卡上验过，需要硬件回归一遍。

---

## 7. 附录：跑一遍

### 环境（`run.sh` 头部）

```bash
source /home/gser/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH="/home/gser/Ascend/cann-9.0.0/x86_64-linux/simulator/Ascend910B1/lib:\
/home/gser/Ascend/cann-9.0.0/x86_64-linux/lib64:\
/home/gser/Ascend/cann-9.0.0/x86_64-linux/devlib/linux/x86_64:$LD_LIBRARY_PATH"
export PATH="$REPO/build/bin:$PATH"
```

`examples/env.sh` 里的 `resolve_ascend_env.sh` 会跳过 sim 库路径，必须手动 export。

### 一行跑

```bash
bash examples/two-elewise-e2e/run.sh
```

期望尾部输出：

```
phase 1 OK → .../groups/network.json
phase 2 OK → .../artifacts
phase 3 OK → .../intermediates_default
phase 4 OK → .../tilings_best.json
[harness] wrote output[0] → .../outputs/out0.npy
[harness] wrote output[1] → .../outputs/out1.npy
network.output[0]: max_diff=0  PASS
network.output[1]: max_diff=0  PASS
```

### pytest

```bash
pytest python/tests/test_two_elewise_e2e.py -v
```
