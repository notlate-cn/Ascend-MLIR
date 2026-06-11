# kg1-hang Debug Session — 2026-05-14

## 一句话
"AscendC → aclnn(host-mode) → AscendC" 第二次 AscendC launch 在 camodel 内 hang >20min。

## 已知背景（不重复 handoff 文档）
- branch: `llm-net` @ 190c814
- 当前 examples/mixed-attn-e2e 是 v1 demo（只 1 AscendC + 1 aclnn），刻意砍掉 kg1 来绕开 hang。
- handoff 链：see `docs/project-status-2026-05-14.md` §7、memory `project_network_runner_v1`。

## 本次目标
重建最小 kg1 repro → 通过切分实验定位是哪个环节让第二次 AscendC launch hang。

## 关键观察（看代码得来，未实验验证）

1. **Host-mode 下 `run_FlashAttentionScore` 完全不碰 acl runtime**
   `lib/Runtime/AclnnOps.cpp:227-230` 直接走 `sdpa_cpu`，纯 `operator new` + 算术。
   → 所谓 "aclnn host-mode interlude" 在内存层面只是普通 CPU 计算。

2. **`network()` 入口仍会调一次 `aclInit(nullptr)`**
   即使 harness 预设了 host-mode，`network_host.cpp:55` 还是会 try `aclInit`。
   → 这是 host-mode 路径里唯一真正接触 acl runtime 的地方。
   → 可通过 `ASCEND_MLIR_FORCE_HOST_MODE=1` 跳过（走 force 分支）。

3. **`HostLaunchHelper` 用 singleton session**
   `HelperState::session` 是 process-wide 持久的 `ExecutionSession`。两次 AscendC launch 共享同一个 session → SimBackend 也是同一份。

4. **`examples/two-elewise-e2e` 跑通了 2× AscendC 背靠背**（commit 99a5128）
   → 单纯 "同一 process 两次 AscendC launch" 不应该挂。

由 2+3 推测：toxin 大概率不是 host-mode aclnn op 本身（纯 CPU），更可能是 `aclInit()` 或某个 ACL/camodel 全局状态被某次接触污染。

## 切分阶梯（按用户指示先 H1 → H3）

| 编号 | 中段内容 | 是否调 aclInit | 预测 |
|------|---------|----------------|------|
| H1   | 空（memcpy kg0_out → kg1_in） | 是（network() 默认行为） | 不 hang（如果挂说明跟 aclnn 无关） |
| H3   | 完整 host-mode sdpa_cpu       | 否（FORCE_HOST_MODE=1）  | ? |
| H2   | aclInit/aclFinalize，不跑 op  | 是                       | 待 H1/H3 结论后决定 |
| H4   | 原现状（aclInit + host-mode sdpa）| 是                  | hang（已知） |

## Repro 构造

新目录 `examples/kg1-hang-repro/`（保留 mixed-attn-e2e 不破坏）：
- `network.mlir`: kg0(static 1×1×2×8) → cast → aclnn FA(dynamic) → kg1(dynamic add-1.0) → cast back
- `kernel_group0.mlir`: 复用 mixed-attn-e2e（q*scale+bias）
- `kernel_group1.mlir`: dynamic 4D fp16，`x + 1.0`（单输入 + init）
- `gen_inputs.py`: 不算 expected.npy，或宽松 atol；信号是"是否 hang"
- `run.sh`: 包 `timeout 90s` 拿明确 hang 信号

## 实验日志

### Step 0: 构造 baseline 3-segment 网络（H4）
- 新建 `examples/kg1-hang-repro/`（network.mlir + kg0/kg1.mlir + gen_inputs.py + run.sh）。
- kg1: dynamic `tensor<?x?x?x?xf16>`，elementwise `out = x + bias`，1×1×2×8 fp16，与 v1 demo 同 shape。
- `run.sh h4` 跑完整 5-phase pipeline，包 `timeout 180s` 检测 hang。

### 结果：**没有 hang。**
- Phase 3（生成的 binary 在同一 process 内 kg0 → host-mode FA → kg1）在 <2s 内完成，retcode 0。
- 但 kg1 输出**全零**，phase 4 autotune 因数值校验失败而终止。
- Autotuner 单独跑 kg1.cpp（不经 HostLaunchHelper / 不经 FA 上下文），25 种 tiling **每一种都算出正确结果** (≈1.05)，但跟"expected"（=phase 3 dump 的零）对不上。
- → kg1 binary 没问题；run path 也不 hang；唯一问题是 phase 3 那次 kg1 launch 写了零。

### Root cause
`network_runner.phase3_default_build_and_dump` 给 default tilings 选**最大** XBLOCK 候选（`python/network_runner.py:241-242`）。
- kg0 space 只列了 `XBLOCK=[16]`（静态 shape，`block_dim_expr=""`） → 默认 16 → 正常工作。
- kg1 space 列 `XBLOCK=[16,32,64,128,256]`（动态 shape） → 默认 **256**。
- Tensor 总元素 16；XBLOCK=256 → `block_dim = ceil(16/256) = 1`。这种 "tile 远大于数据" 的情况下 kernel **没写 output**（caller buffer 维持 init=零）。

验证：手改 `tilings_default.json` kg1 XBLOCK=16 → 重跑同一 binary（无重 link） → 输出与 `expected.npy` 完全一致。
```
out_xb16: [0.9292 1.1025 1.0781 1.0566 0.9805 0.9780 1.1123 1.1729]
expected: [0.9292 1.1025 1.0781 1.0566 0.9805 0.9780 1.1123 1.1729]
```

### 关于"kg1-hang"传闻
当前 llm-net@190c814 + 1×1×2×8 dynamic kg1 下不存在 hang。可能原因：
1. 后续 commit（28c8ea6 vector-plan-isolate-kernel-outputs、af90ff0 host-launch determinism）已修了 hang。
2. 当时 hang 的具体配置（shape / static-vs-dynamic / tiling 选择）不同；本次 repro 没踩中。
3. 当时"hang"的判断可能是误诊——比如 phase 4 autotuner 在 25 种 tiling × camodel-sim-per-trial 上跑很久，从外面看像"卡住 >20min"。

H1/H3 切分实验**不需要做**——没有 hang 要切分。

### Latent bug 表态

这是个独立、可修复的 bug，**不是** "kg1-hang as 2nd launch"。建议：
- **短期**：`network_runner.phase3_default_build_and_dump` 选 default XBLOCK 时考虑数据规模，至少 cap 到 `min(largest_candidate, total_elements)`。或挑**最小**候选作为 default（保守、肯定能跑），让 autotuner 找最优。
- **长期**：kernel 端 ragged-tail（XBLOCK > total）情况下应当 fallback 到正确写出，或在 host 端 reject 这种 tiling。

## 结论与待办

- ✅ 重建 3-segment kg1 repro（dynamic-shape）。
- ❌ "kg1-hang" 现象当前不复现。
- 🆕 发现真正 bug：default-tiling-picker 在小 tensor 上选过大 XBLOCK 导致 kg1 输出全零。
- 📝 待办：
  - 跟用户确认 latent bug 的 fix 方向（host 端 cap，还是 kernel 端 fallback）。
  - 更新 memory `project_network_runner_v1`：kg1-hang 当前不复现，新增"过大 XBLOCK 致零输出"。
  - 更新 `docs/project-status-2026-05-14.md` §7。
