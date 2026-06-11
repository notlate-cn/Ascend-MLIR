# Debug 系统 session 总结（2026-05-27 → 2026-05-29）

**Branch:** `develop` @ `1489dcb6`（已 `git push`）
**作用域:** 这一 session 推进了 `docs/auto-fuse/debug.md` §7.2–7.7 的工程落地，并在过程中接入其他 session 并行交付的 logger / conv-aclnn / ResNet18 / GPT-2 work。

---

## 1. 本 session 直接交付的 commit（按时间顺序）

```
8993a9b3  feat(network-runner): write manifest.json + optional stages/ symlinks (P0-1)
2c24e8cb  feat(debug): ascend_diff.py — unified L0 final-output diff tool (P0-2)
7ebc872e  feat(group-outline): emit network.provenance.json sidecar (P1-3)
fa94ea19  feat(debug): kernel DAG viewer — HTML+JS, consumes provenance (P1-4)
ed89202a  feat(runtime): per-kernel wall-clock timing → <profile-dir>/<kernel>.timing.json (P1-5)
2109805d  feat(debug): ascend_diff.py 'locate' subcommand — sim-as-reference first-bad-kernel (P2)
81f5bf5c  fix(debug): inline dag.json into dag.html for file:// direct-open (P1-4 polish)
a19a40e3  test(op-role-classifier): lit coverage for fused_ops_summary + op_role (P1-3 protection)
55ed59a2  fix(logger): preserve cross-run history + scoped phase context + PID-suffixed run-id
1489dcb6  docs(walkthrough): refresh for develop@55ed59a2 — Conv/BN aclnn, logger, verified networks
```

**10 commits 共 ~1300 LOC**（debug 工具 + 测试 + 文档）。

## 2. 其他 session 并行交付（本 session 期间，与本 session 工作互相不冲突）

```
683de3e3  feat(logger): unified python debug logger
bef0c3d5  refactor(network_runner): print() → logger
ec437625  refactor(runner_utils): route build_host link step through run_subprocess
78f01680  refactor(tools): diagnostic prints in ascend_diff / gen_dag_json → logger
9cc53ab7  fix(auto-fuse): unblock ResNet-18 phase-1 tensor.pad lowering
b80b68b4  feat(auto-fuse): route Conv2D groups through aclnn fallback
f755480b  feat(auto-fuse): BatchNorm + Pool2D aclnn fallback
901d3216  fix(resnet18-e2e): dump + pass BatchNorm buffer inputs (full e2e PASS)
b7f222cc  feat(gpt2-e2e): tiny GPT-2 (12-layer) bring-up
```

加 4 个 ResNet18 scaffolding / handoff doc commit。

## 3. 关键 mid-stream 调整（plan 改动）

| 节点 | 原 plan | 实际 | 触发原因 |
|---|---|---|---|
| §7.5 provenance 落点 | 扩 `network.json` 加 source_ops 字段 | 独立 `network.provenance.json` + 双 writer 同 emitter | 用户 push back：违反 §7.2"provenance 不能进编译契约" |
| OpRoleMap 数据源 | 复用既有的 `OpRoleMap` 抽象 | 仓库不存在该抽象，自写 `classifyOpRole` ~50 LOC | grep 0 命中 |
| §7.6 L1 CPU reference | torch hook ref（D1 自动 mapping） | sim-as-reference + locate 子命令 | torch-mlir 导出不保 FX loc，自动 mapping 无锚点；改 sim-as-reference 也契合 MEMORY `[BERT bring-up]` 实际工作流 |
| pad wall（ResNet18 phase 1） | option 1 vs 2 vs 3（handoff `d9d9b188` 三选项） | option 1 + Conv→aclnn 双管齐下（其他 session 实施） | 用户提示"conv 我们只能调 aclnn"——pad 单独修不够 |

## 4. 当前系统能力快照

**三层架构**（详情见 `docs/network-runner-e2e-walkthrough.zh.md`）：

1. **流程追踪 + 日志**（其他 session 加，本 session §9 加固）
   - `runner_utils/logger.py` 统一 logger + contextvars
   - `runner_utils/run_subprocess.py` 子进程 stderr tee
   - 跨 run 不丢日志 + phase 标签按 scope 复位 + PID 区分 run

2. **编译期 artifact**
   - `<work>/manifest.json` + `stages/` symlinks
   - `<work>/groups/network.json` + **`network.provenance.json`**（本 session）
   - `<work>/intermediates_default/` + **`profiles/`**（本 session）

3. **诊断工具**
   - `ascend_diff` L0 final-output + `locate` first-bad-kernel
   - `ascend_kernel_dag_viz/` HTML+JS DAG viewer（self-contained，file:// 直开）

**端到端已通过的网络**：encoder（910C，max_diff=7.15e-7）、BERT-tiny（910C，1.4e-5）、ResNet-18（sim）、GPT-2 small phase 1-3（sim）+ 7 个小 example。

## 5. 真实做过的事件时间线（不全，挑代表性）

- **5/27 上午**：plan 阶段，多轮 iteration `/home/gser/.claude/plans/docs-auto-fuse-debug-md-drifting-wadler.md`（7 个决策）
- **5/27 下午**：6 个核心 commit 一气呵成（P0-1, P0-2, P1-3, P1-4, P1-5, P2）
- **5/27 晚**：BERT-tiny 26-kernel 验证 + viz 内联修复
- **5/28**：其他 session 加 logger + ResNet18 + Conv-aclnn
- **5/29 上午**：用户跑 GPT-2 phase 1-3 → 247-kernel 拓扑分析 → 识别 13 退化 add + 12 transpose 链
- **5/29 下午**：补 lit 测 + 检视 logger 代码 + 修 3 sharp edge

## 6. 已知 deferred items（按优先级）

### High value

| 项 | 价值 | 已有 plan? |
|---|---|---|
| **CV-fusion MVP**（matmul + epilogue 进 cube kernel）| BERT 3 + GPT-2 13 退化 add 全消，aclnn-matmul 后的 bias 都能融 | MEMORY `[AF CV fusion port plan]`，~750 LOC，stub 在 `GroupAnalysisPass.cpp:147-150` |
| **Transpose-eliminate**（perm 传 consumer）| GPT-2 12 个 transpose 链可省，复杂网络更多 | MEMORY `[GroupAnalysis fusion bugs]` future 段，AF doc `transpose全流程说明.md` 有原型 |

### Medium

| 项 | 价值 |
|---|---|
| `aclrtRecordEvent` 替换 wall_us | 真 NPU perf 可信度（当前 wall_us 含 dispatch overhead） |
| auto sim-vs-NPU 双跑封装（`network_runner --ref-from sim`）| 把 4.3 工作流一键化 |
| GPT-2 phase 4-5 跑完 | 验证 246-kernel autotune 不挂 |

### Low / 暂缓

- 四层 report（Kernelize/Schedule/Realize/Translate decision 结构化）
- L3 anchor 重编译
- ascend_diff pytest（lit 已覆盖核心）
- `_state` 全局 dict 加锁（当前不并发）

## 7. 待补的 sharp edges（commit message 已写明 deferred）

- tee stderr 排序问题（`logger.py` 审计时怀疑，复现失败，可能不存在）
- `readline()` 对无 `\n` 输出不友好（当前调用方都按行，未踩）
- SIGINT 不传播给 subprocess（不痛）

## 8. 这一 session 学到的（给下个 session）

1. **MEMORY 是宝**：`[Session role: real-NPU runner]` / `[feedback_discuss_before_editing]` / `[externals_dir_vs_symlink]` 这几条避了多次坑——session 开始时跑一遍 MEMORY 是值的。
2. **doc plan 不等于 code 现实**：debug.md §7.5 假设 `OpRoleMap` 存在，仓库其实没有；§7.9.1 一堆假设抽象都是 plan-only。**别盲从 plan，先 grep 确认基础设施在不在**。
3. **plan 文档对齐胜过 code 完美**：本 session 7 个 plan 决策（不扩 network.json、不立 ascend-debug 二进制、HTML+JS DAG、CPU ref 不覆盖 aclnn、stage_input 锚 recognize-aclnn-后、torch_hint 留 null、name vs role 分层）每条都改变了实现走向；plan 阶段 1 小时换来 commit 阶段半天，值。
4. **Mid-stream 反转是好事**：3 次反转（provenance 分离、OpRoleMap 不存在、torch hook 死路）每次都让设计更对。Plan 不是契约，是初始假设。
5. **真机验证不可替代**：BERT-tiny 26 kernel 验证之后才发现 `fused_ops_summary` 对 chain case 真的可读；GPT-2 247 kernel 才显出残差+bias `add+add` 模式占多数。**不跑真实网络就不知道工具好不好用**。

## 9. 推荐下个 session 选项

| 选 | 适合 |
|---|---|
| **CV-fusion MVP 实施** | 一整个 session，~750 LOC，能 kill 大量退化 add，是工具链覆盖出来的最大 ROI 工作 |
| **Transpose-eliminate 实施** | 0.5-1 session，~300-500 LOC，effect 可量化（GPT-2 -12 dispatch） |
| **`aclrtRecordEvent` per-kernel 接入** | 半 session，~50 LOC NPU backend 改 + 测试 |
| **跑 GPT-2 phase 4-5 看 autotune** | 半小时（autotune 慢），纯黑盒测，能发现新 wall |
| **lit 补 rule 1/2/3** + ascend_diff pytest | 1 hr，提升保护网；非紧急 |

## 10. 关键文件 / 路径速查

```
# Plan 文档
/home/gser/.claude/plans/docs-auto-fuse-debug-md-drifting-wadler.md

# 工具
python/network_runner.py                   # 主驱动 + --debug-out + --profile-dir
python/tools/ascend_diff.py                # L0 + locate
python/tools/ascend_kernel_dag_viz/        # DAG viewer
python/runner_utils/logger.py              # logger
python/runner_utils/run_subprocess.py      # subprocess tee

# Compiler 侧
lib/Conversion/AutoFuse/GroupOutline/NetworkJsonEmitter.cpp     # network.json + provenance
lib/Conversion/AutoFuse/GroupOutline/OpRoleClassifier.{h,cpp}   # op_role 分类
lib/Conversion/AutoFuse/GroupOutline/GroupOutlinePass.cpp       # outline 编排
lib/Runtime/Execution/HostLaunchHelper.cpp                      # dump + timing

# 测试
test/Conversion/Group/op-role-classifier/*.mlir                 # 4 lit tests

# 文档
docs/auto-fuse/debug.md                                         # 原始 plan
docs/network-runner-e2e-walkthrough.zh.md                       # 用户走查
docs/network-runner-walkthrough.zh.md                           # v1 内部细节
docs/superpowers/notes/2026-05-29-debug-system-session-summary.md  # 本文件
```
