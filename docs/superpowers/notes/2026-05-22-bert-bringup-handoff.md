# BERT Bring-up Handoff — phase1 全通，codegen 8/15，两堵墙

**日期**: 2026-05-22
**分支/worktree**: `bert-bringup`（基于 `dev-network` @ `5f2c754e`，worktree `.claude/worktrees/bert-bringup`）
**spec**: `docs/superpowers/specs/2026-05-22-bert-bringup-design.md`
**plan**: `docs/superpowers/plans/2026-05-22-bert-bringup.md`

## 目的

拿现成开源 BERT-base（真结构、极小配置 hidden=64/heads=1/seq=8/1层）从 torch 前端压过整条
pipeline，目标 camodel 仿真对 PyTorch max_diff。本轮按用户决定：**只做诊断 + handoff，修复留后续**。

## 已建成的可复现入口（`examples/bert-e2e/`）

- `export_bert.py` — HF `BertLayer`(tiny) → `torch.export` → `step0_linalg.mlir` + `input_0.npy` + `expected_0.npy`。
  直接喂 `hidden_states`，跳过 embedding(gather) 和 attention mask。CLI: `--hidden/--heads/--seq/--intermediate/--dtype/--seed/--outdir`。
- `env_sibling.sh` — source `examples/env.sh` 后把工具指向 sibling `encoder-robustness/build`（同 commit，含 recognize passes）。
- `run.sh` — export + `network_runner.py --input-linalg ... --max-phase N`。dev-network 的 runner 是 **sim-only（无 --backend）**。

**复现**：
```bash
cd <worktree>
bash examples/bert-e2e/run.sh 1   /tmp/bert_e2e/tiny   # phase1 OK
bash examples/bert-e2e/run.sh 3   /tmp/bert_e2e/tiny   # phase2 codegen 在 group1 失败
```

## 结论速览

| 阶段 | 结果 |
|------|------|
| 前端 torch.export → linalg | ✅ 干净。op 概览：8 batch_matmul / 10 transpose / 1 exp / 1 erf / 2 rsqrt / collapse-expand 一堆。**无 gather/scatter**（embedding-skip 生效） |
| recognize-attention/layernorm | ✅（**仅 fp32**，见 Bug-0） |
| phase1 group-analysis + outline | ✅ 34 kernel：1 FlashAttentionScore + 2 LayerNorm + 6 Matmul + 11 Transpose（**均走 aclnn**）+ 14 ascendc vector |
| phase2 AscendC codegen | ⚠️ **15 个 ascendc kernel：8 通过，7 失败**（两类，见 Wall-A/B） |
| aclnn-backend 编译（Matmul/Transpose/FA/LN） | ❓ 未验证（phase2 在 group1 提前中止，没跑到 aclnn 那批） |
| phase3+ / sim | ❌ 未到达 |

## Bug-0：fp16 触发 recognize-attention 生成非法 collapse_shape

- 现象：`--dtype fp16` 时 phase1 报
  `tensor.collapse_shape op expected collapsed type to be 'tensor<1x8x64xf16>', but got 'tensor<1x8x64xf32>'`。
- 根因：BERT 的 attention 第二个 matmul（`probs @ V`）是 **f16 输入 / f32 累加输出**，后接 `truncf`→f16。
  `recognize-attention` 重写时按 matmul 的 f32 out-type 造了个 `collapse_shape (…f16) -> (…f32)`，元素类型不一致 → 非法 IR。
- 现状：**用 fp32 绕过**（matmul 全 f32，无 truncf，collapse 类型一致）。`run.sh` 已默认 fp32。
- 修复方向：recognize-attention 在重建输出 reshape 时应取**真实操作数的元素类型**，而非 matmul DPS out 的类型；
  或在 fold 前把 matmul 的 f32-accumulate+truncf 归一。源：`lib/Conversion/LowerNonLinalgOps/RecognizeAttentionPass.cpp`。
- 注意：fp16 是项目主路径，这个 bug 后面真要修（fp32 只是 bring-up 权宜）。

## Wall-A：weight-copy 退化 kernel → PackTilingData 无 tiling_infos

- 失败 kernel：`group1,7,11,19,24,30`（共 6 个），shape `64x64 / 64x256 / 256x64`。
- 形态：每个就一条 `linalg.generic`，body 只 `linalg.yield %in`（纯拷贝）。来源 = BERT `nn.Linear` 的**转置权重 `.contiguous()` 拷贝**。
- 报错：`PackTilingData: missing schema_version=2 auto_fuse.tiling_infos entry for kernel kernel_groupN`。
  `--auto-fuse-codegen` 把纯拷贝 DCE 成 `return %arg0`（空 body，只剩 `ascendc.pipe`），没有 compute loop → 没 tiling_infos。
- 为什么不该存在：这些是**常量权重**（network.json 里 from=const）。理想情况下权重转置应被 aclnn Matmul 吸收（Matmul 带 transpose_b），
  不该 outline 成独立 vector 拷贝 kernel。encoder 的 "const/weight threading" 显然没覆盖 BERT 这种 transposed-weight 拷贝。
- 修复方向（候选）：
  1. aclnn Matmul 吸收权重 transpose/contiguous（最干净，消灭这些 kernel）；
  2. 或 outliner 把 const 拷贝走 host-side（类似 aclnn Transpose 对 const 的处理 / CoordEmitter host-gen 的 reshape-view），不 outline 成 device kernel；
  3. 或给纯拷贝 Vector kernel 在 codegen 补一条最小 copy loop + tiling_infos（兜底，最不优雅）。
- 源：`--auto-fuse-group-analysis/outline`（`lib/Conversion/AutoFuse/...`）+ `ascendc-pack-tiling-data`。

## Wall-B：erf-GELU 的 divf + erf 在 vector 路径不支持

- 失败 kernel：`group27`，body = bias-add(`addf`) + erf-GELU：
  `divf %in, 1.41421354`（x/√2）→ `erf` → `addf 1.0` → `mulf 0.5` → `mulf %in`。
- 报错（先撞 divf）：`LinalgToAscendC: unsupported op in linalg.generic body: arith.divf`。
- `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp` 支持集 = AddF/SubF/MulF/MaximumF/…；
  **`divf`、`erf`、`exp`、`sqrt`、`rsqrt` 都不支持**（exp/sqrt/rsqrt 不影响——已 fold 进 aclnn FA/LN）。
  所以 GELU 同时缺 `divf` 和 `erf` 两个 op，不止 divf。
- 修复方向（候选）：
  1. ComputeConversion 补 `divf`（div-by-const → 乘倒数；一般 div → AscendC Div）+ `erf`（需 AscendC erf intrinsic，确认 API）；
  2. 或新增 `recognize-gelu` pass 把 erf-GELU fold 成 aclnn（类比 recognize-layernorm），彻底绕开 vector erf——可能更省事；
  3. bring-up 临时可把 `BertConfig(hidden_act="relu")` 换掉 GELU（relu=maximumf 已支持），但 **Wall-A 仍然挡路**，单换激活到不了 sim。

## 给 fix session 的建议顺序

1. **Wall-A 必修**（任何配置都挡路）：优先让 aclnn Matmul 吸收权重 transpose，消灭 weight-copy kernel。
2. **Wall-B**：建议走 recognize-gelu→aclnn（与 LN/attention 一致的架构），比在 vector 补 erf intrinsic 风险低。
3. **Bug-0**：fp16 路径修 recognize-attention 的 reshape 元素类型（回到项目主 dtype）。
4. 全绿后继续 plan 的 S4（`run.sh 5`）对 max_diff，再 S5 放大。

## 隔离性

全程只在 `bert-bringup` 新增 `examples/bert-e2e/*` 与本 docs，**未改 dev-network pass 源码**；
工具复用 sibling build（只读执行）。并行 encoder-robustness 工作不受影响。
