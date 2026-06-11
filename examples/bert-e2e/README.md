# BERT E2E (bring-up)

现成开源 HuggingFace `BertLayer`（极小配置）→ linalg → `network_runner`
（aclnn matmul/attention/layernorm/transpose + AscendC vector）→ camodel sim → 对 PyTorch max_diff。

直接喂 `hidden_states`，**跳过 embedding 查表和 attention mask**，让 matmul/attention/layernorm/vector
主干成为被压的对象。

## 跑

```bash
cd <worktree-root>
bash examples/bert-e2e/run.sh <max_phase> [outdir]
# 例：
bash examples/bert-e2e/run.sh 1 /tmp/bert_e2e/tiny   # phase1 group-outline
bash examples/bert-e2e/run.sh 3 /tmp/bert_e2e/tiny   # 到 phase2 codegen
```

工具复用 sibling `encoder-robustness/build`（见 `env_sibling.sh`，同 dev-network commit）。
本 worktree 没有自己的 `build/`。dev-network 的 `network_runner` 是 **sim-only**（无 `--backend`）。

单独生成前端产物：
```bash
conda run -n torch-mlir python examples/bert-e2e/export_bert.py \
  --hidden 64 --heads 1 --seq 8 --dtype fp32 --outdir /tmp/bert_e2e/tiny
```
参数：`--hidden --heads --seq --intermediate --dtype{fp16,fp32} --seed --outdir`。

## 当前状态（2026-05-22）

- **phase1 group-outline：✅** 34 kernel（1 FlashAttentionScore + 2 LayerNorm + 6 Matmul + 11 Transpose 走 aclnn，14 AscendC vector）。
- **phase2 AscendC codegen：8/15 vector kernel 通过**，2 类失败挡住后续（见下）。
- phase3+/sim 未到达。

详细诊断与修复方向见 `docs/superpowers/notes/2026-05-22-bert-bringup-handoff.md`。

## 已知限制

- **必须用 fp32**：fp16 会让 `recognize-attention` 在 f16输入/f32累加 的 matmul 上生成非法
  `collapse_shape (f16)->(f32)`。fp32 绕过（`run.sh` 已默认 fp32）。
- **Wall-A weight-copy**：BERT 转置权重 `.contiguous()` 拷贝被 outline 成无 compute 的 Vector kernel，
  `PackTilingData` 无 `tiling_infos` → 失败（group1/7/11/19/24/30）。
- **Wall-B GELU**：erf-GELU 的 `arith.divf` + `math.erf` 在 `LinalgToAscendC` 不支持（group27）。
- 起点配置 hidden=64/heads=1/seq=8/1层；放大（多头/长 seq/多层）待两堵墙修好后再做。
