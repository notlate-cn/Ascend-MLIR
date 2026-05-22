# BERT Bring-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把开源 BERT-base（真结构、极小配置）从 PyTorch 前端一路压过整条 pipeline，最终在 camodel 仿真上跑出数值并对 PyTorch 参考的 `max_diff`。

**Architecture:** HF `BertLayer`(tiny config) → `torch.export` → linalg MLIR（复用 `python/torch/torch2linalg`）→ `network_runner.py --input-linalg`（phase1 group-analysis/outline 切 kernel：matmul→aclnn / attention→FA / LN→aclnn / vector→AscendC）→ camodel sim → 数值验证。增量 bring-up，逐阶段 gate。

**Tech Stack:** PyTorch 2.11 + transformers 5.5.4（torch-mlir conda 环境）；afir-opt / afir-translate / aclnn-backend / runtime-session / autotuner（**复用 sibling `encoder-robustness/build` 二进制**，同 commit `5f2c754e`）；camodel 仿真（Ascend910B1）。

**约定（路径）：** 所有命令在 worktree `/home/gser/code/Ascend-MLIR/.claude/worktrees/bert-bringup` 下执行（下文记作 `$WT`）。Python 用 `conda run -n torch-mlir`。

---

## File Structure

- `examples/bert-e2e/export_bert.py`（新增）— 构造 tiny `BertLayer`、export 成 linalg、dump `input_0.npy` / `expected_0.npy` / `step0_linalg.mlir`。CLI 参数化 config。
- `examples/bert-e2e/env_sibling.sh`（新增）— source `examples/env.sh` 后，把工具路径覆盖到 sibling build（`AFIR_OPT` 等 + PATH 前缀）。
- `examples/bert-e2e/run.sh`（新增）— 串起 export + network_runner，透传 `--max-phase`。
- `examples/bert-e2e/README.md`（新增）— 怎么跑、各阶段 gate、已知限制。

不改 dev-network 现有 pass / runner 源码，除非 bring-up 暴露必须修的 bug（届时单独小改并记录）。

---

### Task 1: 工具环境（复用 sibling build）

**Files:**
- Create: `$WT/examples/bert-e2e/env_sibling.sh`

- [ ] **Step 1: 写 env_sibling.sh**

```bash
#!/usr/bin/env bash
# Source examples/env.sh for sim LD_LIBRARY_PATH, then override tool paths to the
# sibling encoder-robustness build (same dev-network commit 5f2c754e) since this
# worktree has no build/ of its own.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "${WT_ROOT}/examples/env.sh"

SIBLING_BIN="/home/gser/code/Ascend-MLIR/.claude/worktrees/encoder-robustness/build/bin"
export AFIR_OPT="${SIBLING_BIN}/afir-opt"
export AFIR_TRANSLATE="${SIBLING_BIN}/afir-translate"
export ACLNN_BACKEND="${SIBLING_BIN}/aclnn-backend"
export RUNTIME_SESSION="${SIBLING_BIN}/runtime-session"
export AUTOTUNER="${SIBLING_BIN}/autotuner"
export PATH="${SIBLING_BIN}:${PATH}"
echo "BERT-e2e tools -> ${SIBLING_BIN}"
```

- [ ] **Step 2: 验证工具可执行且是 dev-network 版（含 recognize passes）**

Run:
```bash
cd $WT && source examples/bert-e2e/env_sibling.sh && \
  "$AFIR_OPT" --help 2>&1 | grep -c "auto-fuse" && \
  "$AFIR_OPT" --help 2>&1 | grep -iE "recognize-attention|recognize-layernorm"
```
Expected: 第一行 ≥1；第二条能 grep 到 recognize-attention / recognize-layernorm（确认是 dev-network 二进制）。

- [ ] **Step 3: Commit**

```bash
cd $WT && git add examples/bert-e2e/env_sibling.sh && \
  git commit -m "feat(bert-e2e): tool env pointing at sibling dev-network build"
```

---

### Task 2: export_bert.py — tiny BERT → linalg + 参考数据

**Files:**
- Create: `$WT/examples/bert-e2e/export_bert.py`

- [ ] **Step 1: 先确认本机 transformers 5.5.4 的 BertLayer.forward 签名/返回**

Run:
```bash
cd $WT && conda run -n torch-mlir python -c "
import inspect
from transformers.models.bert.modeling_bert import BertLayer
print(inspect.signature(BertLayer.forward))
"
```
Expected: 打印签名。**记下第一个非 self 形参是否 `hidden_states`、是否返回 tuple。** 若返回 tuple，wrapper 取 `[0]`；若已返回单 tensor，wrapper 直接返回。下一步代码按返回 tuple 写（5.x 常见 `(layer_output,)`），若实测为单 tensor 则把 `[0]` 去掉。

- [ ] **Step 2: 写 export_bert.py**

```python
#!/usr/bin/env python3
"""Tiny BERT layer -> linalg MLIR + reference npy, for the network_runner pipeline.

Feeds hidden_states directly into a HuggingFace BertLayer (skips embedding lookup
and attention mask) so the matmul/attention/layernorm/vector backbone is what gets
exercised. Outputs step0_linalg.mlir, input_0.npy, expected_0.npy into --outdir.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Make torch2linalg importable (mirrors examples/torch_e2e/conftest.py).
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python" / "torch"))
from torch2linalg import torch_to_linalg  # noqa: E402

from transformers import BertConfig  # noqa: E402
from transformers.models.bert.modeling_bert import BertLayer  # noqa: E402


class BertLayerWrapper(nn.Module):
    """Single BertLayer taking hidden_states, returning the layer output tensor."""
    def __init__(self, config):
        super().__init__()
        self.layer = BertLayer(config)

    def forward(self, hidden_states):
        out = self.layer(hidden_states)
        # BertLayer returns a tuple in transformers 5.x; take the hidden states.
        return out[0] if isinstance(out, (tuple, list)) else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--intermediate", type=int, default=None,
                    help="FFN intermediate size (default 4*hidden)")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    inter = args.intermediate or 4 * args.hidden

    config = BertConfig(
        hidden_size=args.hidden,
        num_attention_heads=args.heads,
        intermediate_size=inter,
        num_hidden_layers=1,
        max_position_embeddings=max(args.seq, 16),
        vocab_size=32,                 # unused (we skip embeddings) but must be set
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
    )

    model = BertLayerWrapper(config).eval().to(dtype)
    # batch=1; hidden_states shape [B, seq, hidden]
    hidden = torch.randn(1, args.seq, args.hidden, dtype=dtype)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Reference output from PyTorch.
    with torch.no_grad():
        expected = model(hidden)
    np.save(outdir / "input_0.npy", hidden.numpy())
    np.save(outdir / "expected_0.npy", expected.numpy())

    # torch.export -> linalg (static shapes; no dynamic_shapes for bring-up).
    mlir_text = torch_to_linalg(model, [hidden], None)
    (outdir / "step0_linalg.mlir").write_text(mlir_text)

    print(f"hidden_states={tuple(hidden.shape)} expected={tuple(expected.shape)} dtype={args.dtype}")
    print(f"wrote: {outdir}/step0_linalg.mlir, input_0.npy, expected_0.npy")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行 export，验证产物存在**

Run:
```bash
cd $WT && conda run -n torch-mlir python examples/bert-e2e/export_bert.py \
  --hidden 64 --heads 1 --seq 8 --dtype fp16 --outdir /tmp/bert_e2e/tiny
ls -la /tmp/bert_e2e/tiny/
```
Expected: 打印 `hidden_states=(1, 8, 64) expected=(1, 8, 64)`；目录里有 `step0_linalg.mlir`、`input_0.npy`、`expected_0.npy`。
若 export 报错（torch.export trace 失败/不支持算子）→ 进 **systematic-debugging**：看 traceback 定位是哪个算子（softmax/gelu/erf 等），记录到 README 的"已知限制"。

- [ ] **Step 4: 验证 step0_linalg.mlir 能被 afir-opt 解析**

Run:
```bash
cd $WT && source examples/bert-e2e/env_sibling.sh && \
  "$AFIR_OPT" --linalg-fold-unit-extent-dims /tmp/bert_e2e/tiny/step0_linalg.mlir -o /dev/null && \
  echo "PARSE OK"
echo "--- op 概览 ---"
grep -oE "linalg\.[a-z_]+|tensor\.[a-z_]+|math\.[a-z_]+|arith\.[a-z_]+" \
  /tmp/bert_e2e/tiny/step0_linalg.mlir | sort | uniq -c | sort -rn | head -30
```
Expected: `PARSE OK`；op 概览里能看到 `linalg.matmul`/`batch_matmul`、softmax 相关（`math.exp` 等）、layernorm 相关、`linalg.generic`。**人工核对有无明显不支持的算子（如 gather/scatter）。**

- [ ] **Step 5: Commit**

```bash
cd $WT && git add examples/bert-e2e/export_bert.py && \
  git commit -m "feat(bert-e2e): export tiny BertLayer to linalg + reference npy"
```

---

### Task 3: S2 — group-analysis/outline 切 kernel（network_runner --max-phase 1）

**Files:**
- Create: `$WT/examples/bert-e2e/run.sh`

- [ ] **Step 1: 写 run.sh**

```bash
#!/usr/bin/env bash
# Drive: export tiny BERT -> network_runner pipeline up to --max-phase.
# Usage: examples/bert-e2e/run.sh [MAX_PHASE] [BACKEND] [OUTDIR]
set -euo pipefail
MAX_PHASE="${1:-1}"
BACKEND="${2:-sim}"
OUTDIR="${3:-/tmp/bert_e2e/tiny}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "${SCRIPT_DIR}/env_sibling.sh"

conda run -n torch-mlir python "${SCRIPT_DIR}/export_bert.py" \
  --hidden 64 --heads 1 --seq 8 --dtype fp16 --outdir "${OUTDIR}"

conda run -n torch-mlir python "${WT_ROOT}/python/network_runner.py" \
  --input-linalg "${OUTDIR}/step0_linalg.mlir" \
  --inputs "${OUTDIR}/input_0.npy" \
  --expected "${OUTDIR}/expected_0.npy" \
  --workdir "${OUTDIR}/work" \
  --backend "${BACKEND}" \
  --max-phase "${MAX_PHASE}" \
  --atol 1e-2 --rtol 1e-2
```

- [ ] **Step 2: 跑 phase 1**

Run:
```bash
cd $WT && bash examples/bert-e2e/run.sh 1 sim /tmp/bert_e2e/tiny
```
Expected: `phase 1 OK → .../network.json`。若 group-analysis/outline 崩（SIGSEGV/assert）→ systematic-debugging（参考 memory 里 GroupAnalysis fusion bug 的处理思路：非连续 group、glue-aware cycle 等），但**优先只诊断、改动交评估**。

- [ ] **Step 3: 看切出哪些 kernel**

Run:
```bash
cat /tmp/bert_e2e/tiny/work/groups/network.mlir | grep -iE "func.func|aclnn|matmul|attention|layer_norm" | head -40
ls /tmp/bert_e2e/tiny/work/groups/
```
Expected: 看到若干 outlined func；理想包含 aclnn matmul / FlashAttention / layer_norm + 若干 AscendC vector kernel。记录实际切分到 README。

- [ ] **Step 4: Commit**

```bash
cd $WT && git add examples/bert-e2e/run.sh && \
  git commit -m "feat(bert-e2e): run.sh driving network_runner; phase1 outline verified"
```

---

### Task 4: S3 — codegen 干净（--max-phase 3）

**Files:** 无新增（用 run.sh）

- [ ] **Step 1: 跑到 phase 3**

Run:
```bash
cd $WT && bash examples/bert-e2e/run.sh 3 sim /tmp/bert_e2e/tiny
```
Expected: `phase 3 OK → ...`。所有 kernel 的 aclnn + AscendC codegen 编译通过。

- [ ] **Step 2: 失败则分诊**

若某 kernel codegen 报错：定位是 attention（FA 折叠没匹配 BERT 的 softmax 分解）、layernorm、gelu(erf)、还是 vector tile。记录到 README "已知限制"，并判断是否需小改 dev-network pass（需用户确认再动）。常见嫌疑：
- softmax 没被 recognize-attention 折叠 → attention 退化成裸 bmm+exp，看 vector 路径能否 codegen。
- gelu(erf) → `math.erf` 是否被 LinalgToAscendC 支持。

- [ ] **Step 3: Commit（记录阶段结论）**

```bash
cd $WT && git add examples/bert-e2e/README.md && \
  git commit -m "docs(bert-e2e): phase3 codegen status + known limitations"
```
（README 在 Task 6 建；若此处先记结论，先建最小 README 再 commit。）

---

### Task 5: S4 — camodel 仿真数值验证（--max-phase 5）

**Files:** 无新增

- [ ] **Step 1: 跑全 5 phase（sim）**

Run:
```bash
cd $WT && bash examples/bert-e2e/run.sh 5 sim /tmp/bert_e2e/tiny 2>&1 | tee /tmp/bert_e2e/run_phase5.log
```
Expected: 跑到 phase 5，打印 verify 结果（pass / max_diff）。

- [ ] **Step 2: 检查数值**

Run:
```bash
grep -iE "max_diff|PASS|FAIL|verify" /tmp/bert_e2e/run_phase5.log | tail -20
```
Expected（最小成功）：`max_diff` 在 atol/rtol（1e-2）内 → PASS。
若 fp16 精度过不去但路径通：重跑 `export_bert.py --dtype fp32` 复验（需 run.sh 临时改 dtype 或加参数）；若特定 kernel 数值错（参考 memory：encoder transpose accuracy 是已知雷区）→ 进 systematic-debugging，定位到具体 outlined kernel，记录。

- [ ] **Step 3: Commit（记录 S4 结论）**

```bash
cd $WT && git add examples/bert-e2e/README.md && \
  git commit -m "docs(bert-e2e): phase5 sim numerical result (max_diff)"
```

---

### Task 6: README + 放大（S5）

**Files:**
- Create: `$WT/examples/bert-e2e/README.md`

- [ ] **Step 1: 写 README**

```markdown
# BERT E2E (bring-up)

Tiny HuggingFace BertLayer → linalg → network_runner（aclnn matmul/attention/layernorm + AscendC vector）→ camodel sim → 对 PyTorch max_diff。

## 跑

    cd <worktree-root>
    bash examples/bert-e2e/run.sh <max_phase> <backend> <outdir>
    # 例：bash examples/bert-e2e/run.sh 5 sim /tmp/bert_e2e/tiny

工具复用 sibling `encoder-robustness/build`（见 env_sibling.sh）。

## Bring-up 阶段
- phase1：group-analysis/outline 切 kernel
- phase3：codegen 干净
- phase5：sim 数值验证（atol/rtol 1e-2）

## 配置
export_bert.py：`--hidden --heads --seq --intermediate --dtype --outdir`
起点 hidden=64/heads=1/seq=8/1 层 fp16。

## 已知限制
（bring-up 过程中填：跳过 embedding + mask；softmax/gelu/transpose 等遇到的问题）
```

- [ ] **Step 2: 放大复跑（逐项，遇阻即停记录）**

依次尝试，每项跑 `run.sh 5 sim`：
```bash
# 2 头
conda run -n torch-mlir python examples/bert-e2e/export_bert.py --hidden 128 --heads 2 --seq 8 --dtype fp16 --outdir /tmp/bert_e2e/h2
# seq 32
conda run -n torch-mlir python examples/bert-e2e/export_bert.py --hidden 64 --heads 1 --seq 32 --dtype fp16 --outdir /tmp/bert_e2e/s32
```
（对每个 outdir 再跑对应 network_runner；或扩展 run.sh 接受 config 参数。）
Expected: 记录每个配置到哪一 phase、max_diff 多少，写进 README。

- [ ] **Step 3: Commit**

```bash
cd $WT && git add examples/bert-e2e/README.md && \
  git commit -m "docs(bert-e2e): README + scale-up results"
```

---

## Self-Review

- **Spec coverage**：方案 A（HF BertLayer tiny、跳 embedding/mask）→ Task 2；network_runner --input-linalg 驱动 → Task 3；bring-up 阶段 S1–S5 → Task 2/3/4/5/6；构建复用 sibling → Task 1；dtype fp16 默认+fp32 回退 → Task 2 参数 + Task 5 step2；softmax/gelu/transpose 风险 → Task 4/5 分诊步骤。覆盖完整。
- **Placeholder**：README "已知限制" 是 bring-up 运行时填的实测结论（探索性任务的预期产物），非计划占位；其余步骤均有具体命令/代码。
- **一致性**：`export_bert.py` 的 CLI 参数（--hidden/--heads/--seq/--intermediate/--dtype/--outdir）在 run.sh、Task 6 调用处一致；产物文件名 `step0_linalg.mlir`/`input_0.npy`/`expected_0.npy` 与 network_runner `--input-linalg/--inputs/--expected` 对应一致；工具 env 变量名与 network_runner.py 读取的 `AFIR_OPT/AFIR_TRANSLATE/ACLNN_BACKEND/RUNTIME_SESSION/AUTOTUNER` 一致。
