# Simple LLM Block Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a minimal transformer block (LayerNorm → MHA → Residual → LayerNorm → FFN → Residual) in pure PyTorch, run it through the torch→NPU pipeline, and document where the pipeline breaks — identifying gaps for Cube-Vector fusion, batched matmul, softmax, and LayerNorm.

**Architecture:** Single test file using the existing `@torch_e2e_test` framework. The model is a standard transformer block with hidden_dim=16, num_heads=2, seq_len=8. We also create sub-tests that isolate each component (FFN, Attention, LayerNorm) to pinpoint exactly which ops break.

**Tech Stack:** PyTorch, torch-mlir, existing Ascend-MLIR pipeline

---

### Task 1: Create the PyTorch transformer block model

**Files:**
- Create: `examples/simple-llm-block/model.py`

- [ ] **Step 1: Create the model file**

```python
"""Minimal Transformer Block for gap analysis.

Parameters: hidden_dim=16, num_heads=2, seq_len=8, 1 layer.
Uses pure PyTorch ops — no manual lowering.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Single transformer block: LN → MHA → Residual → LN → FFN → Residual."""

    def __init__(self, hidden_dim: int = 16, num_heads: int = 2, ffn_mult: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn_up = nn.Linear(hidden_dim, hidden_dim * ffn_mult, bias=False)
        self.ffn_down = nn.Linear(hidden_dim * ffn_mult, hidden_dim, bias=False)

    def forward(self, x):
        # x: [seq_len, hidden_dim]
        h = self.ln1(x)

        # Multi-head self-attention
        q = self.w_q(h)  # [S, H]
        k = self.w_k(h)
        v = self.w_v(h)

        # Reshape to [num_heads, seq_len, head_dim]
        S = q.shape[0]
        q = q.view(S, self.num_heads, self.head_dim).permute(1, 0, 2)
        k = k.view(S, self.num_heads, self.head_dim).permute(1, 0, 2)
        v = v.view(S, self.num_heads, self.head_dim).permute(1, 0, 2)

        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [nH, S, S]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # [nH, S, hD]

        # Reshape back to [S, H]
        attn_out = attn_out.permute(1, 0, 2).contiguous().view(S, -1)
        attn_out = self.w_o(attn_out)

        # Residual
        x = x + attn_out

        # FFN
        h = self.ln2(x)
        h = self.ffn_up(h)
        h = F.gelu(h)
        h = self.ffn_down(h)

        # Residual
        x = x + h
        return x


class FFNOnly(nn.Module):
    """Isolated FFN: Linear → GELU → Linear. Tests matmul + activation CV fusion."""

    def __init__(self, hidden_dim: int = 16, ffn_mult: int = 4):
        super().__init__()
        self.up = nn.Linear(hidden_dim, hidden_dim * ffn_mult, bias=False)
        self.down = nn.Linear(hidden_dim * ffn_mult, hidden_dim, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class AttentionOnly(nn.Module):
    """Isolated self-attention (no LN, no residual).
    Tests batched matmul + softmax + reshape."""

    def __init__(self, hidden_dim: int = 16, num_heads: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        S = x.shape[0]
        q = self.w_q(x).view(S, self.num_heads, self.head_dim).permute(1, 0, 2)
        k = self.w_k(x).view(S, self.num_heads, self.head_dim).permute(1, 0, 2)
        v = self.w_v(x).view(S, self.num_heads, self.head_dim).permute(1, 0, 2)

        scale = self.head_dim ** -0.5
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)
        return out.permute(1, 0, 2).contiguous().view(S, -1)


class LayerNormOnly(nn.Module):
    """Isolated LayerNorm. Tests reduce + sub + mul + rsqrt pattern."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.ln(x)
```

- [ ] **Step 2: Verify the model runs in pure PyTorch**

Run: `cd /home/gser/code/Ascend-MLIR && python -c "
import torch; import sys; sys.path.insert(0, 'examples/simple-llm-block')
from model import TransformerBlock, FFNOnly, AttentionOnly, LayerNormOnly
x = torch.randn(8, 16, dtype=torch.float16)
for cls in [TransformerBlock, FFNOnly, AttentionOnly, LayerNormOnly]:
    m = cls().half()
    y = m(x)
    print(f'{cls.__name__}: input={x.shape} -> output={y.shape}')
"`

Expected: All 4 models print shapes without errors.

- [ ] **Step 3: Commit**

```bash
git add examples/simple-llm-block/model.py
git commit -m "feat: add minimal transformer block model for gap analysis"
```

---

### Task 2: Create torch_e2e tests for each sub-component

**Files:**
- Create: `test/torch_e2e/test_simple_llm_block.py`

The test file uses `@torch_e2e_test` exactly like existing tests. We use **static shapes** (no dynamic dims) to reduce variables — the goal is to find op-level gaps, not dynamic shape gaps.

- [ ] **Step 1: Write the test file**

```python
"""Simple LLM block gap analysis tests.

Incremental: run sub-components first to isolate which ops break.
  1. test_ffn — Linear → GELU → Linear (most likely to work)
  2. test_layernorm — Isolated LayerNorm
  3. test_attention — Q@K^T → softmax → @V with multi-head reshape
  4. test_transformer_block — Full block combining all
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples" / "simple-llm-block"))

import torch
import torch.nn.functional as F
from framework import torch_e2e_test, TensorSpec
from model import TransformerBlock, FFNOnly, AttentionOnly, LayerNormOnly


@torch_e2e_test(verify_shapes={"S": 8})
def test_ffn():
    """Linear → GELU → Linear. Tests matmul + GELU CV fusion opportunity."""
    return FFNOnly(hidden_dim=16).half(), [TensorSpec(("S", 16))]


@torch_e2e_test(verify_shapes={"S": 8})
def test_layernorm():
    """Isolated LayerNorm. Tests reduce + elementwise pattern."""
    return LayerNormOnly(hidden_dim=16).half(), [TensorSpec(("S", 16))]


@torch_e2e_test(verify_shapes={"S": 8})
def test_attention():
    """Self-attention with multi-head. Tests batched matmul + softmax."""
    return AttentionOnly(hidden_dim=16, num_heads=2).half(), [TensorSpec(("S", 16))]


@torch_e2e_test(verify_shapes={"S": 8})
def test_transformer_block():
    """Full transformer block. Expected to fail — run last."""
    return TransformerBlock(hidden_dim=16, num_heads=2).half(), [TensorSpec(("S", 16))]
```

- [ ] **Step 2: Commit**

```bash
git add test/torch_e2e/test_simple_llm_block.py
git commit -m "feat: add torch_e2e tests for LLM block gap analysis"
```

---

### Task 3: Run tests and record gap analysis

This is the core gap analysis task. Run each test, record where it breaks, save the intermediate MLIR files for inspection.

- [ ] **Step 1: Run test_ffn**

```bash
cd /home/gser/code/Ascend-MLIR
conda run -n torch-mlir python -m pytest test/torch_e2e/test_simple_llm_block.py::test_ffn -v -s 2>&1 | tee output/torch_e2e/test_ffn.log
```

Record: which stage fails, the error message, and inspect the last successful MLIR output.

- [ ] **Step 2: Run test_layernorm**

```bash
conda run -n torch-mlir python -m pytest test/torch_e2e/test_simple_llm_block.py::test_layernorm -v -s 2>&1 | tee output/torch_e2e/test_layernorm.log
```

- [ ] **Step 3: Run test_attention**

```bash
conda run -n torch-mlir python -m pytest test/torch_e2e/test_simple_llm_block.py::test_attention -v -s 2>&1 | tee output/torch_e2e/test_attention.log
```

- [ ] **Step 4: Run test_transformer_block**

```bash
conda run -n torch-mlir python -m pytest test/torch_e2e/test_simple_llm_block.py::test_transformer_block -v -s 2>&1 | tee output/torch_e2e/test_transformer_block.log
```

- [ ] **Step 5: Inspect MLIR outputs**

For each test, check the last successful MLIR stage in `output/torch_e2e/<test_name>/`:
- `step0_linalg.mlir` — What ops does torch-mlir produce? `linalg.matmul`? `linalg.batch_matmul`? `linalg.generic` with what iterator_types?
- `step1_fused.mlir` — Did fusion work? How many `linalg.generic` ops remain?
- `step2_tiled.mlir` — Did tiling work? (Expected: probably fails here for matmul/attention)

Key questions to answer per test:
1. What stage fails?
2. What is the error?
3. What does the linalg IR look like? (ops, shapes, iterator_types)
4. Are there CV fusion opportunities visible in the IR? (matmul followed by elementwise)
5. What would be needed to make it work?

- [ ] **Step 6: Write findings to README**

Create `examples/simple-llm-block/README.md` with:
- Summary table: test name | last passing stage | failure stage | error | root cause
- Per-test detailed IR analysis
- CV fusion opportunities observed
- Prioritized list of gaps to address

- [ ] **Step 7: Commit findings**

```bash
git add examples/simple-llm-block/README.md
git commit -m "docs: record LLM block gap analysis findings"
```