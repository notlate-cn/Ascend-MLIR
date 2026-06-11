# Simple LLM Block Demo Design

## Goal

Create a minimal transformer block in PyTorch native code, run it through the existing torch→NPU pipeline, and identify where the pipeline breaks. This is a gap analysis exercise, not an optimization effort.

## Parameters

| Parameter | Value |
|-----------|-------|
| hidden_dim | 16 |
| num_heads | 2 |
| head_dim | 8 (= hidden_dim / num_heads) |
| seq_len | 8 |
| layers | 1 |
| dtype | float16 |

## Model Structure

```
Input x: [seq_len=8, hidden_dim=16]
  │
  ├─ ln1 = LayerNorm(x)
  │
  ├─ Multi-Head Self-Attention:
  │    Q = ln1 @ W_q    # [8, 16] @ [16, 16] → [8, 16]
  │    K = ln1 @ W_k
  │    V = ln1 @ W_v
  │    reshape Q/K/V → [num_heads, seq_len, head_dim] = [2, 8, 8]
  │    attn = softmax(Q @ K^T / sqrt(8)) @ V   # per head
  │    reshape → [8, 16]
  │    out = attn @ W_o  # output projection
  │
  ├─ residual1 = x + out
  │
  ├─ ln2 = LayerNorm(residual1)
  │
  ├─ FFN:
  │    h = ln2 @ W_1     # [8, 16] @ [16, 64] → [8, 64] (4x expansion)
  │    h = GELU(h)
  │    h = h @ W_2       # [8, 64] @ [64, 16] → [8, 16]
  │
  └─ residual2 = residual1 + h
      │
      Output: [8, 16]
```

## PyTorch Implementation Approach

Use standard `torch.nn` modules with native PyTorch ops:
- `torch.nn.LayerNorm` for normalization
- `torch.nn.Linear` (without bias, to simplify) for Q/K/V/O projections and FFN
- `torch.nn.functional.scaled_dot_product_attention` or manual `softmax(Q@K^T/sqrt(d))@V`
- `torch.nn.functional.gelu` for activation
- Standard tensor reshape/transpose for multi-head splitting

No custom ops, no manual lowering. Let torch-mlir handle everything.

## Key Ops Expected in Lowered IR

| PyTorch Op | Expected Linalg Op | Currently Supported? |
|------------|-------------------|---------------------|
| LayerNorm | reduce_sum + sub + mul + rsqrt | Partial (primitives yes, pattern unknown) |
| Linear (matmul) | linalg.matmul | Yes (matmul-add-relu-sum example) |
| Reshape/Transpose | tensor.expand_shape / linalg.transpose | Likely |
| Q @ K^T (batched matmul) | linalg.batch_matmul | Unknown |
| softmax | exp + reduce_max + sub + exp + reduce_sum + div | Partial (primitives yes, pattern unknown) |
| GELU | tanh-based or erf-based approximation | Yes (gelu op exists) |
| Residual add | linalg.add | Yes |

## Cube-Vector (CV) Fusion Analysis

Ascend 910B 有两个计算单元：Cube (AIC, 矩阵乘法) 和 Vector (AIV, elementwise)。
CV 融合的核心是：matmul 结果从 CO1 经 fixpipe 直接搬到 VECIN，避免写回 GM 再读取。

### Transformer Block 中的 CV 融合机会

| 模式 | Cube 部分 | Vector 部分 | 融合价值 |
|------|----------|------------|---------|
| FFN 第一层 | x @ W_1 | GELU(result) | 高 — 经典 matmul+activation |
| FFN 第二层 | h @ W_2 | result + residual | 高 — matmul+add |
| QKV projection | x @ W_q/k/v | reshape only | 低 — 无 vector 计算 |
| Attention score | Q @ K^T | / sqrt(d) → softmax | 高 — matmul+scale+softmax |
| Output projection | attn @ W_o | result + residual | 高 — matmul+add |

### 观察目标

1. Pipeline 当前是否将 matmul 和后续 elementwise 放在同一个 kernel 中？
2. Buffer placement 是否能识别 CO1→VECIN 的搬运模式？
3. 生成的 C++ kernel 是否使用 `Matmul.get_tensor_c()` + vector op 的 CV 融合模式？
4. 如果不能自动融合，手动实现的 gap 有多大？

参考：pyasc 的 `tutorials/05_matmul_leakyrelu/` 展示了手动 CV 融合的模式。

## Expected Gaps (Hypotheses)

1. **LayerNorm lowering**: torch-mlir may lower it to primitives, but the fused pattern may not tile well
2. **Batched matmul**: 3D+ matmul with head dimension may not be supported in current tiling
3. **Softmax**: The reduction pattern (max-subtract-exp-sum-div) across a specific axis may break fusion or tiling
4. **Tensor reshape/view**: Dynamic shape interactions with reshape operations
5. **Multi-head reshape + transpose**: The head splitting pattern `[S,H] → [nH,S,hD] → transpose` may not lower cleanly
6. **CV fusion**: Pipeline 可能无法自动识别 matmul→elementwise 的 CV 融合机会，导致中间结果写回 GM

## Location

```
examples/simple-llm-block/
├── model.py              # PyTorch model definition
├── test_simple_llm.py    # torch_e2e test using the model
└── README.md             # Description of the demo and findings
```

Also add a test in `test/torch_e2e/test_simple_llm_block.py` to integrate with the test framework.

## Success Criteria

This is a gap analysis. Success means:
1. The PyTorch model is defined and produces correct output in pure PyTorch
2. We attempt to run it through the pipeline and document exactly where it breaks
3. Each gap is identified with the specific stage and error
4. Findings are recorded for prioritizing future work

## Approach: Incremental

If the full block fails immediately, decompose into sub-tests:
1. **FFN only**: Linear → GELU → Linear (most likely to work)
2. **Attention only**: Q@K^T → softmax → @V (test batched matmul + softmax)
3. **LayerNorm only**: Isolated normalization
4. **Full block**: Combine all

This way we pinpoint exactly which component introduces each gap.