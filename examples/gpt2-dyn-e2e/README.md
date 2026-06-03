# gpt2-dyn-e2e — dynamic-sequence GPT-2 end-to-end

A small from-scratch GPT (2 transformer blocks, n_embd=64, 4 heads) exported
with a **dynamic sequence dim** (`tensor<1x?x64xf32>`). The causal mask is built
**in-graph** via `torch.triu`. Exercises the full dynamic-seq stack:

- attention → `@__aclnn_flash_attention` (dynamic Q/K/V + a value-less
  `[1,1,S,S]` causal-mask *sentinel*; the FA regenerates the triangular mask
  from the runtime seq length, so the in-graph `triu` DCEs)
- LayerNorm → `@__aclnn_layer_norm` (the dynamic `dim==1?0:idx` broadcast is
  raised to a clean affine broadcast by `--lower-broadcast-extract`)
- MLP / erf-GELU, residual adds, qkv `split` → all dynamic-shape AscendC/aclnn.

The **same compiled artifact** runs any sequence length: `gen.py` defaults to
`seq=48` (a value never "compiled for") to prove the extent is resolved at
runtime.

## Run

```bash
# sim (camodel) — needs torch for the model build + golden
bash examples/gpt2-dyn-e2e/run.sh
# different sequence length, same path
SEQ=96 bash examples/gpt2-dyn-e2e/run.sh
# real NPU (in the 910C container)
BACKEND=npu bash examples/gpt2-dyn-e2e/run.sh
```

Expected: `network.output[0] max_diff≈7e-7 PASS`.

See `docs/superpowers/notes/2026-06-03-dynamic-gpt2-attention-handoff.md` for the
walls cracked to get here (dynamic attention, causal-mask sentinel, dynamic
extract_slice, `--remove-cf-assert`, `--lower-broadcast-extract` for dynamic
LayerNorm, cross-arg `memref.dim` resolution, shape_equalities `-1` skip).
