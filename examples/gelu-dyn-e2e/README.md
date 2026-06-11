# gelu-dyn-e2e — dynamic-shape (from-torch) end-to-end

`gelu(x + b)` on `tensor<1x?x3072xf32>` — the **seq dim is dynamic** (a
torch.export `Dim`). Exercises the from-torch dynamic-shape pipeline: the
explicit `index` dim-arg is eliminated in group-outline (derived in-kernel via
`tensor.dim`), the dynamic `tensor.empty` / `collapse`/`expand_shape` are
host-allocated at runtime, and the dynamic row-count is a `ShapeDerived` tiling
param resolved from the input npy shape at launch.

The concrete inputs use **seq=48** (≠ the GPT-2 static 64) so a pass proves the
extent is resolved at runtime, not baked in.

## Run

```bash
# sim (camodel)
bash examples/gelu-dyn-e2e/run.sh
# real NPU (in the 910C container; reuses the container's CANN)
BACKEND=npu bash examples/gelu-dyn-e2e/run.sh
# different dynamic extent
SEQ=96 bash examples/gelu-dyn-e2e/run.sh
```

Expected: `network.output[0] max_diff≈1e-6 PASS`.

See `docs/superpowers/notes/2026-06-02-dynamic-shape-from-torch-handoff.md`.
