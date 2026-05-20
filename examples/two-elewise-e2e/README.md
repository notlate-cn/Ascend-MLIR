# two-elewise-e2e

End-to-end demo for the **automatic group-outline path** of `network_runner.py`.

## What it shows

`model.mlir` contains two completely independent `linalg.generic` ops in a
single `func.func @model`:

  - `out0 = a + b`
  - `out1 = c * d`

Because the two ops share no inputs and no SSA dependencies, the
`--auto-fuse-group-analysis` + `--auto-fuse-group-outline` passes split
the function into two AscendC kernels (`kernel_group0`, `kernel_group1`).
This complements `examples/mixed-attn-e2e/`, which feeds a hand-written
`network.mlir` directly.

## Run

```bash
bash examples/two-elewise-e2e/run.sh
```

Expected tail of output:

```
network.output[0]: max_diff=...  PASS
network.output[1]: max_diff=...  PASS
```

Phase 1 produces `$WORK/groups/network.json` with two `ascendc` kernels;
phases 2-5 codegen, compile, autotune, and verify both outputs against
`expected0.npy` / `expected1.npy` with `atol=rtol=1e-2`.
