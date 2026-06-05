# two-elewise-e2e

End-to-end demo for the **automatic group-outline path** of `network_runner.py`,
exercising both intra-chain elementwise fusion and the outline split.

## What it shows

`model.mlir` contains two independent elementwise **chains** (2 ops each) in a
single `func.func @model`:

  - `out0 = (a + b) * e`
  - `out1 = (c * d) + f`

Within each chain the two ops share an SSA dependency, so
`--auto-fuse-group-analysis` fuses them into one Vector kernel. The two chains
share no inputs and no SSA dependencies, so `--auto-fuse-group-outline` still
splits the function into two AscendC kernels (`kernel_group0`, `kernel_group1`),
each now containing two fused ops (`add,mul` / `mul,add`). This complements
`examples/mixed-attn-e2e/`, which feeds a hand-written `network.mlir` directly.

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
