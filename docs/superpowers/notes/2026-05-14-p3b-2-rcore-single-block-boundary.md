# P3b-2 RCore single-block boundary — verification report

Date: 2026-05-14
Branch: `llm-net` (P3b-2c/d at `6e59a19`)

## TL;DR

P3b-2 (commits `e3cf5a6` / `f6bacb5` / `6e59a19`) wires full-reduce-to-scalar
end-to-end through vector-plan-codegen. **It produces correct results
exclusively in the single-block regime** (block_dim = 1). Multi-block
RCore is structurally race-prone: every block writes its partial scalar
to the same GM output slot, last writer wins. R1 (the reduce-codegen-status
report's bug) reproducer (R=16) is fixed because XBLOCK=128 picks 1 block.
Anything beyond the autotuner's max-XBLOCK (~256 today) breaks. This is
the P3b-3 line: split into partial+combine kernels (R1 fix needs P3b-3 for
real workloads).

## Setup

Input: `tensor<Rxf16> + tensor<f16> → tensor<f16>`, single `linalg.generic`
with `iterator_types = ["reduction"]`. Through `python network_runner.py`
with autotune (`max-phase=5`). Default XBLOCK / RBLOCK search space:
powers of 2 from 16 up to ≤ R (capped at axis extent per commit `353e388`).

## Results

| R | autotune pick (XBLOCK / bdim) | expected | observed | max_diff | verdict |
|---:|---:|---:|---:|---:|:---|
|  64 | 64 / **1**  |  -12.25  |  -12.24  |  0.008  | **PASS** (f16 reduce noise) |
| 128 | 128 / **1** |   -8.40  |   -8.39  |  0.008  | **PASS** |
| 129 | 32 / **5**  |   -8.30  |   -8.39  |  0.094  | **FAIL** (= sum(x[0:128]); block 0's partial wins) |
| 256 | 256 / **1** |   0.203  |   0.203  |  0.0002 | **PASS** (autotune fits the whole R in one block) |
| 512 | 256 / **2** |   2.26   |   2.06   |  0.19   | **FAIL** (2-way race) |
| 768 | 256 / **3** |   -9.22  |   2.06   | 11.3    | **FAIL** (3-way race; output ≈ R=512's value — GM workspace residue) |

The "max XBLOCK in search space" is the boundary. For R ≤ that max,
autotune can pick XBLOCK = R → 1 block → correct. Beyond it, multi-block
is forced.

## Mechanism (why multi-block fails)

After P3b-2c/d the generated AscendC body is:

```
get_block_idx → block_offset = block_idx * XBLOCK
if block_offset < R {
  acc = 0 (VECCALC)
  for i in 0 .. min(XBLOCK, R - block_offset) step RBLOCK {
    load x[block_offset + i : block_offset + i + RBLOCK] into VECIN
    reduce_sum_2d_l2 partial = sum(VECIN)
    add_l2 acc += partial
  }
  data_copy_l2 gm_output, acc, 1     // ← every block writes the same 1-elem slot
}
```

Every block writes to the SAME rank-0 GM output (no `[block_idx]` offset).
Camodel doesn't atomically serialize these stores, and there's no
cross-block reduction; so:

* one of the partials becomes the final value (race),
* and the kernel never adds them together.

R=129 / R=512 / R=768 all reproduce this pattern. R=256 looks PASS only
because the autotuner picked XBLOCK = R = 256 → 1 block.

## What works (and is now in the e2e gate by induction)

- R1 minimal reproducer (R=16 fp16, rank-0 output): PASS, `max_diff=0.004`.
- R3 reproducer (axis-1 reduce, 8x16→8): unchanged, PASS.
- All 12 existing e2e gates: PASS (none of them hit `yAxes.empty()`, so
  RCore is never selected for them).
- lit 27/27: PASS.

## Why P3b-3 is necessary for real workloads

NanoGPT-scale softmax / LayerNorm reduce over D = 768 / 1024 / 2048 /
hidden_size. Three constraints, all pushing past the single-block boundary:

1. **Search-space cap**: XBLOCK candidates are powers of 2 up to ≤ R, and
   autotune's actual practical max is around 256 for the on-chip UB
   budget — even if we expanded the search space, fitting R = 2048 fp16
   on-chip whole is 4 KiB, doable, but plus the input read tile, broadcasts,
   etc., it exhausts UB.
2. **Block-dim parallelism**: single-block = single AICORE = ignores 24
   parallel cores. R = 2048 on 1 core wastes 23×.
3. **Race**: even if a single block fits, it doesn't generalize — *anything*
   with R > the search-space max will silently corrupt.

Conclusion: P3b-2 fixes the IR-legality bug (R1) but is operationally
useful only for tiny full-reduces. **P3b-3 (partial→combine dual-kernel)
is the real R1 fix for production workloads**, and the necessity is now
empirically confirmed rather than just theoretical.

## Recommended next step

Enter P3b-3 design review per plan §5. The three open questions Q1–Q3
in the plan doc:

- **Q1 (where to split)** — preliminary lean was post-outline / tensor
  level. The probing here doesn't disambiguate Q1 but does add an
  observation: the materialize_in_destination → memref<f16, strided<[],
  offset:?>> chain forced a CannTranslation patch to recover `__gm__`.
  Whatever P3b-3 chooses for workspace representation, the same kind of
  strided/offset memref will appear (workspace per-block scalars are
  inherently offset views) and must be handled cleanly upstream rather
  than patched in CannTranslation.
- **Q2 (workspace tensor)** — partial output shape is `tensor<block_dim
  × ...>` (rank-1 here since post-reduce shape is rank-0). block_dim is
  symbolic at codegen time (a tunable). Workspace needs SSA alloc
  whose size depends on the tunable.
- **Q3 (combine kernel block_dim)** — for nanoGPT-scale R, combine reduces
  ~24 partials → 1 scalar. Single block (bdim=1) is fine.

These should be answered in a sub-spec before any code lands.
