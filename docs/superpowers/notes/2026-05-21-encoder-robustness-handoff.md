# Transformer-encoder robustness + run-it-on-sim — handoff (2026-05-21)

Branch: **`dev-network`** (worktree `.claude/worktrees/encoder-robustness`, based on
`develop@a35c63c`). `develop` untouched (advanced separately to `0b49872`).

## Goal
Run a real network (single-layer transformer encoder) through the `network_runner`
mixed path on camodel sim to test robustness / numerical correctness.
Strategy: **fuse what we can → AscendC; what we can't (matmul/softmax/layernorm) →
aclnn CPU-reference fallback.**

## What works now (all validated max_diff=0 on camodel sim)
- `constarg4` — scalar const (fill 2.0) + add, 4x4.
- `wadd` — inline-dense tensor **weight** const + add, 4x4.
- `mm` — standalone matmul → **aclnn** `run_Matmul`.
Full lit: **90 pass / 0 fail**.

## Commits on dev-network (in order)
1. `dfee99d` chore: commit untracked TmTensor/EliminateCfAssert (develop never git-added them; build needs them)
2. `8a27d07` fix(auto-fuse): 4-layer group analysis/outline robustness
   - shared-scalar horizontal-fusion bug; glue-aware `wouldCreateCycle`;
     `reorderGroupsContiguous`; defensive non-contiguous guard.
3. `1d4df29` feat: emitNetworkJson const/alloc/slice provenance
4. `fd08b73` feat: tensor weight const baking + scalar-only rematerialization
5. `3f9cc75` feat: dense_resource weight baking (torch-imported weights)
6. `13ea2f9` feat: matmul → aclnn (run_Matmul CPU ref + outliner aclnn.op stamp)
7. `23cd647` feat: batch_matmul is a cube op + `disable-cube-fusion` option
8. `47fe249` feat: network_runner routes matmuls to aclnn by default

## Encoder status
Generate WITH weights (else dense_resource is elided / no data):
```
python3 python/torch/torch2linalg/net/transformer_encoder.py \
  --dtype fp32 --batch 2 --seq 8 --d-model 64 --nhead 2 --dim-ff 128 \
  --keep-weights -o /tmp/encoder_kw.mlir
```
Phase-1 (analysis+outline, disable-cube-fusion) now succeeds: **6 aclnn matmuls +
19 AscendC vector/reduce kernels** (was: SIGSEGV, then 13 all-ascendc cube-fused).
All 12 resource weights bake into the host with real data (0 elided).

**Broadcast wall — FIXED** (commit b463454): `DecomposeMultiAxisBroadcast`
greedily peeled the first foldable axis; for a leading multi-axis bias broadcast
`[1,1,N]→[D0,D1,N]` it peeled the outer axis first and stranded the inner
(non-1 prefix AND suffix) → rank-3 op → PyAsc's illegal `reinterpret_cast`. Fixed
with 1-step lookahead (peel inner axis first → two valid 2D row broadcasts).
+1 lit `codegen-bcast-leading-multiaxis.mlir`.

**Transpose wall (kernel_group6) — FIXED** (another session, on dev-network,
`Collapse.cpp` +38/-1, uncommitted at time of writing; +lit
`test/Conversion/Collapse/collapse-b2-glue-transpose.mlir`).
NOTE: this handoff's earlier root-cause guess (unit-dim / `map.numResults <
operand.rank`) was WRONG. **Real root cause:** `Collapse.cpp`'s B2 safety gate
`hasAnyB2` only treated function arguments as boundary inputs.  After
`fold-unit-extent-dims` inserts `collapse_shape` glue, the generic reads the glue
result `%c` (not the func arg), so `hasAnyB2` silently skipped the gate and
force-collapsed the transpose → illegal generic (operand rank ≠ map results) →
slice crash.  Not unit-dim specific (non-unit collapse+transpose also crashed;
identity+glue did not).  Fix: `isBoundaryDerived()` looks through
collapse_shape/expand_shape to the real boundary input so glued transposes take
the same `noCollapse` path as bare ones; + a `#ifndef NDEBUG` consistency assert
after collapse.  Validated: t5u clean, full lit 93 pass.

So phase-2's only remaining walls are the 3 reduce kernels (softmax g13,
layernorm g29/g56) — all covered by the aclnn-direct-call plan below.

**Attention → aclnn (softmax wall g13) — DONE** (commit `b0974cd`): new
linalg-level pre-fusion pass `recognize-attention`
(`lib/Conversion/LowerNonLinalgOps/RecognizeAttentionPass.cpp`). Anchors on the
2nd batch_matmul, walks back from its lhs through the softmax glue
(generic/collapse/expand/fill) to the 1st batch_matmul; the chain must contain a
`math.exp` (that's what distinguishes attention from two arbitrary chained
matmuls). Extracts Q=bmm1.lhs, K=transpose(bmm1.rhs last two), V=bmm2.rhs;
expands each [BH,S,D]→BNSD [BH,1,S,D] (B*H heads in the batch, N=1, since SDPA is
per-(b,n) independent); emits `@__aclnn_flash_attention` (shares the
aclnn.kind="flash_attention" contract + aclnn-finalize-decl with
convert-tm-tensor-attention). Scale 1/sqrt(D) + softmax recomputed inside FA, so
the matched region is left dead for DCE. Matching is topological (not tied to the
exact softmax shape), so it survives the scale-mul / dead-argmax / collapse-expand
glue the torch importer emits. Verified on encoder: 5 batch_matmul→3, exp gone, 1
FA call with correct Q/K/V. +1 lit `test/Conversion/recognize-attention.mlir`.
**NOT yet wired into the network_runner phase-1 pipeline** — must run
pre-group-analysis (before `--auto-fuse-group-analysis`); next step.

NOTE: `--recognize-attention` assumes the model's attention scale == 1/sqrt(D)
(headDim). Standard nn.MultiheadAttention matches; a custom scale would
double/mis-apply (FA always uses 1/sqrt(headDim) in sdpa_cpu/aclnn).

Pre-existing lit fail (NOT from this work): `tools/examples/example-pipelines.mlir`
`broadcast-add-reduce` at the AiCore single-dim multicore scheduling step
(AscendCParallelize) — in the TileFuse/Collapse.cpp path the other session is
editing (uncommitted). Full lit 95/96.

## Roadmap to encoder numerical PASS (remaining, each multi-step)
1. **Broadcast rank-3 codegen** (the current wall) — fix the fold workaround or
   the static_cast. Then re-run phase-2; expect more vector/reduce codegen walls.
2. **softmax / layernorm reduce kernels** — among the 19 AscendC kernels; reduce
   codegen may need work (see [[project_reduce_path_r3]]).
3. **layernorm → aclnn** option if reduce codegen too hard: add `run_LayerNorm`
   CPU ref + recognition (mirror matmul path).
4. **attention → aclnn FlashAttentionScore** (already in AclnnOps.cpp) via pattern
   recognition.
5. **Reference output**: need a PyTorch run of the encoder for `--expected`.
6. **Weight runner note**: weights are baked into the host C++ (static arrays) at
   aclnn-backend generate time — works but bloats for huge weights; revisit
   (NpyIO load) if needed.

## How to run a small net e2e (env)
```
source /home/gser/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/home/gser/Ascend/cann-9.0.0/x86_64-linux/simulator/Ascend910B1/lib:\
/home/gser/Ascend/cann-9.0.0/x86_64-linux/lib64:\
/home/gser/Ascend/cann-9.0.0/x86_64-linux/devlib/linux/x86_64:$LD_LIBRARY_PATH
export PATH=$PWD/build/bin:$PATH
PYTHONPATH=python python3 python/network_runner.py --input-linalg <net.mlir> \
  --inputs ... --expected ... --workdir <W> --soc Ascend910B1 --atol 1e-2 --rtol 1e-2
```
NOTE: 1-D / 8-element elementwise → all-zero output (pre-existing small/1-D tiling
bug, NOT from this work; use ≥2-D / 16-multiple shapes). two-elewise-e2e (4x4) passes.

## Build (worktree)
Shares the main repo's prebuilt LLVM + submodules via symlinks; own `build/`.
`ninja -C build afir-opt aclnn-backend` after edits; `afir-translate` is built.
`AclnnOps.cpp` is compiled into the generated host at link time (no ninja needed
for run_* changes to take effect in a run).
