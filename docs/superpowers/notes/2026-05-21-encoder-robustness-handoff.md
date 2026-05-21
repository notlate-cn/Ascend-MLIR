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

**Transpose wall (kernel_group6) — FIXED** (commit a9a2f04 on dev-network,
`Collapse.cpp` +38/-1; +lit
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
after collapse.  Validated: t5u clean, full lit 94 pass.

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

**Attention WIRED + LayerNorm → aclnn DONE** (commits `3f36271`, `2833937`):
- recognize-attention wired into network_runner phase-1 (commit 3f36271, +fix:
  FA dummy mask/init reused Q's value instead of standalone tensor.empty, which
  the group-outline reorder sank past the call → invalid SSA / no json provenance).
- `recognize-layernorm` (`RecognizeLayerNormPass.cpp`) + `run_LayerNorm` CPU ref
  (AclnnOps.cpp) + registry `{"layer_norm",{"LayerNorm","ND"}}`. Anchors on
  math.rsqrt, walks to x / gamma / beta, folds mean→sub→var→rsqrt→norm→*g+b into
  `@__aclnn_layer_norm(x,gamma,beta)`. eps recomputed inside (torch default 1e-5).
  +1 lit `recognize-layernorm.mlir`.
Encoder now: **phase-1** = 1 FlashAttentionScore + 3 Matmul + 2 LayerNorm aclnn +
AscendC kernels; **phase-2** = ALL kernels codegen+compile clean (both LN walls
gone). Full lit 95/97 (2 fails = pre-existing example-pipelines + flaky
runtime-focused-verification which PASSES in isolation).

**CURRENT WALL (phase-3 host build, NEW + separate, pre-existing gap):**
`network_host_default.cpp` fails to compile — `TensorInfo t41[4] =
{/*unknown*/, t40, ...}`. The AclnnBackend **CoordEmitter** (host C++ generator,
`lib/Runtime/AclnnBackend/AclnnBackend.cpp` ~line 48-72) propagates names through
cast/collapse/expand/empty/const but **NOT `tensor.extract_slice`**. The encoder's
QKV head-split (`extract_slice %collapsed[i,...]` → collapse → expand → kernel_group7)
hits unknown. NetworkJsonEmitter DOES handle slice; the phase-3 C++ host path does
not. Fix: teach CoordEmitter to emit a host-side strided copy for extract_slice
(static offsets/sizes/strides), mirroring the const-baking. NOT attention/LN
related; exposed because the encoder is the first net to host-gen with slices.

NOTE: env to run e2e — also need `LD_LIBRARY_PATH` = sim + lib64 + devlib (see
"How to run" below), else runtime-session can't load `libnpu_drv_camodel.so`.
Reference gen: `/tmp/gen_encoder_ref.py` (seed 0, builds model + dumps
input0.npy/expected0.npy/encoder.mlir consistently).

**CoordEmitter host-gen completeness — DONE** (commit `3c753cb`): 3 fixes in
`AclnnBackend.cpp` exposed by encoder host-gen (first net with slices/reshapes):
(1) `tensor.extract_slice` → host strided copy; (2) `tensor.empty` → actually
allocate (was unallocated → SaveNpy null-deref SIGSEGV); (3) collapse/expand →
reshape *view* with result shape/rank/strides (was name-only alias → FA q.rank
!=4 assert). **Encoder now runs FULL e2e through phase-3** on camodel (all AscendC
kernels + aclnn FA/LayerNorm/Matmul execute, output produced, intermediates
dumped). lit 95/97 unchanged.

**CURRENT WALL — AscendC transpose kernel ACCURACY (numerical, NOT pipeline).**
Whole-encoder output max_abs_diff=0.98 vs torch. Root cause traced by comparing
each kernel's OWN dumped in→out to a numpy transpose oracle
(intermediates_default/):
  - `kernel_group0` [2,8,64]→[8,2,64] perm[1,0,2]: **ALL-ZERO output** (pure
    transpose, totally broken — the "unreasonable XBLOCK ≫ extent → zero output"
    class, see [[project_network_runner_v1]] / [[project_multi_input_dyn_reduce_bug]]).
  - `kernel_group6` rank-5 perm[3,1,2,0,4] & `kernel_group10` rank-4 perm[0,1,3,2]:
    **tail wrong** (max 0.088, exactly the LAST batch tile's region — 128/1024
    elems; first tiles correct → multicore/tail boundary bug).
  - `kernel_group11` perm[0,2,1] & `kernel_group12` perm[2,0,1,3]: CORRECT (0).
  aclnn FA/LayerNorm/Matmul are CPU-exact (trusted). So the entire numerical gap
  is AscendC transpose codegen (zero-output + tail) — the other session's
  transpose/Collapse.cpp/tiling domain. Phase-4 autotuner also fails here
  (kernel_group6 "no variant passes", max 0.088).
**DEEP TILING DIAGNOSIS (2026-05-21):** drilled into whether the transpose errors
are tiling-driven or kernel-codegen. Findings:

A. **Picker bug (REAL, network_runner.py:444-461, MY territory):** the phase-3
   default tiling picker caps EVERY tunable param by the single multicore
   `axis_extent_expr`. But that extent only bounds the multicore param (the one in
   `block_dim_expr`, = `XBLOCK`); inner-axis params (`XBLOCK_X_0`, `XBLOCK_SUB`)
   belong to OTHER axes. Result: group0 `XBLOCK_X_0` capped 8→2 (its space=[8]);
   group10 capped [16,32]→2. Fix: only cap the param(s) named in block_dim_expr;
   for the rest pick from their own `values` (already the codegen-valid candidates).

B. **HostLaunchHelper schema-staging gap (REAL, HostLaunchHelper.cpp:285-340):**
   the helper looks for `<artifacts>/<kernel>__v0/tiling_space.json` to order the
   tiling struct fields; it is NEVER staged (runtime-session/network_runner don't
   write it), so it falls to the alphabetical-key fallback ([XBLOCK, XBLOCK_SUB,
   XBLOCK_X_0]) which ≠ the kernel struct order (mlir_index: [XBLOCK_X_0, XBLOCK,
   XBLOCK_SUB]). Scrambles params when their chosen values differ (e.g. group6
   3/8/8). Fix candidate: stage each `kernel_groupN_space.json` →
   `<artifacts>/<kernel>__v0/tiling_space.json` in phase-2 (its `tiling_params`
   are already in mlir_index order; TilingSchema::fromJson reads that array order).

C. **BUT the transpose accuracy is NOT tiling-driven — it's KERNEL CODEGEN:**
   experiment — staged the (correct-order) schema with the original picker values
   → group6/group10 outputs **byte-identical** to the fallback (still 0.088),
   group11/12 still correct. I.e. the tile-param values don't change these
   transpose kernels' wrong elements. Clincher: phase-4 **autotuner already tried
   all tiling variants for group6 → ALL fail at 0.088** ("no variant produced a
   passing configuration"). So group6/group10 tail errors are a genuine AscendC
   transpose codegen tail/boundary bug (other session's Collapse.cpp/transpose
   domain). group0 zero-output: picker gives XBLOCK_X_0=2, but bugs A+B prevented
   a clean test of whether X_0=8 fixes it; likely codegen too (its own variant
   never validated).

NET: two real network_runner/HostLaunchHelper plumbing bugs (A picker cap, B
schema staging) worth fixing on their own, but fixing them does NOT fix the
transpose numerical errors — those are kernel codegen (autotuner-confirmed for
group6). NEXT (fix-session): (1) AscendC transpose tail/zero codegen [other
session]; (2) optionally A+B plumbing fixes [this side]. Repro: transpose-oracle
over `/tmp/enc_e2e/intermediates_default/kernel_groupN__v0_{in_0,out_0}.npy`;
tiling experiments via editing `/tmp/enc_e2e/tilings_default.json` +/- staging
`*_space.json` as `artifacts/<k>__v0/tiling_space.json`, re-run network_test_default.

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
