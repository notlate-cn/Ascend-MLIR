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

**Next wall (phase-2 codegen, kernel_group6):** a rank-5 `linalg.transpose` with
a UNIT dim (`[1,8,2,3,64]→[3,8,2,1,64]` perm `[3,1,2,0,4]`, from attention head
reshapes) SIGABRTs in `GroupEmitter`/`computeSlice`
(`ExtractSliceOp::inferResultType`: staticSizes count != source rank). Root:
`--linalg-fold-unit-extent-dims` does NOT fold `linalg.transpose`, so the unit
dim survives; the codegen Collapse step then yields an indexing map with
`numResults < operand.rank` (unit dim dropped from the map, not the operand) and
`SliceComputer.cpp:79` emits one slice dim per map result → rank mismatch.
Repro: rank-4 transpose OK; rank-5 with leading unit dim crashes. Fix options:
(a) fold/lower unit-dim transpose to rank-(N-1) transpose + reshapes before
codegen; (b) make computeSlice/GroupEmitter robust to map.numResults <
operand.rank (full-size slice for untiled/collapsed dims). Then expect the
softmax/layernorm reduce kernels as the next walls.

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
