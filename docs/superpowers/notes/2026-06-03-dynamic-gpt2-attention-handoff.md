# Dynamic-shape GPT-2 (full network) — progress + LayerNorm blocker

Date: 2026-06-03
Branch: `develop`
Status: dynamic attention + dynamic LayerNorm both work; 2-layer dynamic GPT
clears phase-1 + phase-2 codegen; now blocked in phase-3 (host) on an
unresolved `-1` dynamic dim propagated through a coordinator `collapse_shape`.

## Update — LayerNorm cracked + codegen walls (committed)

Walls broken since the original blocker below (all on `develop`):
- `7914d11b` **dynamic LayerNorm**: new `--lower-broadcast-extract` pass raises
  torch's extract-based dynamic broadcast (`dim==1 ? 0 : idx` guarded
  `tensor.extract`) to a clean affine-map `ins`-broadcast, so
  RecognizeLayerNorm's `isBroadcastCopy` matches → LN folds to
  `@__aclnn_layer_norm` (no decomposed kernel → no outliner cycle). Guard dropped
  only when its tested dim is the SAME SSA as the output init's loop extent
  (CSE'd via `--canonicalize --cse` first). `findBcast` also traverses
  `expand_shape`. phase-1 reordered: recognize-attention, then canon+cse, then
  broadcast-lower + recognize-layernorm.
- `d32d0ee0` **cross-arg memref.dim**: PackTilingData now canonicalizes the
  ShapeDerived field's `(sourceArg,sourceDim)` (not just the dim query) so a dim
  shared across args via shape_equalities (residual `collapse_shape`:
  arg3.dim1 == arg0.dim0) resolves to its TilingData member instead of leaving a
  `memref.dim` ascir-translate can't print.

**Current wall (phase-3, host):** `shape_equalities[kernel_group8]: arg0.dim0=-1,
arg3.dim1=20`. kernel_group8's arg0 is `%collapsed_66` (a coordinator
collapse_shape); its dynamic dim reaches the host as the `-1` placeholder
instead of 20. The host shape-propagation (AclnnBackend CoordEmitter +
runner_utils) doesn't resolve this intermediate's dynamic dim. NEXT: trace
`%collapsed_66`'s source in the coordinator and fix where the `-1` enters (an
alloc/kernel-result/reshape whose dynamic dim wasn't filled from the input
shape). Likely more host shape-propagation gaps + then dynamic reduce /
embedding / lm_head behind it.

---

## Original blocker (now resolved — kept for context)

## Goal

Compile a **full dynamic-seq GPT-2** (variable sequence length) from torch
through `network_runner`. Builds on the dynamic-shape elementwise path
(`2026-06-02-dynamic-shape-from-torch-handoff.md`, validated on real 910C).

## Done — dynamic attention (committed)

A single-layer dynamic-seq attention passes sim end-to-end (`max_diff=1.8e-7`).
Commits on `develop`:

- `41098902` **dynamic attention**: RecognizeAttention accepts dynamic rank-3
  Q/K/V (dropped the `hasStaticShape` guard; requires only BH/D static), builds
  the K-transpose init / BNSD expands / result collapse with dynamic S via
  `tensor.dim`/OpFoldResult. Dynamic causal mask threaded as a **value-less
  `tensor.empty[1,1,S,S]` sentinel** (FlashAttentionScore regenerates the
  triangular mask from the runtime seq length, so the in-graph `triu` DCEs —
  its index/compare/select chain can't be tiled). `sdpa_cpu` regenerates causal
  from S, matching the device. `AclnnBackend.emitExtractSlice` gained a dynamic
  path (static offset/stride, rank-reducing, runtime sizes) for the qkv `split`.
- `a2b40f3b` **--remove-cf-assert** registered in afir-opt + run first in
  network_runner phase-1: strips torch's `cf.assert "dynamic negative broadcast
  sizes"` guards, which (side-effecting) pin dead in-graph mask shape-arith in
  the coordinator that the network-json emitter can't represent.

Repro: `/tmp/attn/attn.mlir` (single-layer dynamic attention) — PASS.

## Blocker — dynamic LayerNorm

A 2-layer dynamic GPT (`/tmp/gpt2dyn/m.mlir`) fails at phase-1 outline:
`emitNetworkJson: call arg 3 of kernel_group3 has no provenance descriptor`.

Root cause chain:
1. `RecognizeLayerNorm` does **not fire** on the dynamic LN. It anchors on a
   `math.rsqrt` generic then walks `rstd -> isBroadcastCopy -> mul -> sub`. But
   torch lowers the **dynamic** mean/rstd broadcast as
   `linalg.generic { %i = linalg.index; %sel = (dim==1) ? 0 : %i;
   tensor.extract src[0,%sel,0] }` — an **extract-based broadcast** with a
   `dim==1 ? 0 : idx` guard (torch's defensive broadcast-of-a-maybe-1 dim).
   `isBroadcastCopy` only matches a clean affine-map broadcast, so the match
   fails and the LN stays decomposed (mean/sub/sq/sum/div/rsqrt/normalize).
2. The decomposed LN is grouped + outlined into `kernel_group3`. Its rstd `%13`
   (`tensor<?xf32>`) is broadcast back via an `expand_shape [?]->[1,?,1]` that
   the outliner left in the **coordinator** (`%65`) and fed back as the kernel's
   own arg 3 — but `%65 = expand_shape(%64#0)` derives from `kernel_group3`'s
   own output `%64#0`. A use-before-def cycle: the kernel consumes a reshape of
   its own output. emitNetworkJson chokes (no provenance for `%65` when it
   processes the call).

So there are two ways to unblock, both non-trivial:

- **(A) Make RecognizeLayerNorm fire on dynamic LN** → single `@__aclnn_layer_norm`
  call, no decomposed kernel, no cycle. Needs the recognizer's broadcast/sub/mul
  matchers to also accept the **extract-based dynamic broadcast** (`linalg.index`
  + `cmpi eq 1` + `select` + `tensor.extract`). This is the cleaner end-state and
  mirrors how attention was handled. Risk: fragile pattern matching; the same
  extract-broadcast appears in the mean-subtract too.
- **(B) Fix the outliner cycle** generally: sink the reshape glue (`expand_shape`
  of an in-group value) into the kernel so it isn't a coordinator boundary op.
  This is glue-aware grouping (cf. the `wouldCreateCycle` glue work in
  GroupAnalysis). Helps any decomposed-op-with-reshape-glue, not just LN.

### Higher-leverage idea (recommended to evaluate first)

The extract-based broadcast's `select((dim==1), 0, idx)` guard is **spurious**
when `dim` is a symbolized iteration extent (the seq dim, which the consuming op
iterates over — `idx` is always in range, and the source's broadcast dim is the
*feature* dim, not seq). A canonicalization that drops the guard (`select ->
idx`) for dims known via `afir.symbolic_shapes` to be the iteration extent would
turn the extract-based broadcasts back into clean affine-map broadcasts
**everywhere** — fixing LN recognition (A) and simplifying the whole dynamic
path in one place, instead of teaching every recognizer the messy form. This is
the recurring obstacle (it also forced the `triu` mask down the codegen-wall
path). Worth prototyping before committing to (A) or (B).

## Likely walls behind LN (not yet reached)

Full dynamic GPT-2 still needs, after LN: dynamic MLP (erf-GELU on `?` — should
work, it's the gelu-dyn path), dynamic residual adds, weight-tied lm_head /
embedding (`RecognizeEmbedding` may have static assumptions), and the in-graph
causal mask only validated single-layer. Each may surface more.

## Repro

```bash
# 2-layer dynamic GPT (random weights) — fails at LN outline
source examples/env.sh; export PATH=$PWD/build/bin:$PATH
NETWORK_RUNNER_SKIP_AUTOTUNE=1 PYTHONPATH=python python python/network_runner.py \
  --input-linalg /tmp/gpt2dyn/m.mlir --inputs /tmp/gpt2dyn/x.npy \
  --expected /tmp/gpt2dyn/exp.npy --workdir /tmp/gpt2dyn/w --soc Ascend910B1 \
  --backend sim --atol 1e-2 --rtol 1e-2
# kernel_group3 (decomposed dynamic LN) shows the self-referential arg 3.
```

See `project_gpt2_real_npu` (memory) for the broader bring-up;
`2026-06-02-dynamic-shape-from-torch-handoff.md` for the elementwise base.
