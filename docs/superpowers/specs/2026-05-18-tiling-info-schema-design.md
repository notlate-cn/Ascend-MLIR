# Tiling-Info Schema — End-to-End Single Source of Truth

**Date:** 2026-05-18
**Author:** gaoshuer
**Status:** design — pending plan

## 1. Context

Tiling-info today is split across three implicit contracts that nobody validates:

1. **Kernel MLIR** defines TilingData field names via `memref.dim` ops collected by `AscendCPackTilingDataPass`.
2. **`network.json`** (written by `NetworkJsonEmitter`) lists each kernel's input descriptors (provenance + shape), indexed by the kernel's **coordinator-call operand position** — input-only.
3. **`tiling_space.json`** (written by `CannTranslation`) carries `shape_key: "arg<N>_dim<D>"` field names that the host runner is supposed to resolve against (2).

The bridge from (1)→(2) is a heuristic in `PackTilingData::argToOriginal` (commit `287fd52`): each non-identity-layout MemRef arg is redirected to the first shape-equal identity-layout arg, falling back to `arg0`. This survives elementwise / broadcast / shape-preserving cases by accident — output shape matches some input. **It breaks the moment the kernel's output shape differs from every input shape.**

### Observed failures

- **`transpose-elementwise-e2e` sim segfault**: output is `[32,16]`, input is `[16,32]`. `argToOriginal` finds no shape-match, falls back to `arg0`, so the output's d1 stride symbol `dim_arg3_1` is folded into `dim_arg0_1`. Host sends `dim_arg0_1 = 32` (input d1). Kernel writes at row stride 32 instead of 16 → OOB GM writes per block → sim crash. Validation log shows "simulation output mismatch" alongside SIGSEGV.

### Adjacent inconsistencies (same root)

- After P1b multi-variant codegen (2026-05-14, `d102b36`+), `tiling_space.json` (the legacy back-compat path) is no longer rewritten when `aicoreFuncs.size() > 1`. Stale copies in the worktree mislead `run.sh` and autotuner.
- `afir.dim_symbols` / `afir.symbolic_shape` (shape-symbolization, 档 C) does not survive bufferize / canonicalize into `PackTilingData`, so the `symTable` canonicalization branch there is dead code in practice.
- `run.sh` files hand-assemble `TILING_PARAMS` strings — they reference field names that may have moved, disappeared, or never existed in the current struct. No build-time validation.

## 2. Goals

Establish **a single canonical schema per kernel** that drives every downstream consumer. After this lands:

- TilingData struct field names, types, and order — from schema.
- `tiling_space.json` `shape_key` and `fixed`/`tunable` classification — from schema.
- `network.json` kernel arg descriptors (covering inputs **and** outputs / workspace / tile-params) — from schema.
- `network_runner` `shape_key` resolution — from schema's MLIR arg numbering.
- `PackTilingDataPass::argToOriginal` heuristic — **removed**.

Out of scope:
- Static-const field promotion (kind="const"). Deferred until a use case appears; retroactive addition is metadata-only.
- C3 (full KernelSpec covering block_dim_expr, axis_extent_expr, etc.) — existing attrs are adequate.

## 3. Schema

### 3.1 Storage

- **Canonical**: extended `vector_plan.tiling_infos` ModuleOp attribute (already exists from P6a, holds tunables today).
- **Host-facing**: each per-kernel JSON sidecar (`<kernel>_space.json`, written by `CannTranslation`) carries the schema-derived fields. A new `<kernel>_abi.json` (or merged into `network.json` kernel entry) carries the per-arg provenance — chosen during implementation.
- **Version**: `schema_version = 2` on each `tiling_infos` entry. Consumers see v2 → new path; v1 → legacy path. Last-stage cleanup removes v1.

### 3.2 Shape

```mlir
vector_plan.tiling_infos = [{
  kernel_id        = "transpose_relu__v0",
  schema_version   = 2,
  block_dim_expr   = "ceil(32/XBLOCK)",
  axis_extent_expr = "32",
  constraints      = [{kind="divides", lhs="XBLOCK_SUB", rhs="XBLOCK"}, ...],

  // Ordered list = TilingData struct field order (drops abi_index).
  fields = [
    {name="XBLOCK",      kind="tunable", axis_size=32, default_value=128},
    {name="XBLOCK_SUB",  kind="tunable", axis_size=32, default_value=16},
    {name="dim_arg3_0",  kind="shape_derived", source_arg=3, source_dim=0},
    {name="dim_arg3_1",  kind="shape_derived", source_arg=3, source_dim=1},
  ],

  // Per-MLIR-arg provenance.  Drives NetworkJsonEmitter + runner shape_key lookup.
  args = [
    {mlir_index=0, role="input",       network_index=0},
    {mlir_index=1, role="tile_param",  name="XBLOCK"},
    {mlir_index=2, role="tile_param",  name="XBLOCK_SUB"},
    {mlir_index=3, role="output",      result_index=0,
                   shape_expr=["arg0_dim1", "arg0_dim0"]},
    {mlir_index=4, role="workspace"},
    {mlir_index=5, role="tiling_data_struct"},
  ],
}]
```

### 3.3 Field semantics

- `fields[i]` describes TilingData field at struct position `i`.
  - `kind="tunable"`: axis_size, default_value, constraints from existing P6a path.
  - `kind="shape_derived"`: `source_arg` / `source_dim` use **MLIR arg numbering** (post-bufferize, with tile-param args interleaved). Names by convention `dim_arg<source_arg>_<source_dim>`.
- `args[i]` describes MLIR arg at position `i`:
  - `role="input"`: kernel reads it. `network_index` is the position in the coordinator-call operand list (= `network.json` kernel-args index).
  - `role="output"`: kernel writes it (DPS init memref). `result_index` keys into the kernel's `results` list in `network.json`. `shape_expr` is a SymExpr-style list giving each output dim in terms of input dims; runner evaluates it for shape resolution.
  - `role="tile_param"`: scalar (XBLOCK etc.) injected by TilePlanGen; `name` matches a `fields[].name`.
  - `role="workspace"` / `role="tiling_data_struct"`: special slots, not user-facing.
- Output `shape_expr` uses host-visible names (`arg<network_index>_dim<D>`), since the runner evaluates it in network-arg coordinates. For complex output shapes (reduce, matmul) the expression syntax mirrors existing SymExpr.

## 4. Authoring & Consumption

Schema is written in **two stages**, both before `PackTilingDataPass`:

```
TilePlanGen ─── writes fields[] (tunable + shape_derived)
GroupOutline ── writes args[] (per-arg roles + provenance)
        │
        ▼
PackTilingDataPass (consumer): builds TilingData struct from fields[],
                               rewrites memref.dim per fields[].source_*
CannTranslation (consumer):    writes <kernel>_space.json from fields[]
                               (shape_key = "arg<args[source_arg].network_index>_dim<source_dim>")
NetworkJsonEmitter (consumer): writes per-kernel args/results in network.json from args[]
network_runner (consumer):     resolves shape_key against schema args[]
```

`afir.dim_symbols` / `afir.symbolic_shape` are read primarily by TilePlanGen (to author `shape_expr`). **In practice, `PackTilingDataPass`'s schema path also re-reads them** to canonicalize equal-symbol dims onto one TilingData field — needed for `dyn-bucketed-e2e` where multiple inputs share shape symbols and the kernel emits `memref.dim` on each. Removing this canonicalization causes the dim ops to survive into emission. Other downstream consumers (CannTranslation, NetworkJsonEmitter, runner) do not read these attrs.

## 5. Migration

Phased, with old path live in parallel. Each step independently verifiable on the existing e2e gate suite (14 kernels + lit 78).

| Step | Change | Verification |
|------|--------|--------------|
| **S0** | Pre-fix #2/#3 (bcast `memref.collapse_shape` printer error) so schema work isn't blocked by unrelated noise. | bcast-leading-e2e / bcast-trailing-e2e PASS. |
| **S1** | TilePlanGen + GroupOutline emit schema v2 fields. Old path unchanged. | Lit checks attr structure on representative kernels. |
| **S2** | PackTilingData reads schema when present (`schema_version=2`), else falls back to `argToOriginal`. | All existing e2e still PASS; transpose-elementwise PASS (new). |
| **S3** | CannTranslation + NetworkJsonEmitter switch to schema-driven emission. Three e2e categories must PASS: shape-equal (elementwise), shape-not-equal (transpose, reduce), rank-mismatched (split-rcore, dyn-bucketed). | Full e2e + lit. |
| **S4** | `network_runner` resolves `shape_key` via schema `args[]`. e2e `run.sh` files drop hand-written `TILING_PARAMS` for shape-derived fields; host derives them. Tunables (XBLOCK etc.) still pinned in `run.sh`. | Full e2e + autotuner cross-variant. |
| **S5** | Delete `argToOriginal`, the dead `symTable` canonicalization branch, and `schema_version=1` codepaths. | grep clean, lit clean, e2e clean. |

Estimated commits: 1–3 per step. transpose-elementwise bug is fixed at end of S2.

## 6. Risks / Open Questions

- **`shape_expr` complexity for reduce / matmul**: SymExpr already handles dyn-bucketed cases; complex output rank changes (e.g., RCore partial → combine) may need additional SymExpr ops. Investigate during S1 authoring.
- **autotuner field-name expectations**: currently keys by literal `dim_arg<N>_<D>`. May need updates when shape_key resolution moves under schema. Inspect `tools/autotuner/autotuner_main.cpp` during S4.
- **Host-side `<kernel>_abi.json` vs merging into `network.json`**: deferred to implementation; both viable.
- **Schema validation timing**: should CannTranslation refuse to emit when schema is missing / inconsistent, or warn-and-fallback? Recommend hard-fail post-S5.

## 7. Acceptance

- `transpose-elementwise-e2e` `session.validation=pass`.
- All existing e2e gates remain PASS.
- `argToOriginal` removed from the tree.
- `afir.dim_symbols` / `afir.symbolic_shape` read only by TilePlanGen and by PackTilingData's schema path (for shared-symbol canonicalization). No other downstream reads.
- Lit + e2e CI green.
