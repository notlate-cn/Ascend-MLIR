# ascend-debug Re-target to `develop` — Design

**Date:** 2026-06-03
**Status:** Approved (design phase)
**Scope:** Port the `ascend-debug` visual debugging tool from `upstream/dev-nyh` onto the `develop` branch, MVP = `collect` + `open` subcommands including the per-stage structural graph.

---

## 1. Background & Problem

`upstream/dev-nyh` carries a ~14.6k-line `ascend-debug` tool (CLI + HTML dashboard) that visualizes the Ascend lowering pipeline stage-by-stage, with IR views, a structural graph, kernel DAG, memory view, tensor diff, and first-bad-kernel location.

A literal lift onto `develop` is **not feasible**. `develop` and `dev-nyh` both branched from `upstream/main` (`b836a97d`) and diverged independently (399 vs 331 commits). `ascend-debug` is welded to dev-nyh's pipeline architecture, which `develop` does not have:

| | `develop` | `dev-nyh` |
|---|---|---|
| opt tool | `afir-opt` | `ascend-mlir-opt` |
| main namespace | `afir` (83 files) | `ascend` (78 files) |
| lowering model | `--auto-fuse-codegen` single registered pipeline (AutoFuse/Recognize/AscendC passes) | named coarse stages `--ascend-normalize/kernelize/schedule/realize/...` |
| `debug-stage`/`debug-dump-dir`/`dump-report` pass options | absent | present (the C++ DebugOptions integration) |
| stage markers | `auto_fuse.*` / `aclnn.*` / `afir.*` | `ascend.kernel` / `ascend.op_role` / `ascend.schedule.*` |

The tool's `collect` drives dev-nyh's named stage passes and reads `ascend.*` markers; `develop` has neither. Therefore "port" = **re-target** the tool to `develop`'s real pipeline and metadata.

## 2. Goals / Non-Goals

**Goals (MVP):**
- `ascend-debug collect <ir.mlir>` runs `develop`'s lowering in 6 coarse stages and dumps numbered IR artifacts + `manifest.json`.
- `ascend-debug open <run-dir>` generates a read-only HTML dashboard: stage navigation, per-stage IR views, and the per-stage **structural graph** (`stage_graph.py`).
- Zero C++ changes.

**Non-goals (follow-up):**
- `diff` / `locate` subcommands (tensor compare, first-bad-kernel).
- C++ per-kernel checkpoint backend (`DebugOptions.cpp`, `DebugCase.cpp`) and the `debug-dump-dir` pass option.
- Unified graph workspace (`debug_graph.py`, ~2793 lines).
- Memory timeline (`memory.py`) populated with runtime data.
- Optional C++ `ascend-stage-graph` tool (off by default; Python path suffices).

## 3. Architecture & Data Flow

```
ascend-debug collect <ir.mlir> [--out DIR]
  └─ for each of 6 STAGES: afir-opt <prev-out> <stage-pass-list> -o DIR/stages/NNN-<stage>-out.mlir
     └─ write DIR/manifest.json (schema_version, stages[], status, ...)

ascend-debug open <DIR>
  └─ load_manifest(DIR)
     └─ render HTML: stage list + per-stage IR view + per-stage structural graph
```

**Dump mechanism:** sequential independent `afir-opt` invocations, each consuming the previous stage's output and dumping its own — the same model `dev-nyh`'s `collect` already uses. No in-pass `debug-dump-dir` hook needed, hence no C++.

## 4. Stage Table (built into `collect`)

Six coarse stages, derived by splitting `develop`'s `--auto-fuse-codegen` pipeline (`lib/Conversion/AutoFuse/Pipeline.cpp`) at logical boundaries. Order is preserved exactly; only dump points are inserted.

| # | stage | pass sub-sequence |
|---|---|---|
| 010 | normalize | `linalg-generalize-named-ops`, `linalg-fuse-elementwise-ops`, `linalg-fold-unit-extent-dims`, `canonicalize` |
| 020 | kernelize | `AutoFuseGroupAnalysis`, `AutoFuseGroupOutline` |
| 030 | schedule | `AutoFuseRestoreMatmul`, `AutoFuseIsolateKernelOutputs`, `AFIRSymbolizeShapes`, `AutoFuseTileFuse`, `canonicalize` |
| 040 | realize | `AnnotateAscendCKernelKind`, `AutoFuseFoldShadowAlloc`, `AutoFuseInsertTileBuffers`, `AscendCBufferPlacement`, `LinalgToAscendC` |
| 050 | parallelize | `AscendCDecomposeMultiAxisBroadcast`, `AscendCParallelize`, `AscendCFlattenGMPtr`, `canonicalize`, `cse` |
| 060 | finalize | `AutoFuseVerifyTilingInfoSchema`, `AscendCPackTilingData`, `AscendCFinalizeKernel`, `CanonicalizeCannSignature`, `AscendCRCoreCombine` |

The exact pass-flag spellings are the `afir-opt` command-line names corresponding to each `create*Pass()`; the implementation plan resolves each flag against the registered pass arguments.

**Equivalence risk:** running sub-sequences as separate processes (vs one in-process pipeline) must yield equivalent IR. func-nested vs module passes are stable across invocations (dev-nyh already relies on this). If a stage fails because it needs prior pipeline state, merge it into the adjacent stage. Each stage is verified to pass `afir-opt` during implementation.

## 5. `open` / Dashboard

- **Manifest-driven** (`open_view.py`): reads `manifest.json` + each stage artifact path; renders stage list and per-stage IR text views. Ports ~as-is (tool name and contract unchanged).
- **Per-stage structural graph** (`stage_graph.py`), two layers:
  - *Generic structure layer* — func decls, `linalg` out values, memory effects, resource access parsed from MLIR text. Works on any IR, no marker dependency. Ports as-is.
  - *Semantic badge layer* — rewrite `_build_semantic_attrs` / `_build_node_badges` to read `develop`'s attrs instead of `ascend.*`:

    | dev-nyh badge | develop attr |
    |---|---|
    | `ascend.kernel` (kernel id) | `auto_fuse.group_id` / `auto_fuse.topo_index` |
    | `ascend.op_role(s)` / `ascend.primary` | `auto_fuse.kind` / `aclnn.op` / `aclnn.kind` |
    | `ascend.kernelize.template_families` | `auto_fuse.tiling_infos` / `afir.reduce_template` |
    | `ascend.schedule.{tile_params,family,template,tile_binding,...}` | `auto_fuse.tiling_infos` (v2 source of truth), `auto_fuse.default_tile_size`, `afir.block_dim_expr`, `afir.axis_extent_expr` |

- **Graceful degradation:** `open_view.py` imports `memory` and `debug_graph`; with no runtime data / workspace not in scope, those sections render empty rather than erroring.
- `ui_text.py`: keep the coarse stage names/descriptions (6 stages match dev-nyh's labels, so descriptions stay meaningful).

## 6. Modules to Port (Python only, under `tools/ascend-debug/`)

| module | action |
|---|---|
| `ascend-debug.py` | port, trim CLI dispatch to `collect` + `open` |
| `ascend_debug/__init__.py` | port (version string) |
| `ascend_debug/layout.py` | port (numbering, manifest/provenance writers) |
| `ascend_debug/runner.py` | port; `find_tool` default `ascend-mlir-opt` → `afir-opt` |
| `ascend_debug/collect.py` | **rewrite** stage table to §4; drop dev-nyh `--ascend-*` flags |
| `ascend_debug/open_view.py` | port ~as-is |
| `ascend_debug/stage_graph.py` | port; **adapt** semantic badge extraction to §5 attrs |
| `ascend_debug/ui_text.py` | port (coarse stage descriptions) |
| `ascend_debug/memory.py`, `debug_graph.py` | port enough to satisfy `open_view` imports; degrade gracefully (not in MVP feature scope) |

Build wiring: `tools/ascend-debug/CMakeLists.txt` installs the wrapper to `build/bin/ascend-debug`; `tools/CMakeLists.txt` adds the subdir; `test/CMakeLists.txt` adds `ascend-debug` to test deps.

## 7. Testing

- **lit smoke** `test/tools/diagnostics/ascend-debug-cli.mlir` + `test_ascend_debug_cli.sh` (trimmed to collect+open): feed a small `linalg.generic` IR → `collect` → assert 6 numbered artifacts + manifest field ordering → `open` produces HTML without crashing.
- **end-to-end check:** run `collect` on `examples/add-mul-relu-e2e` input; verify all 6 stages pass `afir-opt` and the structural graph populates badges from `auto_fuse.*` attrs.

## 8. Risks

1. **Stage split equivalence** (§4) — mitigated by per-stage `afir-opt` verification; merge stages if state-dependent.
2. **Pass flag spellings** — `create*Pass()` → CLI flag names resolved against registered pass arguments during implementation.
3. **Badge attr coverage** — `develop`'s `auto_fuse.tiling_infos` is the v2 single source of truth; some dev-nyh badge fields (e.g. `tail_policies`, `runtime_top_k`) may have no exact `develop` equivalent and will render blank. Acceptable for MVP.
