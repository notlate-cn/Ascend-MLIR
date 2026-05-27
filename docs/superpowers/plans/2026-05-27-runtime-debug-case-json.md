# Runtime Debug Case JSON Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a user-level artifact-mode `case.json` entry that prepares and runs existing runtime artifacts without requiring users or examples to hand-write `run_manifest.json`, tiling params, block dim, or workspace size.

**Architecture:** Put canonical parsing and conversion in `lib/Runtime/Artifact`, expose it from `runtime-session --case`, and keep `ascend-debug` as a wrapper that delegates to `runtime-session`. First slice supports an already-built artifact set; source MLIR compilation orchestration remains a follow-up slice.

**Tech Stack:** C++ runtime JSON parsing with LLVM Support, existing `ArtifactManifestPrepareRequest`, `runtime-session` CLI, Python `ascend-debug` wrapper, shell/lit and runtime unit tests.

---

### Task 1: Runtime Debug Case Parser

**Files:**
- Create: `include/Runtime/Artifact/DebugCase.h`
- Create: `lib/Runtime/Artifact/DebugCase.cpp`
- Modify: `lib/Runtime/CMakeLists.txt`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [x] Add a failing runtime unit test that writes a minimal artifact manifest and user `case.json`, then expects `emitRunManifestFromDebugCase` to produce a concrete `run_manifest.json`.
- [x] Implement `DebugCasePrepareRequest`, parse `artifact`, `backend`, `shape_args`, `inputs`, `expected_outputs`, and `validation`.
- [x] Infer missing tensor `shape` and `dtype` from `.npy` bindings.
- [x] Reject user-level fields that must remain generated: `tiling`, `block_dim`, and `workspace_size`.
- [x] Reuse `emitRunManifestFromArtifactManifest` so schedule selection and host tiling stay in the existing runtime path.

### Task 2: runtime-session CLI

**Files:**
- Modify: `tools/runtime-session/runtime_session_main.cpp`
- Test: `test/tools/runtime/run_runtime.sh`

- [x] Add `--case <case.json>` and allow it with `--emit-run-manifest` or `--run`.
- [x] Reject ambiguous combinations with `--run-manifest`, direct `--artifact-manifest`, `--artifact-root`, or `--kernel`.
- [x] When running directly from a case, prepare a temporary run manifest, then execute the existing run-manifest path.

### Task 3: ascend-debug Wrapper

**Files:**
- Modify: `tools/ascend-debug/ascend-debug.py`
- Create: `tools/ascend-debug/ascend_debug/run_case.py`
- Test: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [x] Add `ascend-debug run case.json --out <dir>`.
- [x] Delegate to `runtime-session --case ... --emit-run-manifest <out>/run_manifest.json`, then `runtime-session --run-manifest <out>/run_manifest.json --run`.
- [x] Keep debug wrapper behavior thin; do not duplicate runtime case parsing in Python.
