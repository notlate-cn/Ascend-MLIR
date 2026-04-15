# Runtime Complexity Entrypoint Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit runtime mix complexity visible at entry points and document timing diagnostic tools.

**Architecture:** Documentation-only change. Keep behavior unchanged; record current external surfaces and usage rules for diagnostic scripts.

**Tech Stack:** Markdown docs, runtime C++ entry points, shell/Python diagnostic tools.

---

### Task 1: Audit Runtime Entrypoint Complexity

**Files:**
- Create: `docs/superpowers/audits/2026-04-15-runtime-entrypoint-complexity-audit.md`

- [ ] **Step 1: Record entrypoint findings**

Document:

```markdown
# Runtime Entrypoint Complexity Audit

## Scope

- Object: `runtime-session`, C API, `autotuner`, and execution backends.
- Semantics: callers should consume runtime artifact contracts, not mix internal stages.
- Result: current visible mix-specific surfaces are classified as acceptable, watch, or follow-up.

## Findings

| Object | Current Surface | Classification | Result |
| --- | --- | --- | --- |
| `runtime-session` | parses `--kernel-kind=mix`; sets sibling `mix-tiling-helper` env path | watch | CLI still has one helper-discovery detail |
| C API | selects compiled artifact path; materializes user tiling bytes | acceptable | no mix staging dependency |
| `autotuner` | accepts `kernel-kind=mix`; packs tiling candidates | acceptable | no direct mix backend internals |
| `RuntimeSessionRequestBuilder` | loads mix metadata and applies `blockDim` / `tiling.bin` defaults | acceptable | artifact contract boundary |
| `SimBackend` / `NpuBackend` | branch on mix artifact and call packed mix runner | watch | execution still exposes packed mix launch seam |
| `ExecutionRunner` | has `runPackedMixFile` API | follow-up | mix-specific runner seam remains visible |

## Decision

- Keep `RuntimeSessionRequestBuilder` as the boundary for mix metadata to invocation defaults.
- Keep `ArtifactCompiler` as the boundary for mix vs vec/cube compile dispatch.
- Do not move mix staging into CLI, C API, autotuner, or execution summary code.
- Defer `ExecutionRunner::runPackedMixFile` cleanup until packed mix launch can be represented as a generic artifact launch.

## Next Work

- Move `runtime-session` helper discovery behind runtime support if more helper tools appear.
- Design a generic dynamic-library artifact launch interface before changing `ExecutionRunner`.
- Keep timing-stage names internal to diagnostics; callers should not parse them for behavior.
```

- [ ] **Step 2: Verify no placeholders**

Run:

```bash
rg -n "[T]BD|[T]ODO|fill[ ]in" docs/superpowers/audits/2026-04-15-runtime-entrypoint-complexity-audit.md
```

Expected: no matches.

### Task 2: Add Runtime Timing Tool README

**Files:**
- Create: `test/tools/runtime/README.md`

- [ ] **Step 1: Document diagnostic timing tools**

Document:

```markdown
# Runtime Test And Diagnostic Tools

## Default Verification

- Object: `run_runtime.sh`.
- Semantics: focused runtime verification on xvm.
- Result: keeps runtime core, C API, and repeated mix simulation baseline green.

## Mix Timing Diagnostics

| Tool | Semantics | Default Verification |
| --- | --- | --- |
| `run_mix_compile_timing_compare.sh` | Ascend-MLIR direct-source vs legacy-preprocess compile-only timing | no |
| `run_mix_pypto_timing_compare.sh` | Ascend-MLIR timing plus PyPTO frontend/public timing comparison | no |
| `pypto_mix_public_breakdown.py` | PyPTO public call breakdown into Python wrapper, compile, and `_run_with_cpu` | no |

## xvm Commands

```bash
cd /home/niu/code/Codex-Ascend-MLIR
bash test/tools/runtime/run_runtime.sh

RUNS=3 bash test/tools/runtime/run_mix_compile_timing_compare.sh
RUNS=3 bash test/tools/runtime/run_mix_pypto_timing_compare.sh

cd /home/niu/code/pypto
PYTHONPATH=/home/niu/code/pypto/python \
  python3 /home/niu/code/Codex-Ascend-MLIR/test/tools/runtime/pypto_mix_public_breakdown.py --runs 3
```

## Interpretation Rules

- `run_mix_compile_timing_compare.sh` reports Ascend-MLIR compile-only timing.
- `pypto_frontend_compile_private` reports PyPTO private frontend/JIT compile boundary.
- `pypto_compile_and_sim_public` reports PyPTO public compile+SIM/cost-model/report timing.
- `pypto_mix_public_breakdown.py` explains why PyPTO public timing is dominated by `_run_with_cpu`.
```

- [ ] **Step 2: Verify no placeholders**

Run:

```bash
rg -n "[T]BD|[T]ODO|fill[ ]in" test/tools/runtime/README.md
```

Expected: no matches.

### Task 3: Verify And Commit

**Files:**
- Verify: `docs/superpowers/audits/2026-04-15-runtime-entrypoint-complexity-audit.md`
- Verify: `test/tools/runtime/README.md`
- Verify: this plan file

- [ ] **Step 1: Run placeholder check**

```bash
rg -n "[T]BD|[T]ODO|fill[ ]in" \
  docs/superpowers/audits/2026-04-15-runtime-entrypoint-complexity-audit.md \
  test/tools/runtime/README.md \
  docs/superpowers/plans/2026-04-15-runtime-complexity-entrypoint-audit.md
```

Expected: no matches.

- [ ] **Step 2: Commit docs**

```bash
git add \
  docs/superpowers/audits/2026-04-15-runtime-entrypoint-complexity-audit.md \
  docs/superpowers/plans/2026-04-15-runtime-complexity-entrypoint-audit.md \
  test/tools/runtime/README.md
git commit -m "docs: audit runtime entrypoint complexity"
```
