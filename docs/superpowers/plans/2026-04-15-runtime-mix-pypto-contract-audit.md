# Runtime Mix PyPTO Contract Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture the PyPTO timing comparison semantics and audit current runtime mix complexity without changing runtime behavior.

**Architecture:** Add documentation-only artifacts. Keep PyPTO timing semantics separate from runtime contract semantics, then audit where mix-specific responsibilities live today.

**Tech Stack:** Markdown docs, existing runtime timing scripts, xvm verification outputs.

---

### Task 1: Document PyPTO Timing Semantics

**Files:**
- Create: `docs/superpowers/audits/2026-04-15-pypto-mix-timing-semantics.md`

- [ ] **Step 1: Write the timing semantics audit**

Add a Markdown document with these sections:

```markdown
# PyPTO Mix Timing Semantics

## Scope

- Object: PyPTO mix timing comparison on xvm.
- Semantics: distinguish frontend compile, public compile+SIM, and public-call breakdown.
- Result: prevent comparing PyPTO private frontend timing directly against Ascend-MLIR runtime artifact compile timing.

## Measured Cases

| Object | Semantics | xvm Result |
| --- | --- | --- |
| `pypto_frontend_compile_private` | private frontend/JIT compile boundary | about `130-150 ms` |
| `pypto_compile_and_sim_public` | public call: compile + SIM/cost-model/report | about `2.2-2.3 s` |
| `pypto public breakdown compile` | compile portion inside public call | about `90-170 ms` |
| `pypto public breakdown _run_with_cpu` | SIM/cost-model/report portion inside public call | about `2.0-2.2 s` |

## Interpretation

- PyPTO public `2.2 s` is dominated by `_run_with_cpu`.
- PyPTO private frontend timing is not a full artifact compile timing.
- Ascend-MLIR direct-source compile timing remains a runtime artifact compile timing.

## Reproduction

Run on xvm:

```bash
cd /home/niu/code/Codex-Ascend-MLIR
bash test/tools/runtime/run_mix_pypto_timing_compare.sh
cd /home/niu/code/pypto
PYTHONPATH=/home/niu/code/pypto/python \
  python3 /home/niu/code/Codex-Ascend-MLIR/test/tools/runtime/pypto_mix_public_breakdown.py --runs 3
```

## Decision

- Use `pypto_frontend_compile_private` only as a frontend-lightweight reference.
- Use `pypto_compile_and_sim_public` only as user-visible SIM invocation timing.
- Do not compare either directly against Ascend-MLIR compile-only timing without naming the scope.
```

- [ ] **Step 2: Verify the document has no placeholder wording**

Run:

```bash
rg -n "[T]BD|[T]ODO|fill[ ]in" docs/superpowers/audits/2026-04-15-pypto-mix-timing-semantics.md
```

Expected: no matches.

### Task 2: Document Mix Contract Boundaries

**Files:**
- Create: `docs/superpowers/audits/2026-04-15-runtime-mix-contract-boundaries.md`

- [ ] **Step 1: Write the contract boundary audit**

Add a Markdown document with these sections:

```markdown
# Runtime Mix Contract Boundaries

## Scope

- Object: runtime-native mix compile path.
- Semantics: direct-source is the default runtime artifact builder; legacy-preprocess is fallback.
- Result: callers consume compiled artifacts, not mix internals.

## Inputs

- Kernel source path and content.
- Kernel kind and entry name.
- SOC and Ascend/CANN toolchain path.
- Compile flags and include paths.
- Shape and tiling params.
- Runtime ABI and metadata schema.

## Outputs

- Device object/binary.
- `tiling.bin`.
- `blockDim`.
- Generated host stub.
- Runtime ABI metadata.
- Compile timing metadata.

## Current Boundary

- `ArtifactCompiler` dispatches `mix` to `MixDirectBackend`.
- `MixDirectBackend` owns direct-source and fallback-mode selection.
- `MixDirectCompilePipeline` owns staged compilation.
- `RuntimeFrontendCore`, `runtime-session`, C API, and execution backends should consume compiled artifact results only.

## Guardrails

- Do not leak mix preprocessing details into CLI, C API, or execution runners.
- Do not compare PyPTO frontend-only timing with full runtime artifact compile timing.
- Keep legacy-preprocess callable only as fallback while direct-source remains default.
```

- [ ] **Step 2: Verify the document has no placeholder wording**

Run:

```bash
rg -n "[T]BD|[T]ODO|fill[ ]in" docs/superpowers/audits/2026-04-15-runtime-mix-contract-boundaries.md
```

Expected: no matches.

### Task 3: Audit Runtime Complexity Surface

**Files:**
- Create: `docs/superpowers/audits/2026-04-15-runtime-complexity-surface.md`

- [ ] **Step 1: Write the complexity surface audit**

Add a Markdown document with these sections:

```markdown
# Runtime Complexity Surface

## Scope

- Object: `lib/Runtime` and `include/Runtime`.
- Semantics: runtime is now a platform layer, not a single runner.
- Result: complexity is acceptable only when hidden behind stable runtime-native contracts.

## Complexity Sources

- `Artifact`: compile request, manifest, artifact loading, vec/cube and mix backend dispatch.
- `Mix`: AscendC source analysis, direct-source compile, tiling artifact emission, host stub generation, legacy fallback.
- `Execution`: SIM/NPU runners, task graph, output comparison, profile retention.
- `Frontend`: CLI and C API request assembly through `RuntimeFrontendCore`.

## Current Risk

- Mix-specific behavior can leak upward if callers begin depending on compile stages or fallback internals.
- Timing comparisons can become misleading when frontend compile, artifact compile, and SIM invocation are mixed.
- More runtime features will increase maintenance cost unless each one lands behind an existing contract.

## Recommended Next Work

- Keep documenting contract inputs and outputs before performance changes.
- Keep all mix-specific staging under `Runtime/Mix` and `Runtime/Artifact`.
- Add new behavior through `RuntimeFrontendCore` only when both CLI and C API need it.
- Treat cache as a future optimization after contract boundaries are stable.
```

- [ ] **Step 2: Verify the document has no placeholder wording**

Run:

```bash
rg -n "[T]BD|[T]ODO|fill[ ]in" docs/superpowers/audits/2026-04-15-runtime-complexity-surface.md
```

Expected: no matches.

### Task 4: Verify And Commit

**Files:**
- Verify: `docs/superpowers/audits/2026-04-15-pypto-mix-timing-semantics.md`
- Verify: `docs/superpowers/audits/2026-04-15-runtime-mix-contract-boundaries.md`
- Verify: `docs/superpowers/audits/2026-04-15-runtime-complexity-surface.md`

- [ ] **Step 1: Run local Markdown placeholder check**

```bash
rg -n "[T]BD|[T]ODO|fill[ ]in" \
  docs/superpowers/audits/2026-04-15-pypto-mix-timing-semantics.md \
  docs/superpowers/audits/2026-04-15-runtime-mix-contract-boundaries.md \
  docs/superpowers/audits/2026-04-15-runtime-complexity-surface.md
```

Expected: no matches.

- [ ] **Step 2: Commit docs**

```bash
git add \
  docs/superpowers/audits/2026-04-15-pypto-mix-timing-semantics.md \
  docs/superpowers/audits/2026-04-15-runtime-mix-contract-boundaries.md \
  docs/superpowers/audits/2026-04-15-runtime-complexity-surface.md \
  docs/superpowers/plans/2026-04-15-runtime-mix-pypto-contract-audit.md
git commit -m "docs: capture runtime mix pypto timing audit"
```
