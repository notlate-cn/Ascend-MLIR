# Autotuner And Legacy Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a concrete dependency audit for `autotuner` and `lib/Runtime/Legacy`, then derive a safe second-round cleanup sequence without mixing migration and deletion.

**Architecture:** Treat this work as documentation-backed runtime archaeology, not feature implementation. First build a verified dependency map, then isolate `autotuner`-specific blockers, then derive a cleanup candidate list grouped by prerequisite and risk.

**Tech Stack:** Markdown audit artifacts, `rg`, LLVM/C++ source inspection, xvm-focused runtime verification commands for later follow-up

---

## File Map

**Create:**
- `docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md` — concrete reference inventory for `Legacy/*` classes and who still depends on them
- `docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md` — focused `autotuner` audit describing what is already runtime-native vs. what still blocks cleanup
- `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md` — classified cleanup candidate list with prerequisites and verification notes

**Modify:**
- `AGENTS.md` — update `Progress` / `TODO` after the audit artifacts exist so the repo status points at the concrete audit outputs

**Reference Only:**
- `docs/superpowers/specs/2026-04-13-autotuner-legacy-cleanup-design.md`
- `tools/autotuner/autotuner_main.cpp`
- `lib/Runtime/Legacy/*`
- `lib/Runtime/Artifact/ArtifactCompiler.cpp`
- `lib/Runtime/Execution/SimBackend.cpp`
- `lib/Runtime/Execution/NpuBackend.cpp`
- `lib/CAPI/Runtime/Runtime.cpp`
- `test/tools/runtime/run_runtime.sh`

**Verification:**
- `rg -n "Compiler|Executor|HostRunnerGen|SimValidator|CompatRuntime|autotuner" -S tools lib include test`
- `bash test/tools/runtime/run_runtime.sh` is **not** required in this audit-only plan unless a task accidentally changes behavior

---

### Task 1: Produce The Legacy Dependency Audit

**Files:**
- Create: `docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md`
- Reference: `lib/Runtime/Legacy/*`, `lib/Runtime/Artifact/ArtifactCompiler.cpp`, `lib/Runtime/Execution/SimBackend.cpp`, `lib/Runtime/Execution/NpuBackend.cpp`, `lib/CAPI/Runtime/Runtime.cpp`, `tools/autotuner/autotuner_main.cpp`

- [ ] **Step 1: Write the failing audit outline first**

Create the audit file with empty section headers only, so there is a concrete target structure before collecting references:

```md
# Runtime Legacy Dependency Audit

## Scope

## Direct References

### Legacy/Compiler

### Legacy/Executor

### Legacy/SimValidator

### Legacy/HostRunnerGen

### Legacy/CompatRuntime

## By Consumer

## Bucket Classification

## Open Questions
```

- [ ] **Step 2: Run the raw reference inventory command**

Run:

```bash
rg -n "Compiler|Executor|HostRunnerGen|SimValidator|CompatRuntime|autotuner" -S tools lib include test
```

Expected:
- concrete file/line hits covering `Legacy/*` headers, runtime execution code, autotuner, tests, and C API

- [ ] **Step 3: Fill the direct-reference sections with concrete citations**

Populate the audit with a table per legacy unit. Use the actual paths discovered in Step 2. Format each table like this:

```md
### Legacy/Executor

| Consumer | File | Dependency Kind | Notes |
|---|---|---|---|
| SimBackend | `lib/Runtime/Execution/SimBackend.cpp` | implementation dependency | Used to drive simulator execution path |
| NpuBackend | `lib/Runtime/Execution/NpuBackend.cpp` | implementation dependency | Used for real-device launch path |
| test_runtime | `test/tools/runtime/test_runtime.cpp` | direct legacy test coverage | Still tests packed mix executor errors |
```
```

Do this for:
- `Legacy/Compiler`
- `Legacy/Executor`
- `Legacy/SimValidator`
- `Legacy/HostRunnerGen`
- `Legacy/CompatRuntime`

- [ ] **Step 4: Classify each legacy unit into buckets A/B/C**

In `## Bucket Classification`, add a concrete section like:

```md
## Bucket Classification

### Bucket A: Must Keep For Now
- `Legacy/Compiler`
  - required by `lib/Runtime/Artifact/ArtifactCompiler.cpp`
- `Legacy/Executor`
  - required by `lib/Runtime/Execution/SimBackend.cpp`
  - required by `lib/Runtime/Execution/NpuBackend.cpp`

### Bucket B: Candidate For Boundary Shrink
- `Legacy/HostRunnerGen`
  - still has direct tests and specialized tooling references

### Bucket C: Delete After Migration
- none yet
```
```

Use only classifications supported by file references from the audit.

- [ ] **Step 5: Self-check the audit for unsupported claims**

Run:

```bash
rg -n "Bucket A|Bucket B|Bucket C|Legacy/Compiler|Legacy/Executor|Legacy/SimValidator|Legacy/HostRunnerGen|Legacy/CompatRuntime" docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md
```

Expected:
- each bucket and each legacy unit appears in the audit file
- there are no placeholder sections left empty

- [ ] **Step 6: Commit the dependency audit**

```bash
git add docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md
git commit -m "docs: audit runtime legacy dependencies"
```

### Task 2: Audit Autotuner Against The New Runtime

**Files:**
- Create: `docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md`
- Reference: `tools/autotuner/autotuner_main.cpp`, `lib/Runtime/Artifact/ArtifactCompiler.cpp`, `lib/Runtime/Execution/SimBackend.cpp`, `lib/Runtime/Profile/*`

- [ ] **Step 1: Write the failing autotuner audit outline**

Create the file with this structure:

```md
# Autotuner Runtime Normalization Audit

## Scope

## Current Runtime-Native Pieces

## Remaining Legacy-Coupled Seams

## Does Autotuner Still Block Legacy Cleanup?

## Follow-Up Tasks
```

- [ ] **Step 2: Inspect autotuner includes and major helpers**

Run:

```bash
sed -n '1,260p' tools/autotuner/autotuner_main.cpp
```

Expected:
- top-level includes
- compile/setup helpers
- runtime execution and scoring helpers visible in the first part of the file

Then run:

```bash
rg -n "ArtifactCompiler|ExecutionSession|ProfileTrace|Compiler|Executor|SimValidator|HostRunnerGen|msprof|perf-report" tools/autotuner/autotuner_main.cpp
```

Expected:
- a concrete list of runtime-native and legacy-looking references remaining in autotuner

- [ ] **Step 3: Record which parts are already runtime-native**

Populate `## Current Runtime-Native Pieces` with bullets like:

```md
- compile path uses `ArtifactCompiler`
- candidate execution uses `ExecutionSession`
- scoring consumes runtime-retained profiling artifacts
```

Only include claims you can support from the file.

- [ ] **Step 4: Record any remaining cleanup blockers**

Populate `## Remaining Legacy-Coupled Seams` and `## Does Autotuner Still Block Legacy Cleanup?` with concrete answers, for example:

```md
- `ArtifactCompiler` still depends on `Legacy/Compiler`, so autotuner indirectly depends on that legacy layer.
- autotuner no longer shells out to `msprof` or uses `HostRunnerGen` directly.
- autotuner does not currently block removal of `Legacy/HostRunnerGen`, but it still blocks removal of `Legacy/Compiler` indirectly through `ArtifactCompiler`.
```

Do not speculate beyond what the source and audit prove.

- [ ] **Step 5: Derive explicit follow-up tasks**

Populate `## Follow-Up Tasks` with flat bullets only, each in this form:

```md
- Narrow `ArtifactCompiler`'s dependency on `Legacy/Compiler` before attempting to remove `Legacy/Compiler`.
- Re-audit `test/tools/runtime/test_runtime.cpp` if `Legacy/HostRunnerGen` becomes removable.
```

Each item must name a file or subsystem.

- [ ] **Step 6: Commit the autotuner audit**

```bash
git add docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md
git commit -m "docs: audit autotuner runtime normalization"
```

### Task 3: Produce The Second-Round Cleanup Candidate List

**Files:**
- Create: `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`
- Reference: `docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md`, `docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md`

- [ ] **Step 1: Write the failing cleanup candidate outline**

Create the file with this structure:

```md
# Runtime Legacy Cleanup Candidates

## Preconditions

## Safe Now

## Needs Migration First

## Do Not Touch Yet

## Verification Requirements
```

- [ ] **Step 2: Translate the audit into cleanup candidates**

Populate the file using only conclusions already proven in Tasks 1 and 2. Use sections like:

```md
## Safe Now
- none

## Needs Migration First
- `Legacy/Compiler`
  - blocked by `ArtifactCompiler`
  - indirectly blocks `autotuner`
- `Legacy/Executor`
  - blocked by `SimBackend` and `NpuBackend`

## Do Not Touch Yet
- `Legacy/HostRunnerGen`
  - still has direct test coverage and specialized workflow references
```
```

- [ ] **Step 3: Add verification requirements per candidate class**

Populate `## Verification Requirements` with concrete bullets like:

```md
- Any change affecting `SimBackend` or `NpuBackend` must run `bash test/tools/runtime/run_runtime.sh` on xvm.
- Any change affecting autotuner execution or scoring must rerun the existing autotuner smoke command on xvm.
- Any proposed deletion of `Legacy/HostRunnerGen` must update or remove `test/tools/runner/test_runner_gen.cpp` intentionally.
```

- [ ] **Step 4: Sanity-check consistency across the three audit artifacts**

Run:

```bash
rg -n "Legacy/Compiler|Legacy/Executor|Legacy/SimValidator|Legacy/HostRunnerGen|Legacy/CompatRuntime" \
  docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md \
  docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md \
  docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md
```

Expected:
- all relevant legacy units are represented consistently across the audit set

- [ ] **Step 5: Commit the cleanup candidate list**

```bash
git add docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md
git commit -m "docs: define runtime legacy cleanup candidates"
```

### Task 4: Update Repo Status Tracking

**Files:**
- Modify: `AGENTS.md`
- Reference: `docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md`, `docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md`, `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`

- [ ] **Step 1: Add the new audit artifacts to Progress**

Update `AGENTS.md` so `Progress` includes bullets pointing to the three new audit files.

Use this exact style:

```md
- Legacy cleanup audit artifacts now exist:
  - `docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md`
  - `docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md`
  - `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`
```

- [ ] **Step 2: Narrow TODO to implementation work only**

Update `AGENTS.md` so `TODO` no longer says “produce audit artifacts.” Replace that with the next concrete implementation-facing items, for example:

```md
- Decide whether `autotuner` needs further normalization work before deleting any `Legacy` code.
- Execute the second-round `Legacy` cleanup in risk-ordered slices.
- Keep xvm focused runtime verification green while shrinking `Legacy` dependencies.
```

Keep the existing note about the untouched untracked planning file.

- [ ] **Step 3: Verify the status guide reflects the new audit state**

Run:

```bash
sed -n '1,240p' AGENTS.md
```

Expected:
- `Progress` references the audit artifacts
- `TODO` points to implementation follow-up rather than missing documentation work

- [ ] **Step 4: Commit the repo-status update**

```bash
git add AGENTS.md
git commit -m "docs: update agents guide after legacy audit"
```

## Self-Review

- Spec coverage:
  - dependency audit artifact: covered by Task 1
  - autotuner normalization audit: covered by Task 2
  - cleanup candidate list grouped by risk/prerequisite: covered by Task 3
  - repo state tracking update: covered by Task 4
- Placeholder scan:
  - no `TODO`, `TBD`, or undefined “follow existing pattern” style steps remain
- Type/term consistency:
  - bucket names are stable across all tasks: `Safe Now`, `Needs Migration First`, `Do Not Touch Yet` in the cleanup doc, and `Bucket A/B/C` in the dependency audit
  - file paths and legacy unit names match the current repo layout
