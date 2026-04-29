# Runtime Legacy Cleanup Reclassification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the legacy cleanup candidate audit so it reflects the current post-cutover runtime state after the `Legacy/Compiler`, `Legacy/SimValidator`, and `Legacy/Executor` direct seams were removed from the main runtime path.

**Architecture:** This is a documentation-only planning round. Update the cleanup candidate buckets to match current source truth and then sync `AGENTS.md` so the next implementation round starts from correct cleanup priorities.

**Tech Stack:** markdown audits/specs/plans, source inventory cross-check, `AGENTS.md`

---

## File Map

**Modify:**
- `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`
- `AGENTS.md`

**Reference Only:**
- `docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md`
- `docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md`
- `docs/superpowers/specs/2026-04-14-runtime-legacy-cleanup-reclassification-design.md`

**Verification:**
- `rg -n "ArtifactCompiler|Legacy/Compiler|Legacy/Executor|Legacy/SimValidator|DefaultExecutionRunner|VecCubeArtifactBackend" docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md AGENTS.md`
- `git diff --check`

---

### Task 1: Refresh Cleanup Candidate Bucket Definitions

**Files:**
- Modify: `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`

- [ ] **Step 1: Reclassify `Legacy/Compiler`**

Update the audit so it no longer claims `ArtifactCompiler` is still a direct blocker.

The new classification should reflect:
- `ArtifactCompiler` no longer depends on `Legacy/Compiler`
- the remaining question is whether any non-`ArtifactCompiler` consumers still justify more shrink work

- [ ] **Step 2: Reclassify `Legacy/Executor`**

Update the audit so it no longer treats `SimBackend` / `NpuBackend` as direct `Legacy/Executor` consumers.

The new classification should reflect:
- the runtime backend seam is now `DefaultExecutionRunner`
- future cleanup work must start from the adapter boundary

- [ ] **Step 3: Reclassify `Legacy/SimValidator`**

Update the audit so it no longer cites the runtime backends as the direct blocker.

Its new bucket should be based only on remaining real consumers after the comparator extraction.

- [ ] **Step 4: Keep `Legacy/CompatRuntime` and `Legacy/HostRunnerGen` honest**

Do not mechanically move them. Reclassify only if the current evidence justifies it.

- [ ] **Step 5: Commit the cleanup candidate refresh**

```bash
git add docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md
git commit -m "docs: refresh legacy cleanup candidates"
```

### Task 2: Refresh Verification Requirements And Next-Step Guidance

**Files:**
- Modify: `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`

- [ ] **Step 1: Update verification requirements**

Make sure verification gates match the new seams:
- runtime backend changes now verify around `DefaultExecutionRunner`
- compile-path changes now verify around `VecCubeArtifactBackend`
- no stale wording about direct backend dependency on `Legacy/Executor` or `Legacy/SimValidator`

- [ ] **Step 2: Add precise next-step guidance**

Ensure the audit clearly supports the next planning round:
- remaining compiler-surface shrink
- adapter-boundary follow-up around execution
- eventual deeper legacy deletion only after those slices

- [ ] **Step 3: Commit the audit wording cleanup**

```bash
git add docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md
git commit -m "docs: tighten legacy cleanup verification guidance"
```

### Task 3: Sync AGENTS.md To The New Cleanup State

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Update Progress**

Reflect that:
- `ArtifactCompiler -> Legacy/Compiler` direct seam is gone
- `SimBackend/NpuBackend -> Legacy/Executor` direct seam is gone
- runtime backend validation no longer directly depends on `Legacy/SimValidator`

- [ ] **Step 2: Update TODO**

Make the next TODOs coherent with the refreshed cleanup candidate audit.

Expected direction:
- continue shrinking remaining compiler surface
- decide what actually remains of `Legacy/Compiler`
- plan the next deletion slices from the updated buckets

- [ ] **Step 3: Commit the AGENTS update**

```bash
git add AGENTS.md
git commit -m "docs: sync agents after cleanup reclassification"
```

### Task 4: Final Consistency Check

**Files:**
- No new files expected

- [ ] **Step 1: Run document consistency checks**

Run:

```bash
git diff --check
rg -n "ArtifactCompiler|Legacy/Compiler|Legacy/Executor|Legacy/SimValidator|DefaultExecutionRunner|VecCubeArtifactBackend" \
  docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md \
  AGENTS.md
```

- [ ] **Step 2: Confirm the new cleanup picture is internally consistent**

Verify the resulting documents no longer contradict current runtime state:
- no stale direct-seam claims
- no mismatched bucket logic
- no TODOs that refer to already-completed seam removals

