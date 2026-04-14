# Legacy SimValidator Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shrink the residual `Legacy/SimValidator` surface now that runtime backends already validate through the runtime-native output comparator, while keeping only justified retained helper logic.

**Architecture:** This is a residual-surface cleanup slice. Do not redesign execution backends. First identify actual consumers, then remove dead comparison-facing API, and only extract simulator-cycle helpers if a real non-backend consumer still needs them.

**Tech Stack:** C++ runtime library/tests, markdown audits, xvm focused runtime verification

---

## File Map

**Expected Modify:**
- `include/Runtime/Legacy/SimValidator.h`
- `lib/Runtime/Legacy/SimValidator.cpp`
- `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`
- `AGENTS.md`

**Possible Modify:**
- `test/tools/runtime/test_runtime.cpp`
- `test/tools/runtime/test_taskgraph_runtime.cpp`
- `include/Runtime/SimValidator.h`

**Reference Only:**
- `include/Runtime/Execution/OutputComparator.h`
- `lib/Runtime/Execution/OutputComparator.cpp`
- `docs/superpowers/specs/2026-04-14-legacy-simvalidator-cleanup-design.md`

**Verification:**
- `rg -n "Runtime/SimValidator.h|\\bSimValidator\\b|CompareOnly\\(|ValidateBinary\\(" include lib tools test --glob '!build*' --glob '!externals/**'`
- `bash test/tools/runtime/run_runtime.sh`
- `git diff --check`

---

### Task 1: Audit Residual `Legacy/SimValidator` Consumers

**Files:**
- No source edits required at the start

- [ ] **Step 1: Run the residual consumer inventory**

Use repository search to enumerate direct remaining references to:
- `Runtime/SimValidator.h`
- `SimValidator`
- `CompareOnly(`
- `ValidateBinary(`

Capture which sites are:
- explicit retained compatibility
- test-only residue
- future helper extraction candidates

- [ ] **Step 2: Decide whether `CompareOnly(...)` is still externally needed**

If it has no real consumers outside the file itself or explicit legacy-only tests, mark it as dead comparison-facing surface for removal in Task 2.

- [ ] **Step 3: Decide whether `ValidateBinary(...)` is still externally needed**

If it survives, document exactly why.

- [ ] **Step 4: Commit any audit-note additions if needed**

Only if a temporary audit note is needed to keep the workstream coherent; otherwise move directly to Task 2.

### Task 2: Remove Or Isolate Dead Comparison-Facing Surface

**Files:**
- Modify: `include/Runtime/Legacy/SimValidator.h`
- Modify: `lib/Runtime/Legacy/SimValidator.cpp`
- Possible modify: `test/tools/runtime/test_runtime.cpp`
- Possible modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Remove dead comparison-facing entry points**

If `CompareOnly(...)` is no longer justified, remove it or narrow it so it is no longer presented as a general-purpose validator API.

- [ ] **Step 2: Keep any unavoidable retained path explicit**

If `ValidateBinary(...)` or other residual helpers still have a real consumer, keep them but make the retained role obvious in comments and header shape.

- [ ] **Step 3: Adjust affected tests intentionally**

If any runtime-owned test still exercises removed API, either:
- migrate it to the runtime-native comparator path, or
- mark it as explicit retained legacy coverage

- [ ] **Step 4: Commit the surface shrink**

```bash
git add include/Runtime/Legacy/SimValidator.h lib/Runtime/Legacy/SimValidator.cpp \
  test/tools/runtime/test_runtime.cpp test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "refactor: shrink legacy simvalidator surface"
```

### Task 3: Decide Whether Cycle Parsing Deserves Extraction

**Files:**
- Modify: `include/Runtime/Legacy/SimValidator.h`
- Modify: `lib/Runtime/Legacy/SimValidator.cpp`
- Possible add/modify helper files only if a real consumer justifies it

- [ ] **Step 1: Check whether simulator cycle parsing is still consumed outside retained legacy code**

If the answer is no, keep it local and avoid extraction.

If the answer is yes, extract only the smallest helper needed for that consumer.

- [ ] **Step 2: Avoid speculative helper creation**

Do not introduce a new runtime-native cycle parser unless an actual caller needs it now.

- [ ] **Step 3: Commit the cycle-helper decision if it changes code**

```bash
git add include/Runtime/Legacy/SimValidator.h lib/Runtime/Legacy/SimValidator.cpp \
  include/Runtime/* lib/Runtime/*
git commit -m "refactor: narrow legacy simvalidator helpers"
```

### Task 4: Re-Sync Cleanup State And Verify

**Files:**
- Modify: `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update cleanup candidate audit**

Reflect the post-shrink `Legacy/SimValidator` reality:
- what is gone
- what remains
- what the next cleanup decision is

- [ ] **Step 2: Update `AGENTS.md`**

Move TODO state forward so it no longer treats `Legacy/SimValidator` as if the current slice had not happened.

- [ ] **Step 3: Run required verification**

Run:

```bash
rg -n "Runtime/SimValidator.h|\\bSimValidator\\b|CompareOnly\\(|ValidateBinary\\(" \
  include lib tools test --glob '!build*' --glob '!externals/**'
git diff --check
bash test/tools/runtime/run_runtime.sh
```

- [ ] **Step 4: Commit the documentation sync**

```bash
git add docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md AGENTS.md
git commit -m "docs: update simvalidator cleanup status"
```
