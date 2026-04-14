# Legacy Compiler Surface Shrink Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shrink the remaining runtime-owned surface of `Legacy/Compiler` now that `ArtifactCompiler` no longer depends on it, so the residual consumer set becomes explicit, smaller, and easier to classify.

**Architecture:** This is a surface-shrink slice, not a full compiler rewrite. Prefer runtime-native compile paths where possible, keep forwarding compatibility in place, and use the resulting consumer set to drive the next `Legacy/Compiler` cleanup decision.

**Tech Stack:** C++ runtime library/tests, markdown docs, xvm focused runtime verification

---

## File Map

**Expected Modify:**
- `test/tools/runtime/test_runtime.cpp`
- `AGENTS.md`
- `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`

**Possible Modify:**
- `include/CAPI/Runtime.h`
- `lib/CAPI/Runtime/Runtime.cpp`
- `test/tools/runtime/mix_stub_fixture.cpp`

**Reference Only:**
- `include/Runtime/Legacy/Compiler.h`
- `include/Runtime/Artifact/ArtifactCompiler.h`
- `include/Runtime/Artifact/VecCubeArtifactBackend.h`
- `docs/superpowers/specs/2026-04-14-legacy-compiler-surface-shrink-design.md`
- `docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md`
- `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`

**Verification:**
- `rg -n "Runtime/Compiler.h|\\bCompiler compiler\\(" include lib tools test --glob '!build*' --glob '!externals/**'`
- `bash test/tools/runtime/run_runtime.sh`
- `git diff --check`

---

### Task 1: Replace Runtime-Owned Test Dependence Where Practical

**Files:**
- Modify: `test/tools/runtime/test_runtime.cpp`
- Possible modify: `test/tools/runtime/mix_stub_fixture.cpp`

- [ ] **Step 1: Identify the exact legacy assertion the test still needs**

Before changing the test, determine whether the current `Compiler`-based test is proving:
- runtime-native mix artifact generation still works, or
- some uniquely legacy `Compiler` behavior

Do not preserve direct `Compiler` usage if the test intent can be expressed with runtime-native compile paths.

- [ ] **Step 2: Move the test to the runtime-native compile path**

Prefer:
- `ArtifactCompiler`
- existing mix compile request helpers

Keep the assertion focused on the same observable behavior:
- artifact path returned
- output exists on disk
- mix artifact semantics remain valid

- [ ] **Step 3: Isolate any unavoidable legacy-only test**

If some direct `Compiler` coverage is still truly required, rename or scope the test so it is clearly a retained legacy test instead of a generic runtime test.

- [ ] **Step 4: Commit the test-surface shrink**

```bash
git add test/tools/runtime/test_runtime.cpp test/tools/runtime/mix_stub_fixture.cpp
git commit -m "test: shrink legacy compiler runtime test surface"
```

### Task 2: Reposition Public Surface And Comments

**Files:**
- Possible modify: `include/CAPI/Runtime.h`
- Possible modify: `lib/CAPI/Runtime/Runtime.cpp`

- [ ] **Step 1: Remove stale `Compiler`-centric wording**

Update comments or nearby API wording so they no longer imply the C API still wraps an internal `Compiler` implementation.

The code path already routes through runtime-native compile request building; comments should match that fact.

- [ ] **Step 2: Keep compatibility without overselling it**

Do not remove compatibility names in this slice. Only ensure the surrounding descriptions position them as compatibility surface rather than preferred runtime architecture.

- [ ] **Step 3: Commit the wording cleanup**

```bash
git add include/CAPI/Runtime.h lib/CAPI/Runtime/Runtime.cpp
git commit -m "docs: reposition legacy compiler compatibility wording"
```

### Task 3: Re-Audit The Residual `Legacy/Compiler` Consumer Set

**Files:**
- Modify: `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Re-run the residual consumer inventory**

Use repository search to capture the post-shrink direct sites for:
- `Runtime/Compiler.h`
- `Compiler compiler(`

Classify each remaining site as one of:
- required legacy compatibility
- explicit legacy test
- future extraction target

- [ ] **Step 2: Update cleanup candidates**

Adjust the audit so `Legacy/Compiler` reflects the new smaller residual surface and the next likely action:
- keep as narrow retained legacy compatibility, or
- schedule one more extraction pass

- [ ] **Step 3: Sync AGENTS TODO**

Update `AGENTS.md` so the next TODO reflects the post-shrink state instead of the pre-shrink assumption.

- [ ] **Step 4: Commit the re-audit**

```bash
git add docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md AGENTS.md
git commit -m "docs: update legacy compiler residual surface"
```

### Task 4: Verification And Final Consistency Check

**Files:**
- No new files expected

- [ ] **Step 1: Run focused verification**

Run:

```bash
bash test/tools/runtime/run_runtime.sh
```

This is the required acceptance gate for the slice.

- [ ] **Step 2: Re-check the residual search surface**

Run:

```bash
rg -n "Runtime/Compiler.h|\\bCompiler compiler\\(" include lib tools test --glob '!build*' --glob '!externals/**'
git diff --check
```

Verify:
- no accidental new `Compiler`-centric sites appeared
- residual direct uses are intentional and documented
- no whitespace or patch hygiene issues remain

- [ ] **Step 3: Summarize the resulting cleanup position**

At the end of the slice, the repo state should make one thing obvious:
- whether `Legacy/Compiler` is now just a small retained compatibility unit
- or whether one more concrete extraction slice is still justified
