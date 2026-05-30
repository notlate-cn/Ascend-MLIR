# Ascend Debug Failure Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ascend-debug` produce an openable debug workspace when compile or runtime execution fails.

**Architecture:** Add a small failure artifact helper that writes `run_status.json`, records failed commands in `manifest.json`, and lets `open` render a failure-first summary. Keep command execution in `runner.py`; callers decide which stage/phase failed and write partial manifests before returning non-zero.

**Tech Stack:** Python stdlib, existing `ascend-debug` shell integration test, fake CLI tools for deterministic failure cases.

---

### Task 1: Red Tests

**Files:**
- Modify: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [ ] **Step 1: Add compile failure case**

Add a fake `ascend-mlir-opt` that exits non-zero for `--ascend-kernelize`. Verify `ascend-debug collect --mode quick` returns non-zero but leaves `manifest.json`, `run_status.json`, the successful source/normalize stages, the failed command record, and a renderable `index.html`.

- [ ] **Step 2: Add runtime failure case**

Add a fake `runtime-session` that succeeds during `--emit-run-manifest` and fails during `--run-manifest ... --run`. Verify `ascend-debug run` returns non-zero but leaves `manifest.json`, `run_status.json`, `run_manifest.json`, prepare/run logs, and a renderable `index.html`.

- [ ] **Step 3: Verify red**

Run `bash test/tools/diagnostics/test_ascend_debug_cli.sh test/tools/diagnostics/ascend-debug-cli.mlir`. Expected: failure because failed runs do not yet emit partial manifests/status.

### Task 2: Failure Artifact Contract

**Files:**
- Create: `tools/ascend-debug/ascend_debug/failure.py`
- Modify: `tools/ascend-debug/ascend_debug/runner.py`
- Modify: `tools/ascend-debug/ascend_debug/layout.py`

- [ ] **Step 1: Preserve command failure data**

Extend `CommandError` so `run_command` can attach argv, return code, stderr, stdout path, and stderr report path.

- [ ] **Step 2: Write status and partial manifests**

Add helpers to build failed command records, write `run_status.json`, and write a manifest with `status: failed`, `failed_stage`, `failed_phase`, and `failure_status`.

### Task 3: Collect and Run Integration

**Files:**
- Modify: `tools/ascend-debug/ascend_debug/collect.py`
- Modify: `tools/ascend-debug/ascend_debug/run_case.py`

- [ ] **Step 1: Wrap collect stage execution**

For each collect mode, catch command failures, record the failed command, include stages whose files already exist, and write failure artifacts before re-raising.

- [ ] **Step 2: Wrap runtime execution**

For artifact and source cases, record prepare/run/compile command successes and failures. On failure, write a runtime manifest with reports and logs produced so far.

### Task 4: Open View Failure Rendering

**Files:**
- Modify: `tools/ascend-debug/ascend_debug/open_view.py`
- Modify: `tools/ascend-debug/ascend_debug/debug_graph.py` if needed for empty-stage resilience

- [ ] **Step 1: Validate new manifest fields**

Allow `status`, `failure_status`, `failed_stage`, and `failed_phase` in manifests.

- [ ] **Step 2: Render failure summary**

Add a top-level failure panel with stage, phase, exit code, command, stdout/stderr links, and error message. Ensure report views keep readable foreground/background colors.

### Task 5: Verification

**Files:**
- Test: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [ ] **Step 1: Local checks**

Run Python compile checks, bash syntax check, `git diff --check`, and the diagnostic shell test locally.

- [ ] **Step 2: xvm checks**

Sync/build `ascend-debug` on xvm and run `bash test/tools/diagnostics/test_ascend_debug_cli.sh test/tools/diagnostics/ascend-debug-cli.mlir`.
