# Mix Direct Source Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make mix compilation build its runtime contract directly from source/analyzer semantics and skip the expensive preprocess/probe/extract/update host-stub chain on the default path.

**Architecture:** Introduce an explicit mix compile contract source model. The binary build consumes only contract fields, while contract construction can use either `direct-source` or retained `legacy-preprocess` fallback. Direct mode uses `MixSourceAnalyzer` definitions and `MixStubTemplate` output instead of CANN host-stub extraction.

**Tech Stack:** C++17, LLVM support APIs, AscendC `bisheng`/`ld.lld`, existing runtime xvm shell verification.

---

### Task 1: Contract Source Semantics

**Files:**
- Modify: `lib/Runtime/Mix/MixDirectCompileInternal.h`
- Modify: `lib/Runtime/Mix/MixDirectPreprocess.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] Add a focused unit test that constructs a direct-source contract and checks the source path, runtime kernel name, AIC/AIV definitions, and generated stub/header paths.
- [ ] Run `cmake --build /home/niu/code/Codex-Ascend-MLIR/build --target test_taskgraph_runtime -j8` on xvm and confirm the new test fails because direct-source contract construction is missing.
- [ ] Add explicit contract source metadata and a direct-source builder seam.
- [ ] Run the focused test and confirm it passes.
- [ ] Commit as `runtime: add mix direct-source contract`.

### Task 2: Direct Stub/Header Generation

**Files:**
- Modify: `lib/Runtime/Mix/MixDirectPreprocess.cpp`
- Modify: `lib/Runtime/Mix/MixDirectBinaryBuild.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] Add a test proving direct-source mode writes `host_stub.cpp` and `aclrtlaunch_<kernel>.h` through `MixStubTemplate`.
- [ ] Run the focused test and confirm it fails before implementation.
- [ ] Implement direct stub/header generation with `HAVE_WORKSPACE` and `HAVE_TILING` definitions added explicitly.
- [ ] Run the focused test and confirm it passes.
- [ ] Commit as `runtime: generate mix direct host stub`.

### Task 3: Default Auto Mode

**Files:**
- Modify: `lib/Runtime/Mix/MixDirectCompilePipeline.cpp`
- Modify: `lib/Runtime/Mix/MixDirectPreprocess.cpp`
- Modify: `lib/Runtime/Mix/MixDirectBinaryBuild.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] Add a test that the default mix compile path records direct-source contract mode and does not require preprocess command/config fields.
- [ ] Run the focused test and confirm it fails on the current legacy-only path.
- [ ] Make the default contract mode `auto`: try direct-source first, retain legacy preprocess fallback for unsupported cases.
- [ ] Run focused tests and confirm they pass.
- [ ] Commit as `runtime: default mix contract to direct source`.

### Task 4: xvm Functional And Timing Verification

**Files:**
- Modify: `test/tools/runtime/run_runtime.sh` only if a new focused comparison helper is needed.

- [ ] Sync changed files to `/home/niu/code/Codex-Ascend-MLIR` on xvm.
- [ ] Build runtime targets with `-j8`.
- [ ] Run focused mix direct/legacy compile comparison.
- [ ] Run `bash test/tools/runtime/run_runtime.sh`.
- [ ] Record old/new compile timing from `compile_timing.json`.
- [ ] Commit any test helper changes as `test: compare mix direct source compile`.

### Task 5: Cleanup And Fallback Documentation

**Files:**
- Modify: `docs/superpowers/specs` or runtime docs only if fallback behavior needs a durable note.
- Modify: `AGENTS.md` only if explicitly requested.

- [ ] Document direct-source default and legacy fallback trigger.
- [ ] Confirm `AGENTS.md` remains excluded unless explicitly requested.
- [ ] Run final focused xvm verification.
- [ ] Commit documentation/test-only cleanup if needed.

