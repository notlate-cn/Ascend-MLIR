# Runtime NPU Pre-Device Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten the runtime-owned NPU contract, expand NPU-focused tests, and document the real-device bring-up boundary before hardware is available.

**Architecture:** Keep `NpuBackend` conservative. Add failing tests first for request-contract and driver-path behavior, then implement the smallest runtime fixes needed to satisfy them. Document the resulting baseline in an audit plus a real-device runbook.

**Tech Stack:** C++, LLVM support types, shell-based xvm verification, Markdown docs

---

### Task 1: Add failing NPU-focused tests

**Files:**
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] Add failing tests for runtime-owned NPU contract gaps:
  - driver-backed error propagation retains the driver message
  - expected-output bindings reject missing output metadata when no golden exists
  - README/runbook-facing assumptions that need code-level coverage stay explicit in test names

- [ ] Run the focused runtime test binary or `run_runtime.sh` to verify the new tests fail for the expected reason.

### Task 2: Tighten `NpuBackend`

**Files:**
- Modify: `lib/Runtime/Execution/NpuBackend.cpp`

- [ ] Implement only the missing request validation and stage attribution needed by Task 1.
- [ ] Keep the scheduler/driver capability model unchanged.
- [ ] Re-run the failing tests and verify they pass.

### Task 3: Write NPU audit and bring-up runbook

**Files:**
- Create: `docs/superpowers/audits/2026-04-24-runtime-npu-pre-device-audit.md`
- Create: `docs/runtime/NPU-REAL-DEVICE-VALIDATION.md`
- Modify: `docs/runtime/README.md`

- [ ] Record the current NPU baseline, validated surface, and remaining gaps.
- [ ] Write a concrete real-device bring-up checklist and command sequence.
- [ ] Cross-link the README to the audit/runbook.

### Task 4: Fresh xvm verification

**Files:**
- No code changes expected

- [ ] Run `bash test/tools/runtime/run_runtime.sh`
- [ ] Run `bash test/tools/runtime/run_simbackend_examples.sh`
- [ ] Run `bash test/tools/examples/example_pipelines.sh`
- [ ] Confirm counts and `RC=0` before claiming completion.
