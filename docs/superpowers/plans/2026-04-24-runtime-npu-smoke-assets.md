# Runtime NPU Smoke Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fixed repository-owned assets for generating minimal `vec` and `mix` NPU smoke manifests plus command templates for the first real-device bring-up.

**Architecture:** Keep this as a small runtime-test helper, not a new product surface. Add a shell test first for manifest generation, then implement a tiny generator script backed by checked-in JSON templates, and finally wire the helper into the real-device validation docs.

**Tech Stack:** Bash, checked-in JSON templates, existing runtime verification scripts, Markdown docs

---

### Task 1: Add failing manifest-generation test

**Files:**
- Create: `test/tools/runtime/test_prepare_npu_smoke_manifests.sh`

- [ ] Add a shell test that expects a helper to generate valid `vec` and `mix` NPU smoke manifests from fixed arguments.
- [ ] Run the test and confirm it fails before the helper exists.

### Task 2: Implement manifest-generation helper

**Files:**
- Create: `test/tools/runtime/prepare_npu_smoke_manifests.sh`
- Create: `test/tools/runtime/npu_smoke/vec_manifest.template.json`
- Create: `test/tools/runtime/npu_smoke/mix_manifest.template.json`

- [ ] Implement the smallest helper that renders the two manifest kinds.
- [ ] Re-run the shell test and confirm it passes.

### Task 3: Keep helper under focused verification

**Files:**
- Modify: `test/tools/runtime/run_runtime.sh`

- [ ] Invoke the new shell test from focused runtime verification.
- [ ] Run `bash test/tools/runtime/run_runtime.sh` and confirm the helper test stays green.

### Task 4: Document exact bring-up commands

**Files:**
- Modify: `docs/runtime/NPU-REAL-DEVICE-VALIDATION.md`
- Modify: `docs/runtime/README.md`

- [ ] Add repository-owned `vec` / `mix` smoke asset descriptions.
- [ ] Add exact helper commands for generating the manifests and running `runtime-session`.
