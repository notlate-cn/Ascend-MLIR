# Example Pipeline Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one unified test entry under `test/` that runs the six example end-to-end pipelines and can be invoked with the rest of the AFIR test suite.

**Architecture:** Add a single shell harness under `test/tools/examples/` that sources `examples/env.sh`, runs each example `run.sh --log`, and enforces PASS semantics. Hook that harness into lit with one `.mlir` driver file so the suite can execute it via `check-afir` without duplicating six shell snippets.

**Tech Stack:** lit, bash, existing example run scripts, existing `test/` lit configuration.

---

### Task 1: Add lit driver and harness

**Files:**
- Create: `test/tools/examples/run_example_pipelines.sh`
- Create: `test/tools/examples/example-pipelines.mlir`

- [ ] Step 1: Add one lit driver file that invokes the harness.
- [ ] Step 2: Implement the harness to run the six example pipelines, log per-example status, and fail if any pipeline misses `PASS` or exits nonzero.
- [ ] Step 3: Make the harness skip cleanly when required tools are missing.

### Task 2: Verify integration

**Files:**
- Modify: `test/lit.cfg.py` only if the new driver suffix/location needs config changes

- [ ] Step 1: Verify the new driver is discovered by lit under the existing config.
- [ ] Step 2: Run the harness directly for syntax/flow validation.
- [ ] Step 3: Run the lit test or equivalent command to prove integration.
