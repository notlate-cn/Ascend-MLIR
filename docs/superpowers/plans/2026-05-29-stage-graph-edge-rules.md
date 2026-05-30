# Stage Graph Edge Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ascend-debug stage graphs use auditable edge rules for SSA, memory, resource, control, and terminal nodes so concat stage60 and later emit stages explain remaining disconnected nodes.

**Architecture:** Keep `stage_graph.py` as the current production graph source. Refactor resource handling into a small rule registry, add graph connectivity audit metadata, and expose the audit in `debug_graph.json` and the workbench inspector/sidebar without adding another standalone page.

**Tech Stack:** Python ascend-debug CLI, JSON graph artifacts, static/dynamic HTML from `debug_graph.py`, xvm diagnostics.

---

### Task 1: Add Regression Tests

**Files:**
- Modify: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [x] **Step 1: Add a Python parser test for emit/scf/audit**

Add a test MLIR snippet in the existing `stage_graph` Python block that includes:
- `emitasc.declare_py_struct`
- `emitasc.member`
- `emitasc.verbatim`
- `scf.if`
- a terminal `func.return`

Assert:
- graph has `connectivity` metadata;
- `emitasc.member` is classified as terminal read-like or connected by value/control, not suspicious isolated;
- `emitasc.declare_py_struct` is ignored as a declaration terminal;
- `scf.if` has a region/control edge or is not reported as a suspicious dangling side-effect;
- stage60-style resource tests still pass.

- [x] **Step 2: Add debug_graph summary assertions**

In the existing `debug_graph.json` assertions, require:
- `primary_stage["graph"]["connectivity"]` exists;
- at least one stage summary in `graph["stage_connectivity"]`;
- audit objects include `isolated_count`, `suspicious_isolated_count`, and `dangling_effect_count`.

- [x] **Step 3: Verify RED**

Run on xvm:

```bash
cd /home/niu/code/Ascend-MLIR
source examples/env.sh
export PATH=$PWD/build/bin:$PATH
bash test/tools/diagnostics/test_ascend_debug_cli.sh test/tools/diagnostics/ascend-debug-cli.mlir
```

Expected: fail because `connectivity` / `stage_connectivity` metadata does not exist yet.

### Task 2: Implement Rule Registry and Audit

**Files:**
- Modify: `tools/ascend-debug/ascend_debug/stage_graph.py`

- [x] **Step 1: Replace resource if-chain with rule registry**

Add small table-driven helpers:
- `RESOURCE_RULES`
- `TERMINAL_OP_RULES`
- `_classify_resource_access(...)`
- `_is_terminal_node(...)`
- `_compute_connectivity(...)`

Keep current stage60 behavior while making rule ownership explicit.

- [x] **Step 2: Add emit/scf rules**

Rules:
- `emitasc.declare_py_struct`: declaration terminal, allowed isolated.
- `emitasc.member`: reads first operand and writes result; if no result consumer, allowed value terminal.
- `emitasc.verbatim`: writes first operand and reads remaining operands when operands exist; resultless final writeback is terminal side-effect, not suspicious.
- `scf.if` / `scf.for`: region/control owner; region child edges already cover nested ops where parsed, otherwise classify no-output structure op as allowed terminal when it has inputs.
- `func.return`: terminal, never suspicious merely for no output.

- [x] **Step 3: Add connectivity audit to every graph**

`_finalize_graph` computes:
- `component_count`
- `isolated_count`
- `suspicious_isolated_nodes`
- `dangling_effect_nodes`
- `allowed_terminal_count`
- `edge_kind_counts`

### Task 3: Surface Audit in Workbench

**Files:**
- Modify: `tools/ascend-debug/ascend_debug/debug_graph.py`

- [x] **Step 1: Add stage connectivity summary**

Include `stage_connectivity` in `summaries/debug_graph.json`, one entry per stage with stage order/name and connectivity counts.

- [x] **Step 2: Render concise audit panel**

Add a compact panel near Stage Diff or inspector:
- total suspicious isolated;
- dangling effect nodes;
- edge kind counts.

Do not add a separate HTML entry point.

### Task 4: Verify with concat

**Files:**
- No source file changes expected.

- [x] **Step 1: Run diagnostics**

Run:

```bash
cd /home/niu/code/Ascend-MLIR
source examples/env.sh
export PATH=$PWD/build/bin:$PATH
bash test/tools/diagnostics/test_ascend_debug_cli.sh test/tools/diagnostics/ascend-debug-cli.mlir
```

Expected: `ALL ASCEND DEBUG CLI TESTS PASSED`.

- [x] **Step 2: Regenerate concat deep view**

Run:

```bash
rm -rf /tmp/ascend-debug-add-broadcast-concat-semantic-view
ascend-debug collect examples/add-broadcast-concat/step0_input.mlir --out /tmp/ascend-debug-add-broadcast-concat-semantic-view --mode deep
ascend-debug open /tmp/ascend-debug-add-broadcast-concat-semantic-view --no-browser
```

Expected:
- stage60 has zero suspicious isolated nodes;
- stage80/90 remaining disconnected items are either reduced or listed as allowed terminals;
- workbench JSON exposes the audit.

- [x] **Step 3: Browser verify**

Open:

```text
http://127.0.0.1:18765/views/debug_graph.html?stage=60
http://127.0.0.1:18765/views/debug_graph.html?stage=80
http://127.0.0.1:18765/views/debug_graph.html?stage=90
```

Expected: page renders resource/control paths and audit counts match JSON.
