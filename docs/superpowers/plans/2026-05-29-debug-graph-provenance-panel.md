# Debug Graph Provenance Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a selected-node provenance panel to the ascend-debug workbench so one Stage Graph node can be traced to its stage, kernel, graph edges, movement/memory metadata, and locate/tensor-diff status.

**Architecture:** Keep the current `summaries/debug_graph.json` schema and compute the first provenance view in browser-side JavaScript from existing `stage.graph`, `kernel_dag`, `overlay_details`, and `kernel_detail_views`. This makes the feature useful immediately without changing collect/open artifacts; later work can move richer provenance into a versioned JSON model.

**Tech Stack:** Python-generated static HTML, browser-side JavaScript, SVG graph state, `test/tools/diagnostics/test_ascend_debug_cli.sh`, xvm diagnostics.

---

### Task 1: Guard the UI Contract

**Files:**
- Modify: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [ ] Add grep checks requiring the generated workbench to contain `Provenance`, `function renderProvenanceSection`, `function directGraphContext`, `function kernelRuntimeStatus`, `function tensorDiffForKernel`, and `function renderKernelLineage`.
- [ ] Run `bash -n test/tools/diagnostics/test_ascend_debug_cli.sh`.
- [ ] Run the diagnostics once before implementation and confirm the new checks fail because the provenance functions do not exist yet.

### Task 2: Implement Selected-Node Provenance

**Files:**
- Modify: `tools/ascend-debug/ascend_debug/debug_graph.py`

- [ ] Add compact CSS for provenance cards and edge chips.
- [ ] Add JS helpers:
  - `kernelDagEntry`
  - `tensorDiffForKernel`
  - `kernelRuntimeStatus`
  - `directGraphContext`
  - `renderKernelLineage`
  - `renderDataflowProvenance`
  - `renderRuntimeProvenance`
  - `renderMovementProvenance`
  - `renderProvenanceSection`
- [ ] Extend `renderStageNodeDetail` or `selectStageNode` so every selected Stage Graph node shows a `Provenance` inspector section before the source excerpt.
- [ ] Keep output concise: direct in/out graph edges, kernel DAG context, locate/tensor-diff status, movement phases, and memory position only.

### Task 3: Verify

**Files:**
- No extra source files.

- [ ] Run local syntax and diff checks:
  - `python3 -m py_compile tools/ascend-debug/ascend_debug/debug_graph.py tools/ascend-debug/ascend_debug/stage_graph.py`
  - `bash -n test/tools/diagnostics/test_ascend_debug_cli.sh`
  - `git diff --check -- tools/ascend-debug/ascend_debug/debug_graph.py test/tools/diagnostics/test_ascend_debug_cli.sh docs/superpowers/plans/2026-05-29-debug-graph-provenance-panel.md`
- [ ] Run xvm diagnostics:
  - `bash test/tools/diagnostics/test_ascend_debug_cli.sh test/tools/diagnostics/ascend-debug-cli.mlir`
- [ ] Regenerate concat deep mode, restart the served workbench on port `18765`, and verify `stage=60&node=n40` shows the Provenance section with kernel/dataflow/runtime entries.
