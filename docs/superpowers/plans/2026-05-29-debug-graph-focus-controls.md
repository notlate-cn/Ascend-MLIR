# Debug Graph Focus Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add richer Stage Graph navigation controls for transitive highlight modes, focus subgraphs, edge filters, path summaries, depth limits, URL persistence, and helper-node folding.

**Architecture:** Keep the existing static HTML/JavaScript workbench in `debug_graph.py`. Add stateful controls in the Stage Graph header, reuse the current `graph.edges` data to compute upstream/downstream closures, and keep generated graph JSON unchanged.

**Tech Stack:** Python-generated static HTML, browser-side JavaScript, SVG classes, existing `test/tools/diagnostics/test_ascend_debug_cli.sh` grep and JSON checks, xvm diagnostics.

---

### Task 1: Guard the UI Contract

**Files:**
- Modify: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [ ] Add failing grep checks for:
  - `id="highlight-mode-controls"`
  - `data-highlight-mode="upstream"`
  - `id="focus-toggle"`
  - `id="highlight-depth"`
  - `data-edge-kind-filter="resource_effect"`
  - `id="fold-helper-toggle"`
  - `function renderPathSummary`
  - `function updateGraphUrlState`
  - `function applyEdgeAndHelperFilters`

- [ ] Run `bash -n test/tools/diagnostics/test_ascend_debug_cli.sh`.

### Task 2: Implement Browser-Side Graph Controls

**Files:**
- Modify: `tools/ascend-debug/ascend_debug/debug_graph.py`

- [ ] Add controls to the Stage Graph panel header.
- [ ] Add CSS for segmented controls, focus-hidden nodes/edges, edge-filtered edges, and helper-collapsed nodes.
- [ ] Extend neighborhood traversal with direction mode and depth limit.
- [ ] Add edge-kind filters and helper-node folding.
- [ ] Add path summary counts and op samples to the inspector.
- [ ] Persist `stage`, `node`, `focus`, `depth`, `view`, `edges`, and `fold` in the URL.

### Task 3: Verify

**Files:**
- No additional source files.

- [ ] Run local syntax and diff checks:
  - `python3 -m py_compile tools/ascend-debug/ascend_debug/debug_graph.py tools/ascend-debug/ascend_debug/stage_graph.py`
  - `bash -n test/tools/diagnostics/test_ascend_debug_cli.sh`
  - `git diff --check -- tools/ascend-debug/ascend_debug/debug_graph.py test/tools/diagnostics/test_ascend_debug_cli.sh docs/superpowers/plans/2026-05-29-debug-graph-focus-controls.md`
- [ ] Sync changed files to xvm, rebuild `ascend-debug`, and run:
  - `bash test/tools/diagnostics/test_ascend_debug_cli.sh test/tools/diagnostics/ascend-debug-cli.mlir`
- [ ] Regenerate concat deep view and verify the active browser page exposes the new controls and behavior.
