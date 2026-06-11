# ascend-debug Workbench Phase 1c — Per-Kernel Drill-Down

**Date:** 2026-06-04
**Status:** Approved (design phase)
**Follow-up to:** Phase 1 (Kernel-DAG from network.json) + 1b (network lowering stages in ingest).

## 1. Goal

Clicking a Kernel-DAG node in the network workbench drills into that kernel's own sub-dashboard, showing the kernel's full post-outline lowering (schedule → bufferize → realize → parallelize → finalize), the generated AscendC `.cpp`, and its tiling JSON.

This completes the picture: **network level** = the 4 lowering stages (recognized→outlined) + Kernel-DAG (Phase 1/1b); **per kernel** = the post-outline codegen stages + cpp + tiling (this phase).

Why per-kernel works where whole-network `collect` crashed: each `groups/kernel_group*.mlir` is a small single-function kernel — verified that `collect` runs all 6 stages cleanly on `kernel_group0.mlir`.

## 2. Architecture

`ingest` already builds the network run dir (`graphs/kernel_dag.summary.json`, network lowering stages, provenance). Add per-kernel drill-down:

For each kernel `kid` listed in `network.json["kernels"]`:
1. **Per-kernel lowering**: run the existing `collect.collect_run` on `<workdir>/groups/<kid>.mlir` with out = `<run_dir>/kernels/<kid>/`. Produces that kernel's `stages/000-source … 060-finalize` + manifest. If a kernel's IR is missing or `collect` fails on it, skip that kernel's drill-down (log it) but keep its DAG node — one bad kernel must not abort ingest.
2. **Generated code**: if `<workdir>/<kid>.cpp` exists, copy it to `<run_dir>/kernels/<kid>/codegen/<kid>.cpp` and register it as an extra stage `070-codegen` (text view; open_view's stage view renders any text file) in that kernel's manifest.
3. **Tiling**: copy `<kid>_best.json` and `<kid>_space.json` (whichever exist) to `<run_dir>/kernels/<kid>/tiling/`. Surface them as manifest `reports` entries (open renders the reports list) or as linked artifacts.
4. **Render**: run `open_view.open_run` on `<run_dir>/kernels/<kid>/` (with `no_browser=True`) so each sub-dashboard has its own `index.html`.

Main workbench is unchanged except the link redirect below.

## 3. Wiring (one minimal open_view edit)

The Kernel-DAG node "Kernel 详情" link resolves to `kernel_detail_views[kid]` (JS, `debug_graph.py` line ~1026), which open_view's `_kernel_detail_views(stages, kernel_summary)` populates with the default `views/kernels/<kid>.html`. `open` regenerates `views/kernels/*` each run, so a pre-written redirect would be clobbered.

**Edit:** in `open_view._kernel_detail_views`, when building the map, if `run_dir/kernels/<kid>/index.html` exists, set `kernel_detail_views[kid] = "kernels/<kid>/index.html"` instead of the default. (Requires passing `run_dir` into `_kernel_detail_views`, which is called from `render_index` where `run_dir` is in scope.) This is the only change to the ported dashboard code; it is additive (falls back to the existing default when no sub-dashboard exists).

## 4. Files

- Modify: `tools/ascend-debug/ascend_debug/ingest.py` — per-kernel collect/open orchestration + cpp/tiling copy. Imports `collect` and `open_view`.
- Modify: `tools/ascend-debug/ascend_debug/open_view.py` — `_kernel_detail_views` gains a `run_dir` param + the sub-dashboard redirect (and its one call site in `render_index`).
- `collect.py` stays **untouched**. After `collect_run` writes the per-kernel manifest (6 stages), ingest copies `<kid>.cpp` into `kernels/<kid>/stages/070-codegen.cpp`, then re-reads that manifest, rebuilds the `StageArtifact` list from it, appends `StageArtifact(order=70, name="codegen", path="stages/070-codegen.cpp", step="codegen")`, and re-writes via `layout.write_manifest`. (The structural-graph parser yields an empty graph for a non-MLIR `.cpp` — no crash; the stage's text view shows the generated code.)
- Test: `tests/tools/ascend-debug/test_perkernel_drilldown.py`.

## 5. Testing

Unit/integration (uses built `afir-opt`; two-elewise has 2 small kernels):
- `ingest examples/two-elewise-e2e/build_e2e --out /tmp/te-wb` then assert:
  - `/tmp/te-wb/kernels/kernel_group0/manifest.json` exists with stages including `010-normalize … 060-finalize` (+ `070-codegen` since `kernel_group0.cpp` exists).
  - `/tmp/te-wb/kernels/kernel_group0/codegen/kernel_group0.cpp` exists; tiling json copied.
  - `/tmp/te-wb/kernels/kernel_group0/index.html` exists (sub-dashboard rendered).
- After `open /tmp/te-wb`, assert the main workspace `kernel_detail_views["kernel_group0"]` == `"kernels/kernel_group0/index.html"` (the drill-down link), so clicking the node opens the sub-dashboard.
- Lit smoke (the single-function `ascend-debug-cli.mlir`) still passes; full pytest passes.

## 6. Risks / scope

- **Scale**: per-kernel collect is O(kernels). Fine for two-elewise (2) / small nets. For GPT-2 (208) this is heavy (208×~6 afir-opt invocations) — out of scope (GPT-2 dropped); a future flag could gate/limit drill-down depth.
- **A kernel that fails `collect`**: skip its drill-down (DAG node remains, no link), log it; never abort the whole ingest.
- **`.cpp` as a "stage"**: open_view's stage view renders text; a `.cpp` shows as a code/text panel. The structural graph for a `.cpp` will be empty/garbage — register codegen as a text-only stage (no graph), or accept an empty graph. Keep simple: register it; if the graph parser chokes on non-MLIR, guard by skipping graph build for non-`.mlir` stage paths.
