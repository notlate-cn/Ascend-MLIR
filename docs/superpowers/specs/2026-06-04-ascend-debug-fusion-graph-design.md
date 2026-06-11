# ascend-debug Fusion-Partition Graph (cytoscape) — Design

**Date:** 2026-06-04
**Status:** Approved (design phase)
**Supersedes** the dev-nyh SVG workbench as the **primary network entry view**. The dev-nyh `ascend-debug` collect/open + per-kernel sub-dashboards remain for per-kernel stage IR / cpp / tiling detail.

## 1. Operation intuition (the target UX)

1. Open → land on the **whole-network fusion-partition graph**.
2. The graph expresses the **key workflow concept: how ops are partitioned into fusion groups/kernels.** Each group is **one collapsible compound node** (default collapsed, labeled **kind + output shape**). **Single-click expands** it to reveal its internal op subgraph (which ops fused, how they connect).
3. A separate **drill** action (button in the right detail panel, or double-click) on a group → opens that kernel's ascend-debug **per-kernel sub-dashboard** (6 lowering stages + `070-codegen` cpp + tiling).
4. From the per-kernel sub-dashboard, a **"← 返回整网"** link returns to the fusion graph.
5. The **right detail panel** shows the selected group's fused-ops summary, source ops, and **tiling** (`*_best.json` / `*_space.json`).

## 2. Why cytoscape (platform decision)

develop's `python/tools/ascend_kernel_dag_viz/viewer.html` already uses **cytoscape 3.28 + dagre** (loaded from unpkg CDN). Cytoscape natively supports **compound (parent/child) nodes**; with the `expand-collapse` extension, "group → big node ⇄ expand to internal ops" is native. The dev-nyh `debug_graph.py` workbench is hand-rolled SVG (no graph library) — building collapse/expand there means large custom JS. So the fusion graph is built on **a cytoscape viewer**, adapted into the ascend-debug tool so it shares the run-dir + links to the per-kernel sub-dashboards.

Tradeoff accepted: cytoscape loads from CDN (needs internet). Vendoring the JS into the repo for offline use is a later follow-up.

## 3. Architecture — one unified run dir produced by `ingest`

`ascend-debug ingest <network_workdir> --out <run>` produces:
```
<run>/index.html                  ← NEW: cytoscape fusion-partition graph (entry)
<run>/fusion_graph.json           ← NEW: compound-node graph data (groups + child ops + edges + kind/shape/tiling/drill hrefs)
<run>/graphs/kernel_dag.summary.json   (kept; existing)
<run>/network.provenance.json          (kept)
<run>/kernels/<kid>/index.html    ← per-kernel sub-dashboard (existing) + "← 返回整网" link
<run>/kernels/<kid>/...           (6 stages + 070-codegen.cpp + tiling/, existing)
```
The existing dev-nyh `open_view` dashboard is still generated for each per-kernel sub-dashboard (that's where stages/IR/cpp live). At the network level, the NEW cytoscape `index.html` replaces the old overview/workbench as the entry. (The old `open <run>` network workbench may stay reachable but is no longer the default entry; ingest writes the cytoscape graph as `<run>/index.html`.)

## 4. Data: `fusion_graph.json` (new builder `ascend_debug/fusion_graph.py`)

`build_fusion_graph(network, provenance, workdir) -> dict` produces cytoscape elements:
- **Group (parent/compound) node** per kernel `kid`: `{data: {id: kid, type: "group", label: <fused_ops_summary>, kind: <kind>, shape: <results[0].shape>, dtype, drill: "kernels/<kid>/index.html", source_ops: [...], tiling: {best: {...}, space: {...}}}}`. Tiling read from `<workdir>/<kid>_best.json` / `<kid>_space.json` (whichever exist) and inlined into the node data (small).
- **Op (child) nodes** per group: parse `<workdir>/groups/<kid>.mlir` with the existing `stage_graph.parse_stage_mlir` to get that kernel's internal ops; emit each op as `{data: {id: "<kid>::<opnode_id>", parent: kid, type: "op", label: <op_name>, ...}}` and the intra-group edges.
- **Inter-group edges**: from network kernel args `from=="kernel"` → `{data: {source: <src kid>, target: <kid>}}` (the kernel DAG edges; reuse `network_dag` edge logic).

(Reuse `network_dag.build_kernel_dag_summary` for edges/labels; add child-op extraction via `stage_graph.parse_stage_mlir` per kernel IR.)

## 5. Viewer: `ascend_debug/fusion_viewer.html` (adapted from develop's cytoscape viewer)

- cytoscape + dagre + **cytoscape-expand-collapse** (CDN). Compound layout: groups as parents, ops as children.
- **Default: all groups collapsed** (show group node with `label / kind / shape`). Single-click a collapsed group → expand (cytoscape-expand-collapse `expand`); single-click expanded → collapse.
- **Right detail panel** (reuse the existing kv-table panel): on group select show summary, kind, shape, source_ops, and a **tiling** section (best/space). Include a **「打开 Kernel 详情 →」** button → `node.data('drill')` (the per-kernel sub-dashboard). (Drill is an explicit button so single-click stays "expand".)
- Loads `fusion_graph.json` (inlined into index.html so `file://` works, matching the existing viewer's file:// goal).

## 6. Per-kernel sub-dashboard: back link

Each `kernels/<kid>/index.html` gets a **"← 返回整网"** link to `../../index.html`. Implementation: ingest passes a flag/field so `open_view` renders a back link in the page header when present (minimal open_view edit: render an optional `manifest["parent_view"]` link), OR ingest post-injects a small banner into the generated index.html. Prefer the manifest-field approach (clean, no string surgery).

## 7. Testing (two-elewise first)

- Unit: `build_fusion_graph` on a synthetic network+provenance+fake kernel IR → assert group parent nodes carry kind/shape/drill/tiling, op child nodes have `parent==kid`, inter-group edges present.
- Integration: `ingest examples/two-elewise-e2e/build_e2e --out /tmp/te-wb` → `index.html` + `fusion_graph.json` exist; fusion_graph has 2 group nodes (kernel_group0 [add], kernel_group1 [mul]) each with child op(s) + tiling inlined; each `kernels/<kid>/index.html` contains a "返回整网" link to `../../index.html`. Open in browser, confirm: collapsed groups show kind+shape; single-click expands to ops; detail panel shows tiling; drill button opens the sub-dashboard; back link returns.
- Regression: existing pytest + lit still pass.

## 8. Scope / risks

- **Scale**: child-op extraction parses each kernel IR — fine for small nets (two-elewise). GPT-2 (208 kernels) heavy; out of scope here.
- **CDN dependency**: cytoscape/dagre/expand-collapse from unpkg; needs internet. Vendoring = follow-up.
- **expand-collapse extension API**: confirm the CDN package + init (`cy.expandCollapse({...})`) during implementation; if flaky, fall back to manual compound show/hide of children on click.
- This makes the cytoscape graph the entry; the dev-nyh network workbench is de-emphasized (still used per-kernel). No code deleted — additive.
