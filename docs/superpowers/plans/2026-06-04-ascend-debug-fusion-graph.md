# ascend-debug Fusion-Partition Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `ingest` produce a cytoscape fusion-partition graph as the network entry (`<run>/index.html`): groups as collapsible compound nodes (kind+shape), single-click expand to internal ops, drill button → per-kernel sub-dashboard, tiling in detail panel; and add a "← 返回整网" link in each per-kernel sub-dashboard.

**Architecture:** New `ascend_debug/fusion_graph.py` (data builder) + `ascend_debug/fusion_viewer.html` (cytoscape template). `ingest.py` builds the graph data, writes `<run>/index.html` (viewer with data inlined), and sets a `parent_view` back-link in per-kernel manifests. One minimal `open_view` edit renders that back-link. Built on cytoscape 3.28 + dagre + expand-collapse (CDN), matching develop's existing `ascend_kernel_dag_viz/viewer.html`.

**Spec:** `docs/superpowers/specs/2026-06-04-ascend-debug-fusion-graph-design.md`.

---

### Task 1: `fusion_graph.py` — cytoscape elements builder

**Files:** Create `tools/ascend-debug/ascend_debug/fusion_graph.py`; test `tests/tools/ascend-debug/test_fusion_graph.py`.

- [ ] **Step 1: Failing unit test** — `tests/tools/ascend-debug/test_fusion_graph.py`:
```python
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "ascend-debug"))
from ascend_debug import fusion_graph


def _setup(tmp_path):
    wd = tmp_path / "wd"
    (wd / "groups").mkdir(parents=True)
    (wd / "groups" / "kernel_group0.mlir").write_text(
        'func.func @kernel_group0(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {\n'
        '  %0 = linalg.generic ins(%a : tensor<4x4xf32>) outs(%a : tensor<4x4xf32>) {\n'
        '  ^bb0(%x: f32, %y: f32): linalg.yield %x : f32 } -> tensor<4x4xf32>\n'
        '  return %0 : tensor<4x4xf32>\n}\n'
    )
    (wd / "kernel_group0_best.json").write_text('{"XBLOCK": 8}')
    network = {"kernels": [
        {"id": "kernel_group0", "kind": "ascendc",
         "args": [{"from": "input", "name": "a"}],
         "results": [{"name": "r", "shape": [4, 4], "dtype": "f32"}]},
        {"id": "kernel_group1", "kind": "aclnn", "op": "MatMul",
         "args": [{"from": "kernel", "kernel": "kernel_group0", "result": 0}],
         "results": [{"name": "r2", "shape": [4, 4], "dtype": "f32"}]},
    ]}
    prov = {"kernels": [
        {"kernel_id": "kernel_group0", "fused_ops_summary": "add", "source_ops": [{"id": "op_000"}]},
        {"kernel_id": "kernel_group1", "fused_ops_summary": "matmul", "source_ops": [{"id": "op_001"}]},
    ]}
    return wd, network, prov


def test_group_nodes_and_children(tmp_path):
    wd, network, prov = _setup(tmp_path)
    g = fusion_graph.build_fusion_graph(network, prov, wd)
    nodes = {e["data"]["id"]: e["data"] for e in g["elements"] if "source" not in e["data"]}
    # group parent nodes
    assert nodes["kernel_group0"]["type"] == "group"
    assert nodes["kernel_group0"]["kind"] == "ascendc"
    assert nodes["kernel_group0"]["shape"] == [4, 4]
    assert nodes["kernel_group0"]["label"] == "add"
    assert nodes["kernel_group0"]["drill"] == "kernels/kernel_group0/index.html"
    assert nodes["kernel_group0"]["tiling"]["best"] == {"XBLOCK": 8}
    assert nodes["kernel_group1"]["label"] == "matmul"
    # child op nodes parented to their group (kernel_group0 had a kernel IR)
    children = [d for d in nodes.values() if d.get("parent") == "kernel_group0"]
    assert children and all(d["type"] == "op" for d in children)


def test_inter_group_edges(tmp_path):
    wd, network, prov = _setup(tmp_path)
    g = fusion_graph.build_fusion_graph(network, prov, wd)
    edges = [e["data"] for e in g["elements"] if "source" in e["data"]]
    inter = [e for e in edges if e["source"] == "kernel_group0" and e["target"] == "kernel_group1"]
    assert inter  # kernel_group1 consumes kernel_group0
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement `fusion_graph.py`.** Use `network_dag` for inter-group edges and `stage_graph.parse_stage_mlir` for per-kernel child ops:
```python
from __future__ import annotations

import json
import pathlib
from typing import Any

from ascend_debug import network_dag, stage_graph


def _read_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def build_fusion_graph(network: dict[str, Any], provenance: dict[str, Any] | None,
                       workdir: pathlib.Path) -> dict[str, Any]:
    summary = network_dag.build_kernel_dag_summary(network, provenance)
    elements: list[dict[str, Any]] = []

    for kid, node in summary["nodes"].items():
        tiling: dict[str, Any] = {}
        best = _read_json(workdir / f"{kid}_best.json")
        if best is not None:
            tiling["best"] = best
        space = _read_json(workdir / f"{kid}_space.json")
        if space is not None:
            tiling["space"] = space
        elements.append({"data": {
            "id": kid,
            "type": "group",
            "label": (node.get("ops") or [{}])[0].get("label") or kid,
            "kind": node.get("kind"),
            "shape": node.get("output_shape"),
            "dtype": node.get("output_dtype"),
            "drill": f"kernels/{kid}/index.html",
            "source_ops": node.get("source_ops", []),
            "tiling": tiling,
        }})

        # child ops from the kernel's outlined IR
        kernel_ir = workdir / "groups" / f"{kid}.mlir"
        if kernel_ir.exists():
            text = kernel_ir.read_text(encoding="utf-8", errors="replace")
            sub = stage_graph.parse_stage_mlir({"order": 0, "name": kid, "path": str(kernel_ir)}, text)
            id_map = {}
            for n in sub.get("nodes", []):
                cid = f"{kid}::{n['id']}"
                id_map[n["id"]] = cid
                elements.append({"data": {
                    "id": cid, "parent": kid, "type": "op",
                    "label": n.get("op_name") or n.get("label") or "op",
                    "op_label": n.get("label"),
                    "result_type": n.get("result_type"),
                }})
            for e in sub.get("edges", []):
                s, t = id_map.get(e.get("from")), id_map.get(e.get("to"))
                if s and t:
                    elements.append({"data": {"id": f"{cid}_e_{s}_{t}_{len(elements)}",
                                              "source": s, "target": t, "intra": True}})

    for e in summary["edges"]:
        elements.append({"data": {"id": f"g_{e['from']}_{e['to']}",
                                  "source": e["from"], "target": e["to"]}})

    return {"schema_version": 1, "tool": "ascend-debug",
            "kernel_count": summary["kernel_count"], "elements": elements}
```

- [ ] **Step 4: Run → pass** (`python3 -m pytest tests/tools/ascend-debug/test_fusion_graph.py -v`).
- [ ] **Step 5: Commit** `feat(ascend-debug): fusion_graph builder (cytoscape compound elements)` (trailer; develop).

---

### Task 2: viewer + ingest wiring + back-link + two-elewise

**Files:** Create `tools/ascend-debug/ascend_debug/fusion_viewer.html`; modify `ingest.py`, `open_view.py`. Test: extend integration.

- [ ] **Step 1: Create `fusion_viewer.html`.** READ `python/tools/ascend_kernel_dag_viz/viewer.html` first and use it as the base (cytoscape 3.28 + dagre + cytoscape-dagre via unpkg). Add:
  - `<script src="https://unpkg.com/cytoscape-expand-collapse@4.1.1/cytoscape-expand-collapse.js"></script>` (register the extension).
  - Read graph data from an inlined `<script type="application/json" id="fusion-data">…</script>` (ingest fills it) instead of `fetch('dag.json')` — so `file://` works.
  - Build cytoscape with the `elements`; style: `node[type="group"]` = big rounded box, label `${label}\n${kind} ${shape}`; `node[type="op"]` = small; compound parents auto-box children.
  - Init `cy.expandCollapse({ ... })`; **collapse all groups on load** (`api.collapseAll()`); **single-click** a group toggles expand/collapse (`tap` on `node[type="group"]` → `api.isCollapsible(node)?api.collapse(node):api.expand(node)`).
  - **Right detail panel** (reuse the kv-table from the base viewer): on select a group, show label, kind, shape, dtype, source_ops list, and a **tiling** section (best = key/value table; space = collapsible/`<pre>`). Add a button **「打开 Kernel 详情 →」** that does `window.location.href = node.data('drill')`.
  - Keep it self-contained for `file://` (data inlined; only the 4 JS libs come from CDN).
  Keep the file as a TEMPLATE with a placeholder token (e.g. `/*__FUSION_DATA__*/`) that ingest replaces with the JSON, OR an empty `<script id="fusion-data">{}</script>` that ingest rewrites.

- [ ] **Step 2: `ingest.py` — write the fusion graph as `<run>/index.html`.** After the per-kernel loop, add:
```python
    from ascend_debug import fusion_graph as _fusion
    fg = _fusion.build_fusion_graph(network, provenance, workdir)
    layout.write_json(run_dir / "fusion_graph.json", fg)
    template = (pathlib.Path(__file__).resolve().parent / "fusion_viewer.html").read_text(encoding="utf-8")
    html = template.replace('<script id="fusion-data" type="application/json">{}</script>',
                            f'<script id="fusion-data" type="application/json">{json.dumps(fg)}</script>')
    (run_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"ascend-debug.ingest.fusion_graph={run_dir / 'index.html'}")
```
(Match the exact placeholder string to whatever `fusion_viewer.html` uses.)

- [ ] **Step 3: Per-kernel back-link.** When ingest finishes a kernel sub-dashboard, set a back link. In the manifest re-write step (where the `070-codegen` stage is appended) OR right before `open_run`, write `parent_view="../../index.html"` into the kernel manifest. Then in `open_view.py`, where the index page header is built (`render_index` / the overview header), render an optional back link if `manifest.get("parent_view")`:
  ```python
  parent = manifest.get("parent_view")
  back_link = f'<a class="back-link" href="{_cell(parent)}">← 返回整网</a>' if parent else ""
  ```
  and place `back_link` in the header HTML. (Find the header assembly in `render_index`; add `back_link` near the title. Minimal additive edit.)
  `layout.write_manifest` has no `parent_view` kwarg — pass it via a manifest post-edit in ingest (read manifest.json, set `manifest["parent_view"]="../../index.html"`, write back) after `open_run`? No — open reads the manifest. So set parent_view in the manifest BEFORE `open_run`. Since ingest already re-writes the kernel manifest to add `070-codegen`, also stash `parent_view`: write it as an extra top-level key by post-editing the manifest JSON dict (read, set key, `layout.write_json(... )`) before calling `open_run`. Then `open_view.load_manifest`/`render_index` reads `manifest["parent_view"]`. (`load_manifest` must not reject unknown keys — verify it doesn't.)

- [ ] **Step 4: Integration on two-elewise.**
```bash
cd /home/gser/code/Ascend-MLIR
export PATH="$PWD/build/bin:$PATH"
rm -rf /tmp/te-wb
PYTHONPATH=tools/ascend-debug python3 tools/ascend-debug/ascend-debug.py ingest examples/two-elewise-e2e/build_e2e --out /tmp/te-wb
test -f /tmp/te-wb/index.html && echo INDEX_OK
test -f /tmp/te-wb/fusion_graph.json && echo FG_OK
python3 - <<'PY'
import json
fg=json.load(open("/tmp/te-wb/fusion_graph.json"))
groups=[e["data"] for e in fg["elements"] if e["data"].get("type")=="group"]
print("groups:", [(g["id"], g["label"], g["kind"], g["shape"], bool(g.get("tiling"))) for g in groups])
ops=[e for e in fg["elements"] if e["data"].get("type")=="op"]
print("op child nodes:", len(ops))
PY
echo "--- back link in sub-dashboard? ---"
grep -c "返回整网" /tmp/te-wb/kernels/kernel_group0/index.html
echo "--- index.html has cytoscape + expand-collapse + inlined data? ---"
grep -oE "cytoscape|expand-collapse|fusion-data" /tmp/te-wb/index.html | sort -u
```
Expected: INDEX_OK, FG_OK; 2 groups with kind/shape/tiling; op child nodes ≥ 2; back-link count ≥ 1; index.html references cytoscape + expand-collapse + fusion-data.

- [ ] **Step 5: Regression** — `python3 -m pytest tests/tools/ascend-debug/ -v` (all pass); lit smoke passes.

- [ ] **Step 6: Open for visual check** — `xdg-open /tmp/te-wb/index.html` (controller/user verifies: collapsed groups show kind+shape; single-click expands to ops; detail panel shows tiling; drill button → sub-dashboard; back link returns).

- [ ] **Step 7: Commit** `feat(ascend-debug): cytoscape fusion-partition graph as network entry + back-link` (trailer; develop).

---

## Self-Review
- Spec §4 data → Task 1 `build_fusion_graph` (groups + child ops + edges + kind/shape/tiling/drill). ✓
- Spec §5 viewer (compound, collapsed default, single-click expand, detail+tiling, drill button) → Task 2 Step 1. ✓
- Spec §3 ingest writes `<run>/index.html` → Task 2 Step 2. ✓
- Spec §6 back-link → Task 2 Step 3 (parent_view + open_view render). ✓
- Spec §7 two-elewise integration → Task 2 Step 4. ✓
- JS behavior (expand-collapse) can't be unit-tested headlessly → Step 6 is human visual verification; Steps 4 verifies the artifacts/structure that drive it.
- Risk: `load_manifest` rejecting unknown `parent_view` key — Task 2 Step 3 says verify it tolerates extra keys (it validates specific fields, ignores others; confirm during impl).
