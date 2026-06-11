# ascend-debug Workbench Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Light up the workbench Kernel-DAG panel by adapting develop's `network.json` (+`network.provenance.json`) into the `kernel_dag.summary.json` shape, exposed via a new `ascend-debug ingest` subcommand.

**Architecture:** Two new pure-Python modules under `tools/ascend-debug/ascend_debug/`: `network_dag.py` (the adapter) and `ingest.py` (the subcommand). One CLI edit. No changes to `open_view.py`/`debug_graph.py` (they already render `kernel_dag.summary.json`). No afir-opt/runtime needed — consumes existing network_runner outline artifacts.

**Tech Stack:** Python 3 stdlib; pytest. Real fixture: `examples/two-elewise-e2e/build_e2e/groups/`.

**Spec:** `docs/superpowers/specs/2026-06-04-ascend-debug-workbench-phase1-design.md`.

---

### Task 1: `network_dag.py` adapter (network.json → kernel_dag.summary.json)

**Files:**
- Create: `tools/ascend-debug/ascend_debug/network_dag.py`
- Test: `tests/tools/ascend-debug/test_network_dag.py`

- [ ] **Step 1: Write failing unit tests**

Create `tests/tools/ascend-debug/test_network_dag.py`:
```python
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "ascend-debug"))

from ascend_debug import network_dag


def _net():
    return {
        "function": "f",
        "inputs": [{"name": "arg0", "shape": [4, 4], "dtype": "f16"}],
        "kernels": [
            {"id": "k0", "kind": "ascendc",
             "args": [{"from": "input", "name": "arg0"}],
             "results": [{"name": "k0_r0", "shape": [4, 4], "dtype": "f16"}]},
            {"id": "k1", "kind": "ascendc",
             "args": [{"from": "input", "name": "arg0"}],
             "results": [{"name": "k1_r0", "shape": [4, 4], "dtype": "f16"}]},
            {"id": "k2", "kind": "aclnn", "op": "MatMul",
             "args": [{"from": "kernel", "kernel": "k0", "result": 0},
                      {"from": "kernel", "kernel": "k1", "result": 0}],
             "results": [{"name": "k2_r0", "shape": [4, 4], "dtype": "f16"}]},
        ],
        "outputs": [{"from": "kernel", "kernel": "k2", "result": 0}],
    }


def _prov():
    return {"kernels": [
        {"kernel_id": "k0", "fused_ops_summary": "add", "source_ops": [{"id": "op_000"}]},
        {"kernel_id": "k1", "fused_ops_summary": "mul", "source_ops": [{"id": "op_001"}]},
        {"kernel_id": "k2", "fused_ops_summary": "matmul", "source_ops": [{"id": "op_002"}]},
    ]}


def test_nodes_keyed_by_kernel_id():
    s = network_dag.build_kernel_dag_summary(_net(), _prov())
    assert isinstance(s["nodes"], dict)
    assert set(s["nodes"]) == {"k0", "k1", "k2"}
    assert s["kernel_count"] == 3


def test_edges_from_kernel_args():
    s = network_dag.build_kernel_dag_summary(_net(), _prov())
    pairs = {(e["from"], e["to"]) for e in s["edges"]}
    assert ("k0", "k2") in pairs and ("k1", "k2") in pairs
    assert s["graph_edges"] == len(s["edges"])


def test_depth_and_critical_path():
    s = network_dag.build_kernel_dag_summary(_net(), _prov())
    assert s["nodes"]["k0"]["depth"] == 0
    assert s["nodes"]["k1"]["depth"] == 0
    assert s["nodes"]["k2"]["depth"] == 1
    assert s["critical_path_depth"] == 1


def test_label_from_provenance_and_shape():
    s = network_dag.build_kernel_dag_summary(_net(), _prov())
    assert s["nodes"]["k2"]["ops"][0]["label"] == "matmul"
    assert s["nodes"]["k2"]["aclnn_op"] == "MatMul"
    assert s["nodes"]["k0"]["output_shape"] == [4, 4]


def test_no_provenance_still_builds():
    s = network_dag.build_kernel_dag_summary(_net(), None)
    assert set(s["nodes"]) == {"k0", "k1", "k2"}
    # label falls back to op / kind
    assert s["nodes"]["k2"]["ops"][0]["label"] in ("MatMul", "aclnn")


def test_cycle_guard_terminates():
    net = {"kernels": [
        {"id": "a", "kind": "ascendc", "args": [{"from": "kernel", "kernel": "b"}], "results": []},
        {"id": "b", "kind": "ascendc", "args": [{"from": "kernel", "kernel": "a"}], "results": []},
    ]}
    s = network_dag.build_kernel_dag_summary(net, None)  # must not hang
    assert set(s["nodes"]) == {"a", "b"}
```

- [ ] **Step 2: Run → fail** — `python3 -m pytest tests/tools/ascend-debug/test_network_dag.py -v` (no module).

- [ ] **Step 3: Implement `network_dag.py`**
```python
from __future__ import annotations

from typing import Any


def _compute_depths(kids: list[str], preds: dict[str, list[str]]) -> dict[str, int]:
    """Longest-path depth from roots; cycle-safe (memo + on-stack guard)."""
    depth: dict[str, int] = {}
    visiting: set[str] = set()

    def resolve(kid: str) -> int:
        if kid in depth:
            return depth[kid]
        if kid in visiting:
            return 0  # cycle: break, treat as root-ish
        visiting.add(kid)
        best = 0
        for p in preds.get(kid, []):
            if p == kid:
                continue
            best = max(best, resolve(p) + 1)
        visiting.discard(kid)
        depth[kid] = best
        return best

    for kid in kids:
        resolve(kid)
    return depth


def build_kernel_dag_summary(network: dict[str, Any],
                             provenance: dict[str, Any] | None) -> dict[str, Any]:
    prov_by_id = {
        k.get("kernel_id"): k
        for k in (provenance or {}).get("kernels", [])
        if isinstance(k, dict)
    }
    kernels = network.get("kernels", []) or []
    kids = [k["id"] for k in kernels if "id" in k]

    # kernel -> predecessor kernels (from args with from == "kernel")
    preds: dict[str, list[str]] = {kid: [] for kid in kids}
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str]] = set()
    for k in kernels:
        kid = k.get("id")
        if kid is None:
            continue
        for arg in k.get("args", []) or []:
            if arg.get("from") == "kernel":
                src = arg.get("kernel")
                if src is None:
                    continue
                preds[kid].append(src)
                key = (src, kid)
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append({"from": src, "to": kid})

    depths = _compute_depths(kids, preds)

    nodes: dict[str, Any] = {}
    for k in kernels:
        kid = k.get("id")
        if kid is None:
            continue
        prov = prov_by_id.get(kid, {})
        results = k.get("results", []) or []
        first = results[0] if results else {}
        label = prov.get("fused_ops_summary") or k.get("op") or k.get("kind") or "kernel"
        node = {
            "id": kid,
            "kind": k.get("kind", "ascendc"),
            "ops": [{"label": label, "op": k.get("op")}],
            "depth": depths.get(kid, 0),
            "workspace_size": 0,
            "output_shape": first.get("shape"),
            "output_dtype": first.get("dtype"),
            "source_ops": prov.get("source_ops", []),
        }
        if k.get("op"):
            node["aclnn_op"] = k["op"]
        nodes[kid] = node

    return {
        "schema_version": 1,
        "tool": "ascend-debug",
        "nodes": nodes,
        "edges": edges,
        "kernel_count": len(nodes),
        "graph_edges": len(edges),
        "critical_path_depth": max(depths.values(), default=0),
    }
```

- [ ] **Step 4: Run → pass** — `python3 -m pytest tests/tools/ascend-debug/test_network_dag.py -v` → 6 passed.

- [ ] **Step 5: Commit**
```bash
git add tools/ascend-debug/ascend_debug/network_dag.py tests/tools/ascend-debug/test_network_dag.py
git commit -m "feat(ascend-debug): network.json -> kernel_dag.summary.json adapter"
```
(trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; develop; don't stage externals/scripts.)

---

### Task 2: `ingest` subcommand + integration

**Files:**
- Create: `tools/ascend-debug/ascend_debug/ingest.py`
- Modify: `tools/ascend-debug/ascend-debug.py` (add `ingest` subparser)
- Test: extend `tests/tools/ascend-debug/test_cli_parser.py`

- [ ] **Step 1: Extend the CLI parser test (failing)**

In `tests/tools/ascend-debug/test_cli_parser.py`, update `test_parser_has_only_collect_and_open` → rename/replace with:
```python
def test_parser_has_collect_open_ingest():
    entry = _load_entry()
    parser = entry.build_parser()
    sub = next(a for a in parser._actions if hasattr(a, "choices") and a.choices)
    assert set(sub.choices) == {"collect", "open", "ingest"}


def test_ingest_args():
    entry = _load_entry()
    parser = entry.build_parser()
    args = parser.parse_args(["ingest", "wd", "--out", "run"])
    assert str(args.network_workdir) == "wd"
    assert str(args.out) == "run"
```
(Delete the old `test_parser_has_only_collect_and_open` since the choice set changes.)

- [ ] **Step 2: Run → fail** — `python3 -m pytest tests/tools/ascend-debug/test_cli_parser.py -v`.

- [ ] **Step 3: Implement `ingest.py`**
```python
from __future__ import annotations

import json
import pathlib
import shutil

from ascend_debug import __version__, layout, network_dag
from ascend_debug.runner import CommandError


def _load_json(path: pathlib.Path, label: str) -> dict:
    if not path.exists():
        raise CommandError(f"{label} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CommandError(f"could not read {label}: {path}: {error}") from error


def _find_network_ir(workdir: pathlib.Path) -> pathlib.Path | None:
    for rel in ("_outlined_combined.mlir", "groups/network.mlir"):
        cand = workdir / rel
        if cand.exists():
            return cand
    groups = sorted((workdir / "groups").glob("kernel_group*.mlir")) if (workdir / "groups").is_dir() else []
    return groups[0] if groups else None


def ingest_run(args) -> int:
    workdir: pathlib.Path = args.network_workdir
    run_dir: pathlib.Path = args.out
    net_path = workdir / "groups" / "network.json"
    network = _load_json(net_path, "network.json")
    prov_path = workdir / "groups" / "network.provenance.json"
    provenance = _load_json(prov_path, "network.provenance.json") if prov_path.exists() else None

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "graphs").mkdir(exist_ok=True)
    (run_dir / "stages").mkdir(exist_ok=True)

    summary = network_dag.build_kernel_dag_summary(network, provenance)
    layout.write_json(run_dir / "graphs" / "kernel_dag.summary.json", summary)
    if provenance is not None:
        layout.write_json(run_dir / "network.provenance.json", provenance)

    # Register the outlined network IR as a single stage so `open` has an IR view.
    network_ir = _find_network_ir(workdir)
    if network_ir is None:
        raise CommandError(
            f"no network IR found under {workdir} "
            "(looked for _outlined_combined.mlir, groups/network.mlir, groups/kernel_group*.mlir)"
        )
    stage_dst = run_dir / "stages" / "000-network.mlir"
    shutil.copyfile(network_ir, stage_dst)
    stages = (
        layout.StageArtifact(order=0, name="network", path="stages/000-network.mlir", step="source"),
    )
    layout.write_manifest(
        run_dir, mode="network-ingest", preset="", pipeline="auto-fuse-outline",
        stages=stages, version=__version__, commands=[], reports=[], graphs=[],
    )
    print(f"ascend-debug.ingest.out={run_dir}")
    print(f"ascend-debug.ingest.kernels={summary['kernel_count']}")
    return 0
```

- [ ] **Step 4: Add the `ingest` subparser in `ascend-debug.py`**

After the `open` subparser block, add:
```python
    from ascend_debug.ingest import ingest_run
    ingest = subparsers.add_parser("ingest", help="Build a workbench run dir from a network_runner work dir")
    ingest.add_argument("network_workdir", type=pathlib.Path)
    ingest.add_argument("--out", type=pathlib.Path, required=True)
    ingest.set_defaults(handler=ingest_run)
```
(Place the `from ascend_debug.ingest import ingest_run` import with the other top-of-file imports rather than inline if that matches the file's style — match existing import placement.)

- [ ] **Step 5: Run parser test → pass** — `python3 -m pytest tests/tools/ascend-debug/test_cli_parser.py -v`.

- [ ] **Step 6: Integration on two-elewise (real artifacts)**
```bash
cd /home/gser/code/Ascend-MLIR
rm -rf /tmp/te-wb
PYTHONPATH=tools/ascend-debug python3 tools/ascend-debug/ascend-debug.py ingest examples/two-elewise-e2e/build_e2e --out /tmp/te-wb
PYTHONPATH=tools/ascend-debug python3 tools/ascend-debug/ascend-debug.py open /tmp/te-wb --no-browser
python3 - <<'PY'
import json
s=json.load(open("/tmp/te-wb/graphs/kernel_dag.summary.json"))
print("kernel_count:", s["kernel_count"], "edges:", s["graph_edges"], "nodes:", list(s["nodes"]))
assert s["kernel_count"] >= 1 and isinstance(s["nodes"], dict)
wb=json.load(open("/tmp/te-wb/summaries/debug_graph.json"))
print("workspace kernel_dag nodes:", len(wb.get("kernel_dag",{}).get("nodes",{})))
assert wb.get("kernel_dag",{}).get("nodes"), "Kernel-DAG panel would be empty!"
print("OK: Kernel-DAG panel populated")
PY
```
Expected: `ingest` prints kernel count; `open` exits 0; the assertions pass (workbench `kernel_dag.nodes` non-empty → Kernel-DAG panel renders).

- [ ] **Step 7: Full suite + commit**
```bash
python3 -m pytest tests/tools/ascend-debug/ -v   # all pass
git add tools/ascend-debug/ascend_debug/ingest.py tools/ascend-debug/ascend-debug.py tests/tools/ascend-debug/test_cli_parser.py
git commit -m "feat(ascend-debug): ingest subcommand lights up workbench Kernel-DAG panel"
```
(trailer; develop; don't stage externals/scripts.)

---

## Self-Review

- Spec §4 adapter → Task 1 (with the exact mapping + cycle guard). ✓
- Spec §3 ingest UX + §5 manifest/IR stage → Task 2 `ingest.py`. ✓
- Spec §6 files → both tasks (network_dag.py, ingest.py, CLI edit, tests). ✓ No open_view/debug_graph changes. ✓
- Spec §7 testing (unit adapter + two-elewise integration + workbench-non-empty assertion) → Task 1 Step 1, Task 2 Step 6. ✓
- GPT-2 (spec §7 second integration) is a manual follow-up after this lands (needs a network_runner GPT-2 work dir) — not a code task here.
- No placeholders; code complete for each step. Names consistent: `build_kernel_dag_summary`, `ingest_run`, `network_workdir`, summary keys (`nodes`/`edges`/`kernel_count`/`graph_edges`/`critical_path_depth`) match the workbench's reads and the adapter output.
