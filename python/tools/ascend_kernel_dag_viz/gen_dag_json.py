#!/usr/bin/env python3
"""gen_dag_json: build a dag.json + dag.html viewer from network.json +
network.provenance.json (and optional runtime artifacts).

P1-4 of docs/auto-fuse/debug.md §7.7. The viewer is a self-contained
HTML page (cytoscape.js + dagre layout, loaded via CDN) that fetches
dag.json next to itself; serve the parent dir over `python -m http.server`
to browse.

Usage:
  gen_dag_json.py <workdir>

  expects <workdir>/groups/network.json and
          <workdir>/groups/network.provenance.json
  writes  <workdir>/dag.json and <workdir>/dag.html
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "python") not in sys.path:
    sys.path.insert(0, str(_REPO / "python"))
from runner_utils import logger as _lg  # noqa: E402

_log = _lg.get_logger("gen_dag_json")


def build_dag(network: dict, provenance: dict) -> dict:
    # Map kernel_id → provenance entry (so we can enrich each kernel node).
    prov_by_id = {k["kernel_id"]: k for k in (provenance or {}).get("kernels", [])}

    nodes = []
    edges = []

    # Input ghost nodes (one per coordinator arg).
    for inp in network.get("inputs", []):
        nodes.append({
            "id": "input:" + inp["name"],
            "kind": "input",
            "label": inp["name"],
            "shape": inp.get("shape"),
            "dtype": inp.get("dtype"),
        })

    # Kernel nodes.
    for k in network.get("kernels", []):
        kid = k["id"]
        prov = prov_by_id.get(kid, {})
        summary = prov.get("fused_ops_summary") or k.get("op") or k.get("kind", "?")
        node = {
            "id": kid,
            "kind": k.get("kind", "ascendc"),
            "label": f"{kid}\n[{summary}]",
            "summary": summary,
            "results": k.get("results", []),
            "source_ops": prov.get("source_ops", []),
            "boundary_source_ops": prov.get("boundary_source_ops", []),
            "output_checkpoint_hint": prov.get("output_checkpoint_hint", []),
        }
        if k.get("op"):
            node["aclnn_op"] = k["op"]
        nodes.append(node)

        # Edges from each kernel arg.
        for ai, arg in enumerate(k.get("args", [])):
            src_id = None
            label = None
            if arg.get("from") == "kernel":
                src_id = arg["kernel"]
                label = f"r{arg.get('result', 0)} → arg{ai}"
            elif arg.get("from") == "input":
                src_id = "input:" + arg["name"]
                label = f"arg{ai}"
            elif arg.get("from") == "alloc":
                # DPS init buffer; usually not interesting to show as an edge,
                # but skip rather than fake a source.
                continue
            elif arg.get("from") == "const":
                # Inline scalar / weight; skip (would clutter graph).
                continue
            else:
                continue
            edges.append({
                "id": f"{src_id}->{kid}#{ai}",
                "source": src_id,
                "target": kid,
                "label": label,
            })

    # Output ghost nodes.
    for out in network.get("outputs", []):
        out_id = "output:" + out["name"]
        nodes.append({
            "id": out_id,
            "kind": "output",
            "label": out["name"],
        })
        if out.get("from") == "kernel":
            edges.append({
                "id": f"{out['kernel']}->{out_id}",
                "source": out["kernel"],
                "target": out_id,
                "label": f"r{out.get('result', 0)}",
            })

    return {
        "schema_version": 1,
        "function": network.get("function"),
        "nodes": nodes,
        "edges": edges,
    }


def main():
    ap = argparse.ArgumentParser(prog="gen_dag_json",
        description="Build dag.json + dag.html from network.json + provenance.")
    ap.add_argument("workdir", help="network_runner workdir (build_e2e/).")
    args = ap.parse_args()

    work = Path(args.workdir).resolve()
    groups = work / "groups"
    nj_path = groups / "network.json"
    pv_path = groups / "network.provenance.json"
    if not nj_path.exists():
        _log.error(f"{nj_path} not found")
        sys.exit(2)

    network = json.loads(nj_path.read_text())
    provenance = json.loads(pv_path.read_text()) if pv_path.exists() else None
    if provenance is None:
        _log.warning(f"{pv_path} not found — proceeding without provenance")

    dag = build_dag(network, provenance or {})
    out_json = work / "dag.json"
    dag_str = json.dumps(dag, indent=2)
    out_json.write_text(dag_str + "\n")
    print(f"dag.json   → {out_json}")

    # Embed dag.json into the viewer so the resulting single HTML opens
    # directly via file:// (no http.server needed, no localhost proxy
    # interference). The fetch() call is replaced with an inline const.
    viewer_src = Path(__file__).resolve().parent / "viewer.html"
    out_html = work / "dag.html"
    if viewer_src.exists():
        tmpl = viewer_src.read_text()
        # Replace the fetch('dag.json') block with an inlined const.
        old = ("  const res = await fetch('dag.json');\n"
               "  if (!res.ok) {\n"
               "    document.getElementById('meta').textContent = "
               "\"fetch dag.json failed: \" + res.status;\n"
               "    return;\n"
               "  }\n"
               "  const dag = await res.json();\n")
        new = f"  const dag = {dag_str};\n"
        if old not in tmpl:
            _log.warning("viewer.html fetch block not found — viewer may "
                         "still require http server")
            shutil.copy(viewer_src, out_html)
        else:
            out_html.write_text(tmpl.replace(old, new))
        print(f"dag.html   → {out_html}")
        print()
        print(f"View in browser (no server needed):")
        print(f"  xdg-open {out_html}     # or: drag {out_html.name} into the browser")
    else:
        _log.warning(f"viewer.html not found at {viewer_src}")


if __name__ == "__main__":
    main()
