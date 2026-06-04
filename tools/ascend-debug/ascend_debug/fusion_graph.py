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

        kernel_ir = workdir / "groups" / f"{kid}.mlir"
        if kernel_ir.exists():
            text = kernel_ir.read_text(encoding="utf-8", errors="replace")
            sub = stage_graph.parse_stage_mlir(
                {"order": 0, "name": kid, "path": str(kernel_ir)}, text)
            id_map: dict[str, str] = {}
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
                    elements.append({"data": {
                        "id": f"intra_{s}_{t}_{len(elements)}",
                        "source": s, "target": t, "intra": True}})

    for e in summary["edges"]:
        elements.append({"data": {
            "id": f"g_{e['from']}_{e['to']}", "source": e["from"], "target": e["to"]}})

    return {"schema_version": 1, "tool": "ascend-debug",
            "kernel_count": summary["kernel_count"], "elements": elements}
