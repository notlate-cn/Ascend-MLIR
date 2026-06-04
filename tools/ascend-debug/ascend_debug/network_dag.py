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
