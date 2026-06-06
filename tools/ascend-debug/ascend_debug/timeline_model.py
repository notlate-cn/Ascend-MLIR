from __future__ import annotations

import pathlib
from typing import Any

LANE_ORDER: tuple[str, ...] = (
    "source",
    "normalize",
    "kernelize",
    "schedule",
    "realize",
    "translate",
    "artifacts",
    "runtime",
)

LANE_TITLES: dict[str, str] = {
    "source": "Source",
    "normalize": "Normalize",
    "kernelize": "Kernelize",
    "schedule": "Schedule",
    "realize": "Realize",
    "translate": "Translate",
    "artifacts": "Artifacts",
    "runtime": "Runtime",
}

OPTIONAL_CONTRACTS: tuple[tuple[str, str, str, str], ...] = (
    ("stage_manifest.json", "stage-manifest", "Stage Manifest", "source"),
    ("kernel_dag.json", "kernel-dag", "Kernel DAG", "kernelize"),
    ("schedule_decisions.json", "schedule-decisions", "Schedule Decisions", "schedule"),
    ("memory_plan.json", "memory-plan", "Memory Plan", "realize"),
    ("run_manifest.json", "run-manifest", "Run Manifest", "runtime"),
)

SUMMARY_CONTRACTS: tuple[tuple[str, str, str, str], ...] = (
    ("graphs/kernel_dag.summary.json", "kernel-dag-summary", "Kernel DAG Summary", "kernelize"),
    ("summaries/memory.json", "memory-summary", "Memory Summary", "realize"),
    ("summaries/tensor_diff.json", "tensor-diff-summary", "Tensor Diff Summary", "runtime"),
    ("summaries/locate.json", "locate-summary", "Locate Summary", "runtime"),
)

KNOWN_ARTIFACT_FIELDS = {
    "kind",
    "label",
    "path",
    "status",
    "diagnostic",
    "producer_stage",
    "producer_step",
    "source_stage",
    "contract_kind",
    "source_contract",
}


def _empty_lane(lane_id: str) -> dict[str, Any]:
    return {
        "id": lane_id,
        "title": LANE_TITLES[lane_id],
        "stage_orders": [],
        "evidence": [],
        "contracts": [],
        "artifacts": [],
        "diagnostics": [],
        "source": "manifest",
    }


def _lower_text(value: Any) -> str:
    return str(value or "").replace("_", "-").lower()


def _lane_id_from_text(value: Any) -> str | None:
    text = _lower_text(value)
    if not text:
        return None
    if text in LANE_ORDER:
        return text
    if "run-manifest" in text or "runtime" in text or "run-status" in text:
        return "runtime"
    if "artifact" in text:
        return "artifacts"
    if "kernel-dag" in text or "kernelized" in text or "kernelize" in text:
        return "kernelize"
    if "schedule" in text or "tiling" in text:
        return "schedule"
    if "memory" in text or "realize" in text or "buffer" in text:
        return "realize"
    if any(
        marker in text
        for marker in ("translate", "compute-lower", "parallelize", "prepare-for-emit", "cann-signature")
    ):
        return "translate"
    if "normalize" in text:
        return "normalize"
    if "source" in text:
        return "source"
    return None


def _stage_lane_id(stage: dict[str, Any]) -> str:
    return (
        _lane_id_from_text(stage.get("phase"))
        or _lane_id_from_text(stage.get("step"))
        or _lane_id_from_text(stage.get("name"))
        or "source"
    )


def _stage_title(stage: dict[str, Any]) -> str:
    step_info = stage.get("step_info") if isinstance(stage.get("step_info"), dict) else {}
    return str(step_info.get("title") or stage.get("step") or stage.get("name") or "stage")


def _artifact_lane_from_explicit_fields(artifact: dict[str, Any]) -> tuple[str | None, str | None]:
    for field in ("producer_stage", "producer_step"):
        value = artifact.get(field)
        lane_id = _lane_id_from_text(value)
        if lane_id:
            return lane_id, "contract"
    for field in ("source_stage", "stage"):
        value = artifact.get(field)
        lane_id = _lane_id_from_text(value)
        if lane_id:
            return lane_id, "legacy_adapter"
    return None, None


def _artifact_lane_from_kind(artifact: dict[str, Any]) -> str:
    kind = _lower_text(artifact.get("kind"))
    path = _lower_text(artifact.get("path"))
    text = f"{kind} {path}"
    if "kernel-cpp" in text or path.endswith("kernel.cpp"):
        return "translate"
    if "host-tiling-cpp" in text or path.endswith("host-tiling.cpp") or path.endswith("host_tiling.cpp"):
        return "translate"
    if "tiling-space" in text or path.endswith("tiling-space.json") or path.endswith("tiling_space.json"):
        return "schedule"
    if "memory-plan" in text:
        return "realize"
    if "run-manifest" in text or path.endswith("run-manifest.json") or path.endswith("run_manifest.json"):
        return "runtime"
    if "artifact-manifest" in text or path.endswith("artifact-manifest.json") or path.endswith("artifact_manifest.json"):
        return "artifacts"
    return "artifacts"


def _artifact_item(artifact: dict[str, Any], *, source: str) -> dict[str, Any]:
    raw = {key: value for key, value in artifact.items() if key not in KNOWN_ARTIFACT_FIELDS}
    item = {
        "kind": artifact.get("kind") or "artifact",
        "label": artifact.get("label") or artifact.get("kind") or artifact.get("path") or "artifact",
        "path": artifact.get("path"),
        "status": artifact.get("status"),
        "diagnostic": artifact.get("diagnostic"),
        "source": source,
    }
    if artifact.get("producer_stage"):
        item["producer_stage"] = artifact.get("producer_stage")
    if artifact.get("producer_step"):
        item["producer_step"] = artifact.get("producer_step")
    if artifact.get("contract_kind"):
        item["contract_kind"] = artifact.get("contract_kind")
    if artifact.get("source_contract"):
        item["source_contract"] = artifact.get("source_contract")
    if raw:
        item["raw"] = raw
    return item


def _graph_lane_id(graph: dict[str, Any]) -> str:
    kind = _lower_text(graph.get("kind"))
    path = _lower_text(graph.get("path"))
    if "kernel-dag" in kind or "kernelized" in kind or "kernel-dag" in path or "kernelized" in path:
        return "kernelize"
    if "artifact-manifest" in kind or "artifact-manifest" in path or "artifact_manifest" in path:
        return "artifacts"
    return _lane_id_from_text(kind) or "artifacts"


def _contract_item(
    *,
    kind: str,
    label: str,
    path: str,
    source: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "label": label,
        "path": path,
        "source": source,
    }


def _append_contract_once(
    lanes: dict[str, dict[str, Any]],
    seen_contract_paths: set[tuple[str, str]],
    lane_id: str,
    item: dict[str, Any],
) -> None:
    path = item.get("path")
    if isinstance(path, str):
        key = (lane_id, path)
        if key in seen_contract_paths:
            return
        seen_contract_paths.add(key)
    lanes[lane_id]["contracts"].append(item)


def _report_lane_id(report: dict[str, Any]) -> str:
    return _lane_id_from_text(report.get("stage")) or _lane_id_from_text(report.get("path")) or "artifacts"


def _source_from_lanes(lanes: dict[str, dict[str, Any]]) -> str:
    sources: set[str] = set()
    for lane in lanes.values():
        for bucket in ("contracts", "artifacts"):
            for item in lane[bucket]:
                source = item.get("source")
                if source in ("contract", "legacy_adapter"):
                    sources.add(source)
    if "contract" in sources and "legacy_adapter" in sources:
        return "mixed"
    if "contract" in sources:
        return "contract"
    return "legacy_adapter"


def _finalize_lane_sources(lanes: dict[str, dict[str, Any]]) -> None:
    for lane in lanes.values():
        sources: set[str] = set()
        for bucket in ("contracts", "artifacts"):
            for item in lane[bucket]:
                source = item.get("source")
                if source in ("contract", "legacy_adapter"):
                    sources.add(source)
        if "contract" in sources and "legacy_adapter" in sources:
            lane["source"] = "mixed"
        elif "contract" in sources:
            lane["source"] = "contract"
        elif "legacy_adapter" in sources:
            lane["source"] = "legacy_adapter"


def build_timeline_model(run_dir: pathlib.Path, manifest: dict[str, Any]) -> dict[str, Any]:
    lanes = {lane_id: _empty_lane(lane_id) for lane_id in LANE_ORDER}
    seen_contract_paths: set[tuple[str, str]] = set()

    for stage in sorted(manifest.get("stages", []), key=lambda item: item.get("order", 0)):
        if not isinstance(stage, dict):
            continue
        lane = lanes[_stage_lane_id(stage)]
        order = stage.get("order")
        lane["stage_orders"].append(order)
        lane["evidence"].append(
            {
                "kind": "stage",
                "label": _stage_title(stage),
                "name": stage.get("name"),
                "step": stage.get("step"),
                "order": order,
                "path": stage.get("path"),
                "source": "manifest",
            }
        )

    for graph in manifest.get("graphs", []):
        if not isinstance(graph, dict) or not isinstance(graph.get("path"), str):
            continue
        lane_id = _graph_lane_id(graph)
        label = graph.get("label") or graph.get("kind") or graph.get("path")
        _append_contract_once(
            lanes,
            seen_contract_paths,
            lane_id,
            _contract_item(
                kind=str(graph.get("kind") or "graph"),
                label=str(label),
                path=str(graph["path"]),
                source="legacy_adapter",
            ),
        )

    for rel_path, kind, label, lane_id in OPTIONAL_CONTRACTS:
        if (run_dir / rel_path).exists():
            _append_contract_once(
                lanes,
                seen_contract_paths,
                lane_id,
                _contract_item(kind=kind, label=label, path=rel_path, source="contract"),
            )

    for rel_path, kind, label, lane_id in SUMMARY_CONTRACTS:
        if (run_dir / rel_path).exists():
            _append_contract_once(
                lanes,
                seen_contract_paths,
                lane_id,
                _contract_item(kind=kind, label=label, path=rel_path, source="legacy_adapter"),
            )

    for artifact in manifest.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        explicit_lane, explicit_source = _artifact_lane_from_explicit_fields(artifact)
        lane_id = explicit_lane or _artifact_lane_from_kind(artifact)
        source = explicit_source or "legacy_adapter"
        lanes[lane_id]["artifacts"].append(_artifact_item(artifact, source=source))

    for report in manifest.get("reports", []):
        if not isinstance(report, dict) or not isinstance(report.get("path"), str):
            continue
        lane_id = _report_lane_id(report)
        lanes[lane_id]["diagnostics"].append(
            {
                "kind": "report",
                "label": report.get("stage") or report.get("path"),
                "path": report.get("path"),
                "source": "manifest",
            }
        )

    _finalize_lane_sources(lanes)
    return {
        "schema_version": 1,
        "source": _source_from_lanes(lanes),
        "lanes": [lanes[lane_id] for lane_id in LANE_ORDER],
    }
