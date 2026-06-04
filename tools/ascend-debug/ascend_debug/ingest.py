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
