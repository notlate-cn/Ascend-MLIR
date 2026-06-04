from __future__ import annotations

import json
import pathlib
import re
import shutil
import types

from ascend_debug import __version__, collect, layout, network_dag, open_view
from ascend_debug.runner import CommandError

# Network-level lowering-stage IR dumps that network_runner's outline phase
# writes, in pipeline order. ingest registers whichever are present so the
# Stage Timeline shows the network's lowering progression.
_NETWORK_STAGES = [
    ("model_recognized.mlir", "recognized"),
    ("model_unit_folded.mlir", "unit-folded"),
    ("model_transpose_folded.mlir", "transpose-folded"),
    ("model_symbolized.mlir", "symbolized"),
    ("_outlined_combined.mlir", "outlined"),
]


def _load_json(path: pathlib.Path, label: str) -> dict:
    if not path.exists():
        raise CommandError(f"{label} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CommandError(f"could not read {label}: {path}: {error}") from error


def _read_json(path: pathlib.Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


_OUT_RE = re.compile(r"out(\d+)\.npy$")
_EXP_RE = re.compile(r"expected(\d+)\.npy$")


def _build_tensor_diff(workdir: pathlib.Path, run_dir: pathlib.Path,
                       atol: float = 1e-2, rtol: float = 1e-2) -> bool:
    """If <workdir> was executed (outputs/out<i>.npy + expected<i>.npy),
    write <run_dir>/summaries/tensor_diff.json (workbench schema). Returns
    True if written, False if the run has no executed outputs."""
    out_dir = workdir / "outputs"
    if not out_dir.is_dir():
        return False
    actual: dict[int, pathlib.Path] = {}
    for p in out_dir.glob("out*.npy"):
        m = _OUT_RE.search(p.name)
        if m:
            actual[int(m.group(1))] = p
    expected: dict[int, pathlib.Path] = {}
    for p in workdir.glob("expected*.npy"):
        m = _EXP_RE.search(p.name)
        if m:
            expected[int(m.group(1))] = p
    indices = sorted(i for i in actual if i in expected)
    if not indices:
        return False
    try:
        import numpy as np
    except Exception:
        return False
    comparisons = []
    for i in indices:
        try:
            a = np.load(actual[i]).astype("float32")
            b = np.load(expected[i]).astype("float32")
        except Exception:
            continue
        if a.shape != b.shape:
            comparisons.append({
                "status": "FAIL", "id": f"network.output[{i}]",
                "kernel_id": None, "task_id": None,
                "max_abs_error": None, "max_rel_error": None,
                "mean_abs_error": None, "atol": atol, "rtol": rtol,
            })
            continue
        diff = abs(a - b)
        max_abs = float(diff.max()) if a.size else 0.0
        mean_abs = float(diff.mean()) if a.size else 0.0
        ok = bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False)) if a.size else True
        comparisons.append({
            "status": "PASS" if ok else "FAIL",
            "id": f"network.output[{i}]",
            "kernel_id": None, "task_id": None,
            "max_abs_error": max_abs, "max_rel_error": None,
            "mean_abs_error": mean_abs, "atol": atol, "rtol": rtol,
        })
    if not comparisons:
        return False
    failed = sum(1 for c in comparisons if c["status"] != "PASS")
    summary = {
        "schema_version": 1, "tool": "ascend-debug",
        "status": "PASS" if failed == 0 else "FAIL",
        "comparison_count": len(comparisons), "failed_count": failed,
        "comparisons": comparisons,
    }
    (run_dir / "summaries").mkdir(parents=True, exist_ok=True)
    layout.write_json(run_dir / "summaries" / "tensor_diff.json", summary)
    return True


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

    try:
        summary = network_dag.build_kernel_dag_summary(network, provenance)
    except (TypeError, AttributeError, KeyError) as error:
        raise CommandError(
            f"network.json has unexpected structure: {net_path}: {error}"
        ) from error
    # Enrich each kernel node with its tiling JSON so the workbench detail panel
    # can show best/space without a second fetch.
    for kid, node in summary.get("nodes", {}).items():
        tiling: dict = {}
        best = _read_json(workdir / f"{kid}_best.json")
        if best is not None:
            tiling["best"] = best
        space = _read_json(workdir / f"{kid}_space.json")
        if space is not None:
            tiling["space"] = space
        node["tiling"] = tiling
    layout.write_json(run_dir / "graphs" / "kernel_dag.summary.json", summary)
    if provenance is not None:
        layout.write_json(run_dir / "network.provenance.json", provenance)

    stages: list = []
    order = 0
    for fname, name in _NETWORK_STAGES:
        src = workdir / fname
        if not src.exists():
            continue
        order += 10
        rel = f"stages/{order:03d}-{name}.mlir"
        shutil.copyfile(src, run_dir / rel)
        stages.append(layout.StageArtifact(order=order, name=name, path=rel, step=name))

    if not stages:
        # Fallback: no named lowering dumps — register a single network IR.
        fallback = _find_network_ir(workdir)
        if fallback is None:
            raise CommandError(
                f"no network IR found under {workdir} "
                "(looked for the model_*.mlir lowering dumps, _outlined_combined.mlir, "
                "groups/network.mlir, groups/kernel_group*.mlir)"
            )
        shutil.copyfile(fallback, run_dir / "stages" / "000-network.mlir")
        stages.append(layout.StageArtifact(order=0, name="network",
                                           path="stages/000-network.mlir", step="source"))

    layout.write_manifest(
        run_dir, mode="network-ingest", preset="", pipeline="auto-fuse-outline",
        stages=tuple(stages), version=__version__, commands=[], reports=[], graphs=[],
    )
    print(f"ascend-debug.ingest.out={run_dir}")
    print(f"ascend-debug.ingest.kernels={summary['kernel_count']}")
    print(f"ascend-debug.ingest.stages={len(stages)}")

    # Per-kernel orchestration: for each kernel that has a pre-codegen IR dump,
    # build a sub-dashboard (collect's 6 stages + generated .cpp + tiling JSON)
    # under run_dir/kernels/<kid>. Resilient: one kernel's failure must not
    # abort ingest.
    dashboard_count = 0
    for k in network.get("kernels", []):
        kid = k.get("id")
        if not isinstance(kid, str) or not kid:
            continue
        kernel_ir = workdir / "groups" / f"{kid}.mlir"
        if not kernel_ir.exists():
            print(f"ascend-debug.ingest.skip={kid} (no groups/{kid}.mlir)")
            continue
        kdir = run_dir / "kernels" / kid
        try:
            collect.collect_run(types.SimpleNamespace(input=kernel_ir, out=kdir))
        except CommandError as error:
            print(f"ascend-debug.ingest.skip={kid} (collect failed: {error})")
            continue

        try:
            cpp_src = workdir / f"{kid}.cpp"
            if cpp_src.exists():
                cpp_dst = kdir / "stages" / "070-codegen.cpp"
                cpp_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cpp_src, cpp_dst)
                man = _load_json(kdir / "manifest.json", f"{kid} manifest.json")
                rebuilt = [
                    layout.StageArtifact(
                        order=s["order"], name=s["name"], path=s["path"], step=s.get("step")
                    )
                    for s in man.get("stages", [])
                ]
                rebuilt.append(
                    layout.StageArtifact(
                        order=70, name="codegen", path="stages/070-codegen.cpp", step="codegen"
                    )
                )
                layout.write_manifest(
                    kdir,
                    mode=man["mode"],
                    preset=man.get("preset", ""),
                    pipeline=man.get("pipeline", "auto-fuse-codegen"),
                    stages=tuple(rebuilt),
                    version=man.get("version", "0.1"),
                    commands=man.get("commands", []),
                    reports=man.get("reports", []),
                    graphs=man.get("graphs", []),
                )

            for suffix in ("_best.json", "_space.json"):
                tiling_src = workdir / f"{kid}{suffix}"
                if tiling_src.exists():
                    tiling_dst = kdir / "tiling" / f"{kid}{suffix}"
                    tiling_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(tiling_src, tiling_dst)

            # Set a back-link to the network entry so the sub-dashboard can
            # return to the fusion graph. (Re-read the manifest in case the
            # 070-codegen rewrite above replaced it.)
            man = _load_json(kdir / "manifest.json", f"{kid} manifest.json")
            man["parent_view"] = "../../index.html"
            layout.write_json(kdir / "manifest.json", man)

            open_view.open_run(types.SimpleNamespace(run_dir=kdir, no_browser=True))

            # Skip the per-kernel overview hop: send the kernel index.html
            # straight to its stage-evolution workbench (views/debug_graph.html
            # defaults to stage mode). Mirrors the main entry redirect below.
            (kdir / "index.html").write_text(
                '<!doctype html><meta charset="utf-8">'
                '<meta http-equiv="refresh" content="0; url=views/debug_graph.html">'
                '<title>ascend-debug · ' + kid + '</title>'
                '<a href="views/debug_graph.html">打开 Kernel 工作台 →</a>',
                encoding="utf-8",
            )
        except (OSError, ValueError, CommandError, KeyError) as error:
            print(f"ascend-debug.ingest.skip={kid} (sub-dashboard failed: {error})")
            continue

        dashboard_count += 1

    print(f"ascend-debug.ingest.kernel_dashboards={dashboard_count}")

    # If the run was executed (sim), write the L0 tensor-diff summary so the
    # Tensor Diff panel shows per-output PASS/FAIL + error metrics.
    if _build_tensor_diff(workdir, run_dir):
        print(f"ascend-debug.ingest.tensor_diff={run_dir / 'summaries' / 'tensor_diff.json'}")

    # The dev-nyh workbench is the network entry: regenerate run_dir/index.html,
    # which reads graphs/kernel_dag.summary.json and defaults to Kernel-DAG mode.
    open_view.open_run(types.SimpleNamespace(run_dir=run_dir, no_browser=True))
    print(f"ascend-debug.ingest.workbench={run_dir / 'index.html'}")

    # Skip the overview hop: send <run>/index.html straight to the workbench.
    # The workbench at views/debug_graph.html resolves its own links via ../,
    # so it works when loaded directly.
    (run_dir / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0; url=views/debug_graph.html">'
        '<title>ascend-debug</title>'
        '<a href="views/debug_graph.html">打开调试工作台 →</a>',
        encoding="utf-8",
    )
    print(f"ascend-debug.ingest.entry={run_dir / 'index.html'} -> views/debug_graph.html")
    return 0
