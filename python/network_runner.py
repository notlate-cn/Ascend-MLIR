#!/usr/bin/env python3
"""Network runner: orchestrate mixed AscendC+aclnn network compilation & execution.

See docs/superpowers/specs/2026-05-13-network-runner-mixed-cpu-sim-design.md.

Phase 1 + Phase 2 are implemented; phases 3-5 will be added in follow-up commits.

Note: Phase 2 (codegen + compile) requires the simulator LD_LIBRARY_PATH.
Run `source examples/env.sh` (or `source examples/env_gser.sh`) before executing.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Make runner_utils importable when the runner is invoked directly.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "python") not in sys.path:
    sys.path.insert(0, str(_REPO / "python"))

from runner_utils.network_json import NetworkJson  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
AFIR_OPT       = os.environ.get("AFIR_OPT", str(REPO / "build/bin/afir-opt"))
AFIR_TRANSLATE = os.environ.get("AFIR_TRANSLATE", str(REPO / "build/bin/afir-translate"))
ACLNN_BACKEND  = os.environ.get("ACLNN_BACKEND", str(REPO / "build/bin/aclnn-backend"))
RUNTIME_SESSION = os.environ.get("RUNTIME_SESSION", str(REPO / "build/bin/runtime-session"))
AUTOTUNER      = os.environ.get("AUTOTUNER", str(REPO / "build/bin/autotuner"))


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def phase1_outline_or_emit_json(args, work):
    groups = work / "groups"
    groups.mkdir(parents=True, exist_ok=True)

    if args.input_linalg:
        # Step a: --linalg-fold-unit-extent-dims
        intermediate = work / "model_unit_folded.mlir"
        run([AFIR_OPT, "--linalg-fold-unit-extent-dims",
             args.input_linalg, "-o", str(intermediate)])
        # Step b: --vector-plan-group-analysis + --vector-plan-group-outline
        run([AFIR_OPT,
             "--vector-plan-group-analysis",
             f"--vector-plan-group-outline=output-dir={groups}",
             str(intermediate),
             "-o", str(work / "_outlined_combined.mlir")])
    else:
        # Hand-written network: copy DIR/*.mlir → groups/, run emit-network-json
        src = Path(args.input_network)
        if not src.is_dir():
            sys.exit(f"--input-network must be a directory: {src}")
        if not (src / "network.mlir").exists():
            sys.exit(f"missing {src / 'network.mlir'}")
        for f in src.glob("*.mlir"):
            shutil.copy(f, groups / f.name)
        # emit network.json from the copied network.mlir
        run([AFIR_OPT, str(groups / "network.mlir"),
             f"--emit-network-json=path={groups / 'network.json'}",
             "-o", "/dev/null"])

    nj = groups / "network.json"
    if not nj.exists():
        sys.exit(f"phase 1 did not produce {nj}")
    print(f"phase 1 OK → {nj}")
    return groups


def phase2_codegen_compile(work, groups, network):
    """For each ascendc kernel: --vector-plan-codegen → -mlir-to-cann → compile.

    Requires the simulator LD_LIBRARY_PATH; source `examples/env.sh` first.
    subprocess.run inherits the calling process's environment, so env vars set before
    invoking the runner (or the test) are automatically available to all subprocesses.
    """
    artifacts = work / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    for k in network.ascendc_kernels():
        kid = k["id"]
        src = groups / k["file"]
        lowered = work / f"{kid}_lowered.mlir"
        cpp = work / f"{kid}.cpp"
        space = work / f"{kid}_space.json"
        run([AFIR_OPT, str(src), "--vector-plan-codegen", "-o", str(lowered)])
        run([AFIR_TRANSLATE, "-mlir-to-cann", str(lowered),
             "-o", str(cpp), f"--tiling-space-out={space}"])
        run([RUNTIME_SESSION,
             "--kernel", str(cpp),
             "--kernel-kind", "vec",
             "--output", str(artifacts / kid),
             "--name", kid])
    print(f"phase 2 OK → {artifacts}")
    return artifacts


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-linalg")
    g.add_argument("--input-network")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--expected", nargs="+", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--soc", default="Ascend910B1")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-2)
    args = ap.parse_args()

    work = Path(args.workdir).absolute()
    work.mkdir(parents=True, exist_ok=True)

    groups = phase1_outline_or_emit_json(args, work)
    print(f"workdir: {work}")
    print(f"groups:  {groups}")

    network = NetworkJson.load(str(groups / "network.json"))
    phase2_codegen_compile(work, groups, network)


if __name__ == "__main__":
    main()
