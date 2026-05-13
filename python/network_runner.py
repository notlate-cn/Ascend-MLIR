#!/usr/bin/env python3
"""Network runner: orchestrate mixed AscendC+aclnn network compilation & execution.

See docs/superpowers/specs/2026-05-13-network-runner-mixed-cpu-sim-design.md.

Phase 1 is implemented; phases 2-5 will be added in follow-up commits.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

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


if __name__ == "__main__":
    main()
