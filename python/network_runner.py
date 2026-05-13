#!/usr/bin/env python3
"""Network runner: orchestrate mixed AscendC+aclnn network compilation & execution.

See docs/superpowers/specs/2026-05-13-network-runner-mixed-cpu-sim-design.md.

Phase 1, 2, 3, and 4 are implemented; phase 5 will be added in a follow-up commit.

Note: Phases 2-4 (codegen + compile + sim run + autotune) require the simulator LD_LIBRARY_PATH.
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


def eval_block_dim(space: dict, params: dict) -> int:
    """Evaluate space['block_dim_expr'] under integer params.

    Grammar (matches autotuner_main.cpp evalBlockExpr):
        expr  ::= id | int | '(' expr op expr ')' | 'ceil(' expr '/' expr ')'
        op    ::= + | - | * | /
        id    ::= tiling param name or shape key (treated as 1 if unknown)

    Returns 1 if the expression is empty or cannot be evaluated.
    """
    expr = (space.get("block_dim_expr") or "").replace(" ", "")
    if not expr:
        return 1

    def _eval(e: str) -> int:
        e = e.strip()
        if not e:
            return 1

        # ceil(A/B) -- find the last top-level '/' inside the parens.
        if e.startswith("ceil(") and e.endswith(")"):
            inner = e[5:-1]
            depth = 0
            slash = -1
            for i, c in enumerate(inner):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                elif c == "/" and depth == 0:
                    slash = i
            if slash == -1:
                return _eval(inner)
            a = _eval(inner[:slash])
            b = _eval(inner[slash + 1:])
            return 1 if b == 0 else (a + b - 1) // b

        # (A op B) -- fully-parenthesized binary expression.
        if e.startswith("(") and e.endswith(")"):
            inner = e[1:-1]
            depth = 0
            for i, c in enumerate(inner):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                elif depth == 0 and i > 0 and c in "+-*/":
                    # Skip unary minus after another operator or '('
                    prev = inner[i - 1]
                    if prev in "+-*/(":
                        continue
                    a = _eval(inner[:i])
                    b = _eval(inner[i + 1:])
                    if c == "+":
                        return a + b
                    if c == "-":
                        return a - b
                    if c == "*":
                        return a * b
                    return 1 if b == 0 else a // b
            return _eval(inner)  # redundant parens

        # Leaf: integer literal or param name.
        try:
            return int(e)
        except ValueError:
            pass
        if e in params:
            return int(params[e])
        return 1  # unknown name — warn-and-default as in C++ version

    try:
        return _eval(expr)
    except Exception:
        return 1


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


def phase3_default_build_and_dump(work, groups, network, artifacts, args):
    """Build network_host.cpp with default tilings, g++ link, run, dump intermediates.

    Steps:
      1. Build tilings_default.json from each kernel's _space.json.
      2. Generate network_host_default.cpp via aclnn-backend.
      3. g++ link against libAscendCRuntime + CANN libs.
      4. Run with --dump-intermediates DIR.

    Returns (tilings_path, intermediates_dir).
    """
    # 1) Build tilings_default.json
    default_tilings: dict = {}
    for k in network.ascendc_kernels():
        kid = k["id"]
        space_path = work / f"{kid}_space.json"
        space = json.loads(space_path.read_text())
        # Tunable params: "fixed": false, default = first entry of "values".
        params: dict = {}
        for p in space.get("tiling_params", []):
            if not p.get("fixed", False):
                vals = p.get("values", [])
                params[p["name"]] = vals[0] if vals else 16
        params["_block_dim"] = eval_block_dim(space, params)
        default_tilings[kid] = params

    tilings_path = work / "tilings_default.json"
    tilings_path.write_text(json.dumps(default_tilings, indent=2))

    # 2) Generate network_host_default.cpp
    host_cpp = work / "network_host_default.cpp"
    run([
        ACLNN_BACKEND,
        "--input",          str(groups / "network.mlir"),
        "--output",         str(host_cpp),
        "--tilings",        str(tilings_path),
        "--kernel-binaries", str(artifacts),
    ])

    # 3) g++ link
    from runner_utils.build_host import link_host
    harness_cpp = REPO / "python/runner_utils/harness.cpp"
    binary = work / "network_test_default"
    cann_home = os.environ.get("ASCEND_HOME_PATH",
                               "/home/gser/Ascend/cann")
    has_aclnn = bool(network.aclnn_kernels())
    link_host(
        host_cpp, harness_cpp, binary, REPO,
        cann_home=cann_home,
        soc=args.soc,
        has_aclnn_ops=has_aclnn,
    )

    # 4) Run with --dump-intermediates. One --output per network output
    # (matters! harness's outputs[] is sized from --output count; if it's
    # smaller than the network's actual output count, network_impl writes
    # past the vector end and the binary segfaults at cleanup).
    inter = work / "intermediates_default"
    inter.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary)]
    for p in args.inputs:
        cmd += ["--input", p]
    for i in range(len(network.outputs)):
        cmd += ["--output", str(work / f"output_default_{i}.npy")]
    cmd += ["--dump-intermediates", str(inter)]
    run(cmd)

    print(f"phase 3 OK → {inter}")
    return tilings_path, inter


def _shape_keys_needed(space: dict) -> list[str]:
    """Collect all shape_key strings the kernel's tiling_space requires."""
    keys = []
    for p in space.get("tiling_params", []):
        sk = p.get("shape_key")
        if sk and sk not in keys:
            keys.append(sk)
    return keys


def _shape_arg_for_kernel(inter: Path, kid: str, space: dict) -> str:
    """Build the --shape KEY=VAL,... string for autotuner.

    shape_key format is `arg<i>_dim<j>` — resolve by reading the dumped
    intermediate input npy and indexing its .shape[j].
    """
    import numpy as np
    import re
    parts = []
    for key in _shape_keys_needed(space):
        m = re.match(r"^arg(\d+)_dim(\d+)$", key)
        if not m:
            sys.exit(f"phase 4: unsupported shape_key format: {key}")
        arg_idx, dim_idx = int(m.group(1)), int(m.group(2))
        npy = inter / f"{kid}_in_{arg_idx}.npy"
        if not npy.exists():
            sys.exit(f"phase 4: missing dumped input for shape_key {key}: {npy}")
        shape = np.load(npy).shape
        if dim_idx >= len(shape):
            sys.exit(f"phase 4: shape_key {key} dim out of range for shape {shape}")
        parts.append(f"{key}={shape[dim_idx]}")
    return ",".join(parts)


def phase4_autotune(work, network, inter, args):
    """Run autotuner per ascendc kernel; aggregate tilings_best.json."""
    tilings_best: dict = {}
    for k in network.ascendc_kernels():
        kid = k["id"]
        space_path = work / f"{kid}_space.json"
        cpp_path   = work / f"{kid}.cpp"
        space = json.loads(space_path.read_text())

        # Collect inputs from the dumped intermediates dir, in arg-index order.
        in_npys = sorted(
            inter.glob(f"{kid}_in_*.npy"),
            key=lambda p: int(p.stem.split("_in_")[-1]),
        )
        if not in_npys:
            sys.exit(f"phase 4: no dumped inputs for {kid} under {inter}")
        # v1 supports only single-output kernels; assert and read out_0.
        out0_npy = inter / f"{kid}_out_0.npy"
        if not out0_npy.exists():
            sys.exit(f"phase 4: missing {out0_npy}")

        shape_arg = _shape_arg_for_kernel(inter, kid, space)
        best_path = work / f"{kid}_best.json"
        profile_dir = work / f"{kid}_autotune_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            AUTOTUNER,
            "--space",   str(space_path),
            "--kernel",  str(cpp_path),
            "--inputs",  ",".join(str(p) for p in in_npys),
            "--expected", str(out0_npy),
            "--shape",   shape_arg,
            "--output",  str(best_path),
            "--profile-out", str(profile_dir),
            "--atol", str(args.atol),
            "--rtol", str(args.rtol),
        ]
        run(cmd)

        bc = json.loads(best_path.read_text())
        params = dict(bc.get("config", {}))
        params["_block_dim"] = int(bc.get("best", {}).get("block_dim", 1))
        tilings_best[kid] = params

    tilings_path = work / "tilings_best.json"
    tilings_path.write_text(json.dumps(tilings_best, indent=2))
    print(f"phase 4 OK → {tilings_path}")
    return tilings_path


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
    ap.add_argument("--max-phase", type=int, default=4,
                    help="Stop after this phase (1=outline, 2=codegen+compile, "
                         "3=default-build+dump, 4=autotune). Default: 4.")
    args = ap.parse_args()

    work = Path(args.workdir).absolute()
    work.mkdir(parents=True, exist_ok=True)

    groups = phase1_outline_or_emit_json(args, work)
    print(f"workdir: {work}")
    print(f"groups:  {groups}")
    if args.max_phase < 2:
        return

    network = NetworkJson.load(str(groups / "network.json"))
    artifacts = phase2_codegen_compile(work, groups, network)
    if args.max_phase < 3:
        return

    tilings_default, inter = phase3_default_build_and_dump(work, groups, network, artifacts, args)
    if args.max_phase < 4:
        return

    phase4_autotune(work, network, inter, args)


if __name__ == "__main__":
    main()
