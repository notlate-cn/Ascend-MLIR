#!/usr/bin/env python3
"""Network runner: orchestrate mixed AscendC+aclnn network compilation & execution.

See docs/superpowers/specs/2026-05-13-network-runner-mixed-cpu-sim-design.md.

Phases 1-5 are implemented.

Note: Phases 2-5 (codegen + compile + sim run + autotune + final verify) require the simulator
LD_LIBRARY_PATH. Run `source examples/env.sh` (or `source examples/env_gser.sh`) before executing.
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


def _eval_axis_extent(space: dict, params: dict) -> int:
    """Evaluate space['axis_extent_expr'] under integer params.

    Same grammar as eval_block_dim. Returns 0 (sentinel "unknown") if expr is
    missing or empty — callers should skip capping in that case.
    """
    expr = (space.get("axis_extent_expr") or "").strip()
    if not expr:
        return 0
    # Wrap in a fake space so we can reuse eval_block_dim's grammar handler.
    return eval_block_dim({"block_dim_expr": expr}, params)


def _read_family(work, kid):
    """Read <kid>_family.json (emitted by CannTranslation P2). Returns the
    parsed dict, or a synthetic single-variant family when the file is missing
    (defensive — TileFuse renames produce family.json for every codegen run)."""
    p = work / f"{kid}_family.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"kernel_id": kid,
            "variants": [{"id": "v0", "func_name": f"{kid}__v0",
                          "space_file": f"{kid}__v0_space.json"}]}


def _variant_kernel_name(work, kid, picked=None):
    """Resolve the variant kernel name for `kid`.

    `picked` is the variant id chosen by autotune (from <kid>_best.json's
    "variant" field). When unset, returns the first (default) variant from
    family.json — which P1a guarantees to be v0.
    """
    family = _read_family(work, kid)
    if picked:
        for v in family["variants"]:
            if v["id"] == picked:
                return v["func_name"]
    return family["variants"][0]["func_name"]


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
        space = work / f"{kid}_space.json"  # legacy back-compat (P2 also writes per-variant)
        run([AFIR_OPT, str(src), "--vector-plan-codegen", "-o", str(lowered)])
        run([AFIR_TRANSLATE, "-mlir-to-cann", str(lowered),
             "-o", str(cpp), f"--tiling-space-out={space}"])
        # Per-variant compile. The .cpp contains all variants' kernel symbols;
        # each runtime-session call picks one via --name and produces a
        # variant-specific artifact dir.
        family = _read_family(work, kid)
        for v in family["variants"]:
            vname = v["func_name"]
            run([RUNTIME_SESSION,
                 "--kernel", str(cpp),
                 "--kernel-kind", "vec",
                 "--output", str(artifacts / vname),
                 "--name", vname])
    print(f"phase 2 OK → {artifacts}")
    return artifacts


def _resolve_kernel_input_shape(network, kid, arg_idx, runner_inputs):
    """Get the runtime shape of `kid`'s `arg_idx`-th input.

    network args descriptor `from:input` → load the corresponding runner npy.
    `from:kernel` → assume the upstream kernel's result has the same shape as
    its first input (elementwise convention; works for v1 examples).
    """
    import numpy as np
    k = network.kernel_by_id(kid)
    arg = k["args"][arg_idx]
    # An arg propagated through tensor.expand_shape / collapse_shape carries an
    # explicit shape override (the rank the kernel actually consumes), distinct
    # from the underlying buffer's shape at the network input.  Prefer that.
    if "shape" in arg:
        return tuple(arg["shape"])
    if arg["from"] == "input":
        # find this input's index in network.inputs
        for i, inp in enumerate(network.inputs):
            if inp["name"] == arg["name"]:
                return np.load(runner_inputs[i]).shape
        raise RuntimeError(f"input {arg['name']} not found in network.inputs")
    # from:kernel — recurse into upstream's first input
    return _resolve_kernel_input_shape(network, arg["kernel"], 0, runner_inputs)


def _shape_key_values_for_kernel(space, network, kid, runner_inputs):
    """Return {shape_key: int} for every shape_key referenced in `space`."""
    import re
    out = {}
    for p in space.get("tiling_params", []):
        sk = p.get("shape_key")
        if not sk or sk in out:
            continue
        m = re.match(r"^arg(\d+)_dim(\d+)$", sk)
        if not m:
            continue
        arg_idx, dim_idx = int(m.group(1)), int(m.group(2))
        shape = _resolve_kernel_input_shape(network, kid, arg_idx, runner_inputs)
        out[sk] = int(shape[dim_idx])
    return out


def phase3_default_build_and_dump(work, groups, network, artifacts, args):
    """Build network_host.cpp with default tilings, g++ link, run, dump intermediates.

    Steps:
      1. Build tilings_default.json from each kernel's _space.json (tunable params
         get their first 'values' entry; shape_key fixed params resolve from the
         actual runner --inputs shapes).
      2. Generate network_host_default.cpp via aclnn-backend.
      3. g++ link against libAscendCRuntime + CANN libs.
      4. Run with --dump-intermediates DIR.

    Returns (tilings_path, intermediates_dir).
    """
    # 1) Build tilings_default.json
    default_tilings: dict = {}
    for k in network.ascendc_kernels():
        kid = k["id"]
        # For default, take the first variant from family.json (v0 by P1a
        # convention). When P1b enables real multi-variant codegen, default
        # picker can still safely pick v0 — the autotuner will revisit.
        family = _read_family(work, kid)
        default_variant = family["variants"][0]
        space_path = work / default_variant["space_file"]
        vkid = default_variant["func_name"]
        space = json.loads(space_path.read_text())
        params: dict = {}
        # Shape-keyed fixed params first: dim_arg*_* — resolve from runner inputs.
        shape_keys = _shape_key_values_for_kernel(space, network, kid, args.inputs)
        for p in space.get("tiling_params", []):
            sk = p.get("shape_key")
            if p.get("fixed", False) and sk and sk in shape_keys:
                params[p["name"]] = shape_keys[sk]
        # Evaluate axis_extent_expr (set by CannTranslation from the per-kernel
        # SymExpr) under this invocation's shape_keys to get the total tile-axis
        # extent. Used to cap tunable XBLOCK-like candidates: a value larger than
        # the runtime extent is invalid (kernel doesn't write output → caller
        # buffer reads as zeros).
        extent = _eval_axis_extent(space, shape_keys)
        # Tunable params: "fixed": false. For default, pick the LARGEST candidate
        # that is also <= extent (camodel sims ~32 cores; we want block_dim small
        # but not so large XBLOCK that no block actually writes output).
        for p in space.get("tiling_params", []):
            if not p.get("fixed", False):
                vals = p.get("values", []) or [16]
                if extent > 0:
                    capped = [v for v in vals if v <= extent]
                    vals = capped if capped else [extent]
                params[p["name"]] = vals[-1]
        # eval_block_dim grammar uses shape_key names directly (e.g. arg0_dim0),
        # not the param names — register the shape_key → value mapping for it.
        block_dim_params = dict(params)
        block_dim_params.update(shape_keys)
        params["_block_dim"] = eval_block_dim(space, block_dim_params)
        # Key by variant kernel name so HostLaunchHelper's tilings lookup
        # (which uses the symbol name P4's aclnn-backend emits) succeeds.
        default_tilings[vkid] = params

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


def _shape_arg_for_kernel(inter: Path, kid: str, space: dict,
                            network=None, kernel_id_in_network: str = None,
                            runner_inputs=None) -> str:
    """Build the --shape KEY=VAL,... string for autotuner.

    shape_key format is `arg<i>_dim<j>`.  Prefer the network.json descriptor
    (which honors expand_shape / collapse_shape rank overrides from
    SplitRCoreGroup) when available; fall back to reading the dumped npy.
    """
    import numpy as np
    import re
    parts = []
    for key in _shape_keys_needed(space):
        m = re.match(r"^arg(\d+)_dim(\d+)$", key)
        if not m:
            sys.exit(f"phase 4: unsupported shape_key format: {key}")
        arg_idx, dim_idx = int(m.group(1)), int(m.group(2))
        shape = None
        if network is not None and kernel_id_in_network is not None \
                and runner_inputs is not None:
            try:
                shape = _resolve_kernel_input_shape(
                    network, kernel_id_in_network, arg_idx, runner_inputs)
            except Exception:
                shape = None
        if shape is None:
            npy = inter / f"{kid}_in_{arg_idx}.npy"
            if not npy.exists():
                sys.exit(f"phase 4: missing dumped input for shape_key {key}: {npy}")
            shape = np.load(npy).shape
        if dim_idx >= len(shape):
            sys.exit(f"phase 4: shape_key {key} dim out of range for shape {shape}")
        parts.append(f"{key}={shape[dim_idx]}")
    return ",".join(parts)


def phase4_autotune(work, network, inter, args):
    """Run autotuner per ascendc kernel family; aggregate tilings_best.json.

    P5: switched from per-variant `--space` to per-family `--family` mode.
    Each family's autotuner call loops over variants internally and picks the
    cross-variant best; we read the winning variant name from best.json's
    `variant` field and key tilings_best by the variant kernel name.
    """
    tilings_best: dict = {}
    for k in network.ascendc_kernels():
        kid = k["id"]
        cpp_path = work / f"{kid}.cpp"
        family_path = work / f"{kid}_family.json"

        # Use the default-variant's space.json (v0) to derive shape args + the
        # extent filter; all variants in a family share the same input shapes
        # and the same axis extent.
        family = _read_family(work, kid)
        default_variant = family["variants"][0]
        default_vkid = default_variant["func_name"]
        space_path = work / default_variant["space_file"]
        space = json.loads(space_path.read_text())

        # Collect inputs from the dumped intermediates dir (named after the
        # variant kernel — P4 emits hostLaunch with the variant suffix, so
        # HostLaunchHelper dumps `<vkid>_in_*.npy`).
        in_npys = sorted(
            inter.glob(f"{default_vkid}_in_*.npy"),
            key=lambda p: int(p.stem.split("_in_")[-1]),
        )
        if not in_npys:
            sys.exit(f"phase 4: no dumped inputs for {default_vkid} under {inter}")
        out0_npy = inter / f"{default_vkid}_out_0.npy"
        if not out0_npy.exists():
            sys.exit(f"phase 4: missing {out0_npy}")

        shape_arg = _shape_arg_for_kernel(inter, default_vkid, space,
                                            network=network,
                                            kernel_id_in_network=kid,
                                            runner_inputs=args.inputs)
        best_path = work / f"{kid}_best.json"
        profile_dir = work / f"{kid}_autotune_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Filter tunable candidates by the runtime extent (write filtered
        # per-variant space.json files so --family-mode autotuner picks them
        # up). Currently N=1 so only v0 is filtered; P1b widens the loop.
        shape_keys = {kv.split("=")[0]: int(kv.split("=")[1])
                       for kv in shape_arg.split(",") if "=" in kv}
        extent = _eval_axis_extent(space, shape_keys)
        if extent > 0:
            for v in family["variants"]:
                vspace_path = work / v["space_file"]
                vspace = json.loads(vspace_path.read_text())
                for p in vspace.get("tiling_params", []):
                    if not p.get("fixed", False) and "values" in p:
                        capped = [val for val in p["values"] if val <= extent]
                        p["values"] = capped if capped else [extent]
                vspace_path.write_text(json.dumps(vspace, indent=2))

        cmd = [
            AUTOTUNER,
            "--family", str(family_path),
            "--kernel", str(cpp_path),
            "--inputs", ",".join(str(p) for p in in_npys),
            "--expected", str(out0_npy),
            "--shape", shape_arg,
            "--output", str(best_path),
            "--profile-out", str(profile_dir),
            "--atol", str(args.atol),
            "--rtol", str(args.rtol),
        ]
        run(cmd)

        bc = json.loads(best_path.read_text())
        params = dict(bc.get("config", {}))
        params["_block_dim"] = int(bc.get("best", {}).get("block_dim", 1))
        # Key by the winning variant's kernel name. Falls back to v0 when
        # autotuner output is missing the field (shouldn't happen in family
        # mode).
        picked = bc.get("variant", "v0")
        vkid = _variant_kernel_name(work, kid, picked=picked)
        tilings_best[vkid] = params

    tilings_path = work / "tilings_best.json"
    tilings_path.write_text(json.dumps(tilings_best, indent=2))
    print(f"phase 4 OK → {tilings_path}")
    return tilings_path


def phase5_final_run_verify(work, groups, artifacts, tilings_best_path, network, args):
    """Re-emit host C++ with best tilings, build, run, compare to --expected.

    Returns 0 if all outputs PASS, non-zero on any FAIL.
    """
    import numpy as np
    from runner_utils.build_host import link_host

    # 1) Re-generate host with best tilings.
    host_cpp = work / "network_host.cpp"
    run([
        ACLNN_BACKEND,
        "--input",           str(groups / "network.mlir"),
        "--output",          str(host_cpp),
        "--tilings",         str(tilings_best_path),
        "--kernel-binaries", str(artifacts),
    ])

    # 2) g++ link.
    harness_cpp = REPO / "python/runner_utils/harness.cpp"
    binary = work / "network_test"
    has_aclnn = bool(network.aclnn_kernels())
    cann_home = os.environ.get("ASCEND_HOME_PATH", "/home/gser/Ascend/cann")
    link_host(
        host_cpp, harness_cpp, binary, REPO,
        cann_home=cann_home,
        soc=args.soc,
        has_aclnn_ops=has_aclnn,
    )

    # 3) Run; emit one --output per network output (mandatory: harness's
    # outputs[] is sized from --output count; mismatch → SIGSEGV at cleanup).
    out_dir = work / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = [str(out_dir / f"out{i}.npy") for i in range(len(network.outputs))]
    cmd = [str(binary)]
    for p in args.inputs:
        cmd += ["--input", p]
    for p in out_paths:
        cmd += ["--output", p]
    run(cmd)

    # 4) Compare each output to --expected with atol/rtol.
    if len(args.expected) != len(out_paths):
        sys.exit(f"phase 5: --expected count ({len(args.expected)}) != "
                 f"network output count ({len(out_paths)})")
    fail = False
    for i, (got, want) in enumerate(zip(out_paths, args.expected)):
        a = np.load(got).astype(np.float32)
        b = np.load(want).astype(np.float32)
        if a.shape != b.shape:
            print(f"network.output[{i}]: SHAPE MISMATCH got={a.shape} want={b.shape}  FAIL")
            fail = True
            continue
        max_diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        ok = np.allclose(a, b, atol=args.atol, rtol=args.rtol, equal_nan=False)
        print(f"network.output[{i}]: max_diff={max_diff:.4g}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            fail = True
    return 0 if not fail else 1


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
    ap.add_argument("--max-phase", type=int, default=5,
                    help="Stop after this phase (1=outline, 2=codegen+compile, "
                         "3=default-build+dump, 4=autotune, 5=final-build+verify). Default: 5.")
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

    tilings_best_path = phase4_autotune(work, network, inter, args)
    if args.max_phase < 5:
        return

    rc = phase5_final_run_verify(work, groups, artifacts, tilings_best_path, network, args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
