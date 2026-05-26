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
import re
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
        # Step a0: recognize decomposed attention (bmm -> softmax -> bmm) and
        # layernorm (reduce/rsqrt) subgraphs and route them to aclnn
        # FlashAttentionScore / LayerNorm, then finalize the aclnn decls
        # (aclnn.kind -> aclnn.op/aclnn.layout) so the network-json emitter tags
        # the calls kind=aclnn. All passes are no-ops when the model has no such
        # subgraph. Runs pre-group-analysis on the rawest IR.
        recognized = work / "model_recognized.mlir"
        run([AFIR_OPT, "--recognize-attention", "--recognize-layernorm",
             "--aclnn-finalize-decl",
             args.input_linalg, "-o", str(recognized)])
        # Step a: --linalg-fold-unit-extent-dims + --canonicalize. Canonicalize
        # folds away identity-copy generics (linalg.generic { yield %in }, e.g.
        # the transposed-weight `.contiguous()` copies torch emits for nn.Linear)
        # BEFORE outlining — otherwise each becomes a degenerate device kernel
        # that --auto-fuse-codegen DCEs to an empty body (no tiling_infos →
        # PackTilingData failure).
        intermediate = work / "model_unit_folded.mlir"
        run([AFIR_OPT, "--linalg-fold-unit-extent-dims", "--canonicalize",
             str(recognized), "-o", str(intermediate)])
        # Step b: --auto-fuse-group-analysis + --auto-fuse-group-outline
        # Cube (matmul/batch_matmul) groups are routed to the aclnn matmul
        # fallback by default — the auto-fuse AscendC cube codegen isn't ready —
        # so disable Cube+Vector epilogue fusion to keep matmuls standalone.
        analysis = "--auto-fuse-group-analysis"
        if getattr(args, "enable_cube_fusion", False) is False:
            analysis += "=disable-cube-fusion=true"
        run([AFIR_OPT,
             analysis,
             f"--auto-fuse-group-outline=output-dir={groups}",
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


def _eval_ub_cost(space: dict, params: dict) -> int:
    """Evaluate UB cost under integer params.

    CannTranslation emits `ub_cost_bytes_exprs` as a list of per-buffer
    aligned-size expressions (pure +-*/; ceilDiv as ((a+b-1)/b)). The
    runtime peak is `2 * max(...)` (2x for input+output TBufs live
    concurrently). Returns 0 if the list is absent — callers should
    treat that as "unknown UB cost" and skip UB-aware pruning.
    """
    exprs = space.get("ub_cost_bytes_exprs") or []
    if not exprs:
        return 0
    peak = 0
    for expr in exprs:
        v = eval_block_dim({"block_dim_expr": expr}, params)
        if v > peak:
            peak = v
    return 2 * peak


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


def phase2_codegen_compile(work, groups, network, soc="Ascend910B1"):
    """For each ascendc kernel: --auto-fuse-codegen → -mlir-to-cann → compile.

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
        run([AFIR_OPT, str(src), "--auto-fuse-codegen", "-o", str(lowered)])
        run([AFIR_TRANSLATE, "-mlir-to-cann", str(lowered),
             "-o", str(cpp), f"--tiling-space-out={space}",
             f"--soc={soc}"])
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
            # Stage the per-variant tiling schema where HostLaunchHelper looks
            # for it (<artifacts>/<vname>/tiling_space.json). Without it the
            # helper falls back to alphabetical-name param order, which differs
            # from the kernel struct's mlir_index order and scrambles tiling
            # params whose chosen values differ. The _space.json's tiling_params
            # array is already in mlir_index order (TilingSchema reads that order).
            vspace = work / v["space_file"]
            if vspace.exists():
                shutil.copy(vspace, artifacts / vname / "tiling_space.json")
    print(f"phase 2 OK → {artifacts}")
    return artifacts


def _validate_shape_equalities(space, network, kid, runner_inputs):
    """Validate that user-passed input shapes satisfy the symbolic equalities
    derived from the kernel's linalg op semantics.

    `space["shape_equalities"]` is a list of groups; each group is a list of
    `[call_arg_index, dim]` pairs that share the same shape root symbol.  All
    input dim values within a group MUST resolve to the same integer.  Raises
    RuntimeError with a precise diagnostic on mismatch (before kernel launch,
    so the user sees the bug instead of garbage output).
    """
    groups = space.get("shape_equalities", [])
    if not groups:
        return
    for group in groups:
        # Resolve each (call_arg_index, dim) → integer via the existing walker.
        resolved = []
        for entry in group:
            if len(entry) != 2:
                continue
            call_idx, dim_idx = int(entry[0]), int(entry[1])
            shape = _resolve_kernel_input_shape(
                network, kid, call_idx, runner_inputs)
            if dim_idx >= len(shape):
                raise RuntimeError(
                    f"shape_equalities[{kid}]: call_arg_index={call_idx} "
                    f"dim={dim_idx} out of range for shape {tuple(shape)}")
            resolved.append((call_idx, dim_idx, int(shape[dim_idx])))
        # All members of a group must have the same value.
        vals = {v for _, _, v in resolved}
        if len(vals) > 1:
            details = ", ".join(
                f"arg{c}.dim{d}={v}" for c, d, v in resolved)
            raise RuntimeError(
                f"shape_equalities[{kid}]: members must agree but got "
                f"different values: {details}.  Symbolically these dims are "
                "the same; pass consistent input shapes.")


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


def _resolve_shape_via_schema(space, network, kid, shape_key, runner_inputs):
    """Resolve a shape_key string against the kernel's schema_args
    (loaded from <kernel>_space.json by phase 4).

    shape_key formats:
      - "arg<N>_dim<D>" where N is a network-arg index (the shape_key
        format emitted by CannTranslation when the kernel field is sourced
        from an input arg, OR the literal shape_expr entry for an output-
        sourced field whose expr happens to be 'argK_dimL').
      - numeric literal (e.g. "32") when the schema's shape_expr resolved
        to a static dim.  Returned as int(value) directly.

    Returns int (resolved dim size) or None if no schema_args is present
    in the space.json (caller raises — schema v2 is required).
    """
    import re
    sch = space.get("schema_args")
    if not sch:
        return None

    # Numeric literal shape_expr → just an int.
    try:
        return int(shape_key)
    except ValueError:
        pass

    m = re.match(r"^arg(\d+)_dim(\d+)$", shape_key)
    if not m:
        # Complex shape_expr (a*b+c, etc.) — runner doesn't evaluate yet.
        raise RuntimeError(
            f"network_runner: shape_key {shape_key!r} for kernel {kid!r} "
            "is not a simple argN_dimD form or numeric literal; complex "
            "shape_expr not yet supported in runner.")
    target_arg, dim_idx = int(m.group(1)), int(m.group(2))

    # The integer in shape_key is the schema's `call_arg_index` — the
    # kernel's coordinator-call operand position (== kernel.args[] index in
    # network.json).  Find the schema entry and resolve via the existing
    # legacy walker using its mlir_index (position in the kernel func
    # signature, == kernel.args[] index).
    for sa in sch:
        if sa.get("role") == "input" and sa.get("call_arg_index") == target_arg:
            kernel_arg_idx = sa.get("mlir_index", target_arg)
            shape = _resolve_kernel_input_shape(
                network, kid, kernel_arg_idx, runner_inputs)
            return int(shape[dim_idx])
    raise RuntimeError(
        f"network_runner: shape_key {shape_key!r} does not match any input "
        f"in kernel {kid!r}'s schema_args.")


def _shape_key_values_for_kernel(space, network, kid, runner_inputs):
    out = {}
    for p in space.get("tiling_params", []):
        sk = p.get("shape_key")
        if not sk or sk in out:
            continue
        val = _resolve_shape_via_schema(space, network, kid, sk, runner_inputs)
        if val is None:
            raise RuntimeError(
                f"network_runner: kernel {kid!r} has no schema_args in its "
                f"_space.json; cannot resolve shape_key {sk!r}.  Re-run "
                "auto-fuse-codegen to regenerate.")
        out[sk] = val
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
        # Validate user-passed shapes are consistent with the kernel's
        # symbolic-shape equalities BEFORE the kernel ever launches.  Catches
        # mismatched-shape inputs (e.g. a.dim0 != b.dim0 for an elementwise
        # `a+b` kernel) with a clear diagnostic instead of garbage output.
        _validate_shape_equalities(space, network, kid, args.inputs)
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
        # but not so large XBLOCK that no block actually writes output) AND that
        # keeps ub_cost_bytes_expr <= ub_budget_bytes (CannTranslation stamps
        # both from the SoC table + symbolic init_buffer sizes). The UB prune
        # is what prevents the TPipe bump-pointer allocator from silently
        # overflowing — pre-2026-05-14 we picked the largest unconditionally
        # and at R≥256 on dyn-bucketed-e2e that produced all-zero output.
        ub_budget = int(space.get("ub_budget_bytes", 0))
        block_dim_expr = space.get("block_dim_expr", "") or ""
        for p in space.get("tiling_params", []):
            if not p.get("fixed", False):
                name = p["name"]
                vals = p.get("values", []) or [16]
                # axis_extent_expr is the MULTICORE axis extent.  Two regimes
                # for using it to bound a tunable:
                #
                #   - Block-dim param (named in block_dim_expr, e.g. XBLOCK):
                #     hard cap to the extent.  If no candidate fits, fall back
                #     to [extent].
                #
                #   - Inner-axis tile params (XBLOCK_X_0 / XBLOCK_SUB / …):
                #     these may legitimately span an axis larger or smaller
                #     than the multicore axis (encoder XBLOCK_X_0 spans M
                #     while multicore is N), so do NOT shrink the candidate
                #     set when at least one value already fits.  BUT when
                #     *every* candidate exceeds the extent — which happens
                #     for dyn-bucketed-e2e at d0*d1=8 with the generic
                #     codegen sweep [16,32,64,128,256] — picking the largest
                #     produces uint32_t-underflowed tail offsets that write
                #     OOB and leave the caller's output buffer all-zero.
                #     Fall back to [extent] in that case.
                is_block_param = re.search(
                    r"\b" + re.escape(name) + r"\b", block_dim_expr) is not None
                if extent > 0:
                    capped = [v for v in vals if v <= extent]
                    if is_block_param:
                        vals = capped if capped else [extent]
                    elif not capped:
                        vals = [extent]
                pick = vals[-1]
                if ub_budget > 0 and space.get("ub_cost_bytes_exprs"):
                    trial = dict(params)
                    trial.update(shape_keys)
                    fits = []
                    for v in vals:
                        trial[p["name"]] = v
                        if 0 < _eval_ub_cost(space, trial) <= ub_budget:
                            fits.append(v)
                    if fits:
                        pick = fits[-1]
                params[p["name"]] = pick
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
        try:
            run(cmd)
        except subprocess.CalledProcessError:
            # The autotuner found no passing/profilable config for this kernel
            # (e.g. its search space excludes the one that works). The phase-3
            # default tiling is already verified correct, so fall back to it
            # rather than aborting the whole network for one untunable kernel.
            default_tilings = json.loads(
                (work / "tilings_default.json").read_text())
            if default_vkid in default_tilings:
                tilings_best[default_vkid] = default_tilings[default_vkid]
                print(f"phase 4: {kid} autotune found no config → "
                      f"falling back to default tiling")
                continue
            raise

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
        backend=args.backend,
    )

    # 3) Run; emit one --output per network output (mandatory: harness's
    # outputs[] is sized from --output count; mismatch → SIGSEGV at cleanup).
    # For --backend npu, HostLaunchHelper selects the real device via
    # NETWORK_RUNNER_BACKEND; ASCEND_DEVICE_ID picks the device (default 0).
    out_dir = work / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = [str(out_dir / f"out{i}.npy") for i in range(len(network.outputs))]
    cmd = [str(binary)]
    for p in args.inputs:
        cmd += ["--input", p]
    for p in out_paths:
        cmd += ["--output", p]
    run_env = None
    if args.backend == "npu":
        # Strip the simulator + devlib dirs from LD_LIBRARY_PATH for the device
        # binary. examples/env.sh prepends "SIM_LIB:BASE_LIB:DEV_LIB" (needed by
        # the sim phases 1-4), but for a real-device run two of those shadow the
        # driver libs the dlopen'd libruntime.so needs:
        #   - simulator/<soc>/lib  ships camodel npu_drv/stars/model_top stubs
        #   - devlib/linux/<arch>  ships a stub libascend_hal.so
        # Either ahead of /usr/local/Ascend/driver/lib64 (added by setenv.bash)
        # makes rtSetDevice resolve to a stub HAL → 107001 (INVALID_DEVICEID).
        # The working --case path never sources env.sh, so it gets the real
        # driver HAL; dropping both dirs here matches that.
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        ld = ":".join(p for p in ld.split(":")
                      if p and "/simulator/" not in p and "/devlib/" not in p)
        run_env = {**os.environ, "NETWORK_RUNNER_BACKEND": "npu",
                   "LD_LIBRARY_PATH": ld}
    run(cmd, env=run_env)

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
    ap.add_argument("--backend", choices=["sim", "npu"], default="sim",
                    help="Execution backend for the final run+verify (phase 5). "
                         "sim=CANN simulator (default); npu=real Ascend device "
                         "(ASCEND_DEVICE_ID, default 0). Phases 2-4 always use sim.")
    ap.add_argument("--enable-cube-fusion", action="store_true",
                    help="Allow Cube+Vector epilogue fusion (AscendC cube "
                         "codegen). Default: off — matmuls route to aclnn.")
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
    artifacts = phase2_codegen_compile(work, groups, network, soc=args.soc)
    if args.max_phase < 3:
        return

    tilings_default, inter = phase3_default_build_and_dump(work, groups, network, artifacts, args)
    if args.max_phase < 4:
        return

    # Accuracy-only mode: NETWORK_RUNNER_SKIP_AUTOTUNE skips the multi-round
    # autotuner and feeds phase 5 the default tilings from phase 3. Use this to
    # validate numerics fast without paying for performance search.
    if os.environ.get("NETWORK_RUNNER_SKIP_AUTOTUNE", "") not in ("", "0", "false", "no"):
        print("[phase4] NETWORK_RUNNER_SKIP_AUTOTUNE set — skipping autotune; "
              "using default tilings (accuracy-only, no perf tuning).")
        tilings_best_path = work / "tilings_default.json"
    else:
        tilings_best_path = phase4_autotune(work, network, inter, args)
    if args.max_phase < 5:
        return

    rc = phase5_final_run_verify(work, groups, artifacts, tilings_best_path, network, args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
