"""
pytest integration: the @torch_e2e_test decorator.

    @torch_e2e_test
    def test_my_model():
        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.relu(a * b + a)
        return Model(), [TensorSpec(("M", "N"), torch.float16),
                         TensorSpec(("M", "N"), torch.float16)]

This driver runs the *vector-plan* lowering path (afir-opt --vector-plan-codegen),
which carries the linalg-level symbolic shapes (afir.dim_symbols / afir.axis_extents /
afir.symbolic_shape arg-attrs) through to the AscendC kernel: de-duped TilingData
dim fields and a populated block_dim_expr in tiling_space.json.  See
docs/superpowers/specs/2026-05-12-mlir-shape-symbolization-design.md.
"""

import functools
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from torch2linalg import torch_to_linalg

REPO_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_ROOT = REPO_ROOT / "output" / "torch_e2e"


# ================================================================
# Public API
# ================================================================

def torch_e2e_test(func=None, *, verify_shapes: dict[str, int] | None = None):
    """Decorate a function returning (model, list[TensorSpec]).  Runs the full
    torch -> linalg -> vector-plan-codegen -> CANN kernel -> autotuner pipeline.

    verify_shapes maps dynamic-dim names to concrete sizes used in the numeric
    verification (default 64).
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper():
            model, specs = fn()
            test_name = fn.__name__
            work_dir = OUTPUT_ROOT / test_name
            if work_dir.exists():
                shutil.rmtree(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

            # 1. dynamic_shapes for torch.export
            dim_pool: dict = {}
            shapes_list = []
            has_dynamic = False
            for spec in specs:
                dims = spec.dynamic_dims(dim_pool)
                shapes_list.append(dims if dims else {})
                has_dynamic = has_dynamic or bool(dims)
            dynamic_shapes = tuple(shapes_list) if has_dynamic else None

            # 2. dummy trace inputs (shape/dtype only)
            trace_inputs = [spec.make_sample() for spec in specs]

            # 3. torch -> linalg
            print("\n[Stage 0] torch -> linalg MLIR")
            mlir_text = torch_to_linalg(model, trace_inputs, dynamic_shapes)
            (work_dir / "step0_linalg.mlir").write_text(mlir_text)

            # 4. MLIR pipeline -> kernel.cpp + tiling_space.json
            print("\n[Stage 0b-8] MLIR pipeline (vector-plan-codegen)")
            assert _run_mlir_pipeline(work_dir), "MLIR pipeline failed"

            # 5. concrete verify inputs + PyTorch reference
            verify_inputs = [spec.make_sample(verify_shapes or {}) for spec in specs]
            print("\n[Stage 9] reference data")
            for i, t in enumerate(verify_inputs):
                np.save(work_dir / f"input_{i}.npy", t.numpy())
            with torch.no_grad():
                expected = model.eval()(*verify_inputs)
            np.save(work_dir / "expected_0.npy", expected.numpy())
            print(f"  inputs={[tuple(t.shape) for t in verify_inputs]} expected={tuple(expected.shape)}")

            # 6. autotuner: compile + tune + verify
            print("\n[Stage 10] autotuner (compile + tune + verify)")
            tiling_space = work_dir / "step8_kernel.tiling_space.json"
            shape_str = _build_shape_str(tiling_space, verify_inputs)
            assert _run_autotuner(work_dir, len(verify_inputs), shape_str), \
                "autotuner verification failed"

            print(f"\n  PASS: {test_name}")

        return wrapper

    return decorator(func) if func is not None else decorator


# ================================================================
# MLIR pipeline (vector-plan path)
# ================================================================

def _find_tool(name: str) -> str | None:
    p = shutil.which(name)
    if p:
        return p
    default = REPO_ROOT / "build" / "bin" / name
    return str(default) if default.exists() else None


def _run_afir_opt(src: Path, dst: Path, passes: list[str], afir_opt: str) -> bool:
    proc = subprocess.run([afir_opt, *passes, str(src), "-o", str(dst)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  FAIL: afir-opt {' '.join(passes)}")
        print(f"  stderr: {proc.stderr[:800]}")
        return False
    return True


def _run_mlir_pipeline(work_dir: Path) -> bool:
    """step0b (fold-unit-extent-dims) -> --vector-plan-codegen -> strip cf.assert
    -> afir-translate -mlir-to-cann.  The vector-plan-codegen pipeline already
    runs afir-symbolize-shapes before tile-fuse, so the symbolic shapes reach
    the kernel."""
    linalg_path = work_dir / "step0_linalg.mlir"
    assert linalg_path.exists()

    afir_opt = _find_tool("afir-opt")
    if afir_opt is None:
        print("  afir-opt not on PATH; skip")
        return False

    step0b = work_dir / "step0b_folded.mlir"
    print("  [step0b] --linalg-fold-unit-extent-dims")
    if not _run_afir_opt(linalg_path, step0b, ["--linalg-fold-unit-extent-dims"], afir_opt):
        return False

    step7 = work_dir / "step7_cann.mlir"
    print("  [step1-7] --vector-plan-codegen")
    # Also dump the IR after the interesting pipeline stages into stages.mlir
    # (stderr of --mlir-print-ir-after); handy for inspecting the symbolic-shape
    # flow without re-running by hand.
    stage_passes = ("afir-symbolize-shapes,vector-plan-tile-fuse,one-shot-bufferize,"
                    "linalg-to-ascendc,ascendc-parallelize,ascendc-pack-tiling-data,"
                    "canonicalize-cann-signature")
    proc = subprocess.run([afir_opt, "--vector-plan-codegen",
                           f"--mlir-print-ir-after={stage_passes}",
                           str(step0b), "-o", str(step7)],
                          capture_output=True, text=True)
    (work_dir / "stages.mlir").write_text(proc.stderr)
    if proc.returncode != 0:
        print("  FAIL: afir-opt --vector-plan-codegen")
        print(f"  stderr tail: {proc.stderr[-800:]}")
        return False

    # afir-translate cannot print cf.assert (dynamic-broadcast checks); drop them.
    content = step7.read_text()
    if "cf.assert" in content:
        content = re.sub(r"^\s*cf\.assert\b.*\n", "", content, flags=re.MULTILINE)
        step7.write_text(content)
        print("  [step7c] stripped cf.assert")

    afir_translate = _find_tool("afir-translate")
    if afir_translate is None:
        print("  afir-translate not on PATH; skip codegen")
        return False
    step8_cpp = work_dir / "step8_kernel.cpp"
    step8_tiling = work_dir / "step8_kernel.tiling_space.json"
    print("  [step8] afir-translate -mlir-to-cann")
    proc = subprocess.run([afir_translate, "-mlir-to-cann", str(step7),
                           "-o", str(step8_cpp),
                           "--tiling-space-out", str(step8_tiling)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print("  FAIL: afir-translate -mlir-to-cann")
        print(f"  stderr: {proc.stderr[:800]}")
        return False
    if step8_tiling.exists():
        print(f"  tiling_space.json: {step8_tiling.read_text().strip()[:400]}")
    return True


# ================================================================
# autotuner
# ================================================================

def _patch_tiling_space(path: Path) -> None:
    """Fill empty search ranges for tunable params (afir-translate emits a
    skeleton).  block_dim_expr is now populated by afir-translate (C2); only
    fall back if it's still empty (multi-op kernels)."""
    ts = json.loads(path.read_text())
    patched = False
    for p in ts.get("tiling_params", []):
        if not p.get("fixed", False) and not p.get("values"):
            p["values"] = [16, 32, 64]
            patched = True
    if not ts.get("block_dim_expr"):
        tb = next((p["name"] for p in ts.get("tiling_params", [])
                   if not p.get("fixed", False)), None)
        dk = next((p["shape_key"] for p in ts.get("tiling_params", [])
                   if p.get("fixed", False) and p.get("shape_key")), None)
        if tb and dk:
            ts["block_dim_expr"] = f"ceil({dk}/{tb})"
            patched = True
            print(f"  [patch] block_dim_expr = {ts['block_dim_expr']}")
    if patched:
        path.write_text(json.dumps(ts, indent=2) + "\n")


def _build_shape_str(tiling_space_path: Path, verify_inputs: list[torch.Tensor]) -> str:
    """Build the autotuner --shape arg from the JSON's fixed (shape_key) params."""
    ts = json.loads(tiling_space_path.read_text())
    parts = []
    for p in ts.get("tiling_params", []):
        if not p.get("fixed", False):
            continue
        key = p.get("shape_key")
        if not key:
            continue
        arg_idx = int(key.split("_")[0].removeprefix("arg"))
        dim_idx = int(key.split("_")[1].removeprefix("dim"))
        parts.append(f"{key}={verify_inputs[arg_idx].shape[dim_idx]}")
    return ",".join(parts)


def _cleanup_sim_dumps() -> None:
    cwd = Path.cwd()
    for pat in ("*.dump", "*.vcd", "profile_*.toml",
                "ffts_verify_log*.log", "core*_summary_log"):
        for f in cwd.glob(pat):
            f.unlink()


def _run_autotuner(work_dir: Path, num_inputs: int, shape_str: str) -> bool:
    for i in range(num_inputs):
        assert (work_dir / f"input_{i}.npy").exists()
    expected_npy = work_dir / "expected_0.npy"
    assert expected_npy.exists()
    assert (work_dir / "step8_kernel.cpp").exists()
    ts_path = work_dir / "step8_kernel.tiling_space.json"
    assert ts_path.exists()
    assert shape_str, "shape_str empty"

    _patch_tiling_space(ts_path)

    autotuner = _find_tool("autotuner")
    if autotuner is None:
        print("  autotuner not on PATH; skip")
        return False

    input_npys = ",".join(str(work_dir / f"input_{i}.npy") for i in range(num_inputs))
    cmd = [autotuner,
           "--space", str(ts_path),
           "--kernel", str(work_dir / "step8_kernel.cpp"),
           "--inputs", input_npys,
           "--expected", str(expected_npy),
           "--shape", shape_str,
           "--output", str(work_dir / "best_config.json"),
           "--profile-out", str(work_dir / "perf_out")]
    print(f"  cmd: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    _cleanup_sim_dumps()
    if proc.returncode != 0:
        print("  FAIL: autotuner")
        print(f"  stdout: {proc.stdout[-800:]}")
        print(f"  stderr: {proc.stderr[:800]}")
        return False
    print(f"  autotuner stdout:\n{proc.stdout[-400:]}")
    return True
