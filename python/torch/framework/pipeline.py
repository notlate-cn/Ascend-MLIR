"""
pytest 集成：torch_e2e_test 装饰器

用法：
    @torch_e2e_test
    def test_my_model():
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.relu().sum(dim=1)
        return Model(), [TensorSpec((None, None), torch.float16)]
"""

import glob
import re
import subprocess
import shutil
import functools
from pathlib import Path

import json
import numpy as np
import torch

from torch2linalg import torch_to_linalg

REPO_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_ROOT = REPO_ROOT / "output" / "torch_e2e"


# ================================================================
# Public API: torch_e2e_test decorator
# ================================================================

def torch_e2e_test(func=None, *, verify_shapes: dict[str, int] | None = None):
    """
    pytest 装饰器：定义一个 torch → NPU e2e 测试。

    被装饰函数应返回 (model, specs: list[TensorSpec])。
    装饰器执行完整 pipeline 并验证数值正确性。

    Args:
        verify_shapes: 验证阶段使用的具体维度大小，如 {"M": 64, "N": 128}。
                       未指定的动态维度默认为 64。
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

            # ── 编译阶段：torch → linalg → kernel ──

            # 1. 构建 dynamic_shapes（torch.export 要求）
            dim_pool = {}
            shapes_list = []
            has_dynamic = False
            for spec in specs:
                dims = spec.dynamic_dims(dim_pool)
                if dims:
                    has_dynamic = True
                    shapes_list.append(dims)
                else:
                    shapes_list.append({})
            dynamic_shapes = tuple(shapes_list) if has_dynamic else None

            # 2. 生成 dummy tensor 用于 trace（值无所谓，只需 shape/dtype）
            trace_inputs = [spec.make_sample() for spec in specs]

            # 3. torch → linalg
            print(f"\n[Stage 0] torch → linalg MLIR")
            mlir_text = torch_to_linalg(model, trace_inputs, dynamic_shapes)
            linalg_path = work_dir / "step0_linalg.mlir"
            linalg_path.write_text(mlir_text)
            print(f"  输出: {linalg_path}")

            # 4. MLIR pipeline (stage 0b-8)
            print(f"\n[Stage 0b-8] MLIR pipeline")
            assert _run_mlir_pipeline(work_dir), "MLIR pipeline 失败"

            # ── 验证阶段：生成测试数据 → autotuner 验证 ──

            # 5. 用 verify_shapes 实例化具体 tensor
            actual_shapes = verify_shapes or {}
            verify_inputs = [spec.make_sample(actual_shapes) for spec in specs]

            # 6. PyTorch reference run → 存 npy
            print(f"\n[Stage 9] 生成 reference data")
            model_eval = model.eval()
            for i, tensor in enumerate(verify_inputs):
                np.save(work_dir / f"input_{i}.npy", tensor.numpy())
            with torch.no_grad():
                expected = model_eval(*verify_inputs)
            np.save(work_dir / "expected_0.npy", expected.numpy())
            print(f"  inputs: {[t.shape for t in verify_inputs]}")
            print(f"  expected: {expected.shape}")

            # 7. Autotuner (compile + tune + verify)
            print(f"\n[Stage 10] Autotuner (compile + tune + verify)")
            tiling_space = work_dir / "step8_kernel.tiling_space.json"
            shape_str = _build_shape_str(tiling_space, verify_inputs)
            assert _run_autotuner(work_dir, len(verify_inputs), shape_str), \
                "Autotuner 验证失败"

            print(f"\n  ✓ 测试通过: {test_name}")

        return wrapper

    # 支持 @torch_e2e_test 和 @torch_e2e_test(...) 两种用法
    if func is not None:
        return decorator(func)
    return decorator


# ================================================================
# Helpers
# ================================================================

def _npy_to_txt(npy_path: Path, txt_path: Path) -> None:
    """将 .npy 文件转为人类可读的 .txt（与 examples/build_e2e 格式一致）。"""
    arr = np.load(npy_path)
    with open(txt_path, "w") as f:
        f.write(f"# shape: {' '.join(str(d) for d in arr.shape)}\n")
        for val in arr.flat:
            f.write(f"{val:.4f}\n")


def _patch_tiling_space(path: Path) -> None:
    """为 tiling_space.json 填充缺失的搜索范围和 block_dim_expr。

    TODO: afir-translate -mlir-to-cann 应生成合理的 min/max/step 或 values，
          以及 block_dim_expr，届时可移除此 workaround。
    """
    ts = json.loads(path.read_text())
    patched = False

    # 填充空 values 的搜索参数
    for p in ts.get("tiling_params", []):
        if not p.get("fixed", False) and not p.get("values"):
            p["values"] = [16, 32, 64]
            patched = True

    # 填充 block_dim_expr: 找到第一个非 fixed 参数（核间 tiling）
    # 和它对应的 shape 维度（第一个 fixed 参数），生成 ceil 表达式
    if not ts.get("block_dim_expr"):
        tb_param = None
        dim_key = None
        for p in ts.get("tiling_params", []):
            if not p.get("fixed", False) and tb_param is None:
                tb_param = p["name"]
            if p.get("fixed", False) and p.get("shape_key") and dim_key is None:
                dim_key = p["shape_key"]
        if tb_param and dim_key:
            ts["block_dim_expr"] = f"ceil({dim_key}/{tb_param})"
            patched = True
            print(f"  [patch] block_dim_expr = {ts['block_dim_expr']}")

    if patched:
        path.write_text(json.dumps(ts, indent=2) + "\n")


def _build_shape_str(tiling_space_path: Path,
                     verify_inputs: list[torch.Tensor]) -> str:
    """从 tiling_space.json 的 shape_key 构建 autotuner --shape 参数。

    读取所有 fixed=true 参数的 shape_key（如 "arg0_dim0"），
    从 verify_inputs 的实际 shape 中查找对应值。
    """
    ts = json.loads(tiling_space_path.read_text())
    parts = []
    for p in ts.get("tiling_params", []):
        if not p.get("fixed", False):
            continue
        key = p.get("shape_key")
        if not key:
            continue
        # shape_key 格式: "argN_dimD"
        arg_idx = int(key.split("_")[0].removeprefix("arg"))
        dim_idx = int(key.split("_")[1].removeprefix("dim"))
        size = verify_inputs[arg_idx].shape[dim_idx]
        parts.append(f"{key}={size}")
    return ",".join(parts)


def _find_tool(name: str) -> str | None:
    """查找工具路径，优先 PATH，其次 build/bin/"""
    path = shutil.which(name)
    if path:
        return path
    default = REPO_ROOT / "build" / "bin" / name
    if default.exists():
        return str(default)
    return None


def _run_afir_opt(input_path: Path, output_path: Path, passes: list[str],
                  afir_opt: str) -> bool:
    """运行 afir-opt，成功返回 True"""
    cmd = [afir_opt] + passes + [str(input_path), "-o", str(output_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  失败: {' '.join(passes)}")
        print(f"  stderr: {proc.stderr[:500]}")
        return False
    return True


def _cleanup_sim_dumps() -> None:
    """清理模拟器运行产生的 dump/log 文件。"""
    cwd = Path.cwd()
    for pattern in ("*.dump", "*.vcd", "profile_*.toml",
                    "ffts_verify_log*.log", "core*_summary_log"):
        for f in cwd.glob(pattern):
            f.unlink()


# ================================================================
# MLIR Pipeline: stage 1-8
# ================================================================

def _generate_transform_script(fused_path: Path, work_dir: Path) -> Path | None:
    """
    读取 step1 fused IR，注入通用 transform 序列，生成 step2_transform.mlir。

    适用场景：单个 linalg.generic（fusion 后最常见的情况）。
    切分策略：沿第一个 parallel 维度做两级 tile（TB 核间 + Tb UB批次），
    其余维度不切（tile_size=0）。
    """
    content = fused_path.read_text()

    match = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', content)
    if not match:
        print(f"  [transform-gen] 未找到 linalg.generic iterator_types，跳过")
        return None

    iters = [s.strip().strip('"') for s in match.group(1).split(",")]
    ndims = len(iters)

    par_idx = next((i for i, t in enumerate(iters) if t == "parallel"), None)
    if par_idx is None:
        print(f"  [transform-gen] 无 parallel 维度，跳过 tiling")
        return None

    def _tile_sizes(param_name: str) -> str:
        sizes = ["0"] * ndims
        sizes[par_idx] = f"%{param_name}"
        return "[" + ", ".join(sizes) + "]"

    tb_tile = _tile_sizes("tb_m")
    tb_inner_tile = _tile_sizes("tb_inner_m")

    new_content = content.replace(
        "module {",
        "module attributes {transform.with_named_sequence} {",
        1,
    )

    transform_seq = f"""
  // ═══════════ 自动生成的 Transform 调度脚本 ═══════════
  transform.named_sequence @__transform_main(
      %root : !transform.any_op {{transform.readonly}}
  ) {{
    %func = transform.structured.match ops{{["func.func"]}} in %root
        : (!transform.any_op) -> !transform.any_op
    %func_new, %tb_m, %tb_inner_m =
        transform.func.add_index_args %func, 2
            : (!transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %generic = transform.structured.match ops{{["linalg.generic"]}} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %tiled_tb, %loop_tb =
        transform.structured.tile_using_for %generic
            tile_sizes {tb_tile}
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    %true_param = transform.param.constant true -> !transform.any_param
    transform.annotate %loop_tb "ascendc.parallel"
        = %true_param : !transform.any_op, !transform.any_param

    %tiled_inner, %loop_inner =
        transform.structured.tile_using_for %tiled_tb
            tile_sizes {tb_inner_tile}
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    %prologue = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %epilogue = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    transform.annotate %loop_inner "ascendc.prologue"
        = %prologue : !transform.any_op, !transform.any_param
    transform.annotate %loop_inner "ascendc.epilogue"
        = %epilogue : !transform.any_op, !transform.any_param

    %vector_unit = transform.param.constant "AiCore.Vector" -> !transform.any_param
    transform.annotate %tiled_inner "ascendc.unit"
        = %vector_unit : !transform.any_op, !transform.any_param

    transform.yield
  }}
"""

    last_brace = new_content.rfind("}")
    new_content = new_content[:last_brace] + transform_seq + "}\n"

    out_path = work_dir / "step2_transform.mlir"
    out_path.write_text(new_content)
    print(f"  [transform-gen] iterator_types={iters}, tile parallel dim d{par_idx}")
    return out_path


def _run_mlir_pipeline(work_dir: Path) -> bool:
    """
    运行 MLIR pipeline（stage 1-8）:
      step0b: --linalg-fold-unit-extent-dims
      step1: --linalg-fuse-elementwise-ops
      step2: --ascend-normalize
      step3: --ascend-kernelize
      step4: --ascend-schedule
      step5: --ascend-realize
      step6: --ascend-compute-lower
      step7: --ascend-parallelize / --ascend-prepare-for-emit
      step7b: --ascend-canonicalize-cann-signature
      step8: afir-translate -mlir-to-cann
    """
    # 入口校验
    linalg_path = work_dir / "step0_linalg.mlir"
    assert linalg_path.exists(), f"缺少 {linalg_path}"

    afir_opt = _find_tool("afir-opt")
    if afir_opt is None:
        print("  afir-opt 不在 PATH 中，跳过")
        return False

    # ── Stage 0b: Fold unit-extent dims (消除 expand_shape 等) ──
    step0b = work_dir / "step0b_folded.mlir"
    print(f"  [step0b] --linalg-fold-unit-extent-dims")
    if not _run_afir_opt(linalg_path, step0b,
                         ["--linalg-fold-unit-extent-dims"], afir_opt):
        return False

    # ── Stage 1: Fusion ──
    step1 = work_dir / "step1_fused.mlir"
    print(f"  [step1] --linalg-fuse-elementwise-ops")
    if not _run_afir_opt(step0b, step1,
                         ["--linalg-fuse-elementwise-ops"], afir_opt):
        return False

    # ── Stage 2: Ascend Normalize ──
    step2 = work_dir / "step2_normalized.mlir"
    print(f"  [step2] --ascend-normalize")
    if not _run_afir_opt(step1, step2, ["--ascend-normalize"], afir_opt):
        return False

    # ── Stage 3: Ascend Kernelize ──
    step3 = work_dir / "step3_kernelized.mlir"
    print(f"  [step3] --ascend-kernelize")
    if not _run_afir_opt(step2, step3, ["--ascend-kernelize"], afir_opt):
        return False

    # ── Stage 4: Ascend Schedule ──
    step4 = work_dir / "step4_scheduled.mlir"
    print(f"  [step4] --ascend-schedule")
    if not _run_afir_opt(
            step3, step4,
            ["--ascend-schedule=target-tile-policy=legacy-default"],
            afir_opt):
        return False

    # ── Stage 5: Ascend Realize ──
    step5 = work_dir / "step5_realized.mlir"
    print(f"  [step5] --ascend-realize")
    if not _run_afir_opt(
            step4, step5,
            ["--ascend-realize=materialization-mode=memory-space-annotate"],
            afir_opt):
        return False

    # ── Stage 6: Ascend Compute Lower ──
    step6 = work_dir / "step6_ascendc.mlir"
    print(f"  [step6] --ascend-compute-lower")
    if not _run_afir_opt(step5, step6,
                         ["--ascend-compute-lower", "--canonicalize", "--cse"],
                         afir_opt):
        return False

    # ── Stage 7: Parallelize + Prepare For Emit ──
    step7 = work_dir / "step7_kernel.mlir"
    print(f"  [step7] --ascend-parallelize --ascend-prepare-for-emit")
    if not _run_afir_opt(
            step6, step7,
            ["--ascend-parallelize", "--ascend-prepare-for-emit",
             "--canonicalize", "--cse"],
            afir_opt):
        return False

    # ── Stage 7b: Canonicalize CANN Signature ──
    step7b = work_dir / "step7_cann.mlir"
    print(f"  [step7b] --ascend-canonicalize-cann-signature")
    if not _run_afir_opt(step7, step7b,
                         ["--ascend-canonicalize-cann-signature"], afir_opt):
        return False
    # TODO: cf.assert 应由 afir-translate 处理（注册 cf dialect 或在 codegen 前用 pass 消除）
    # ── Stage 7c: 移除 cf.assert（动态 shape broadcast 检查，afir-translate 不支持）──
    content = step7b.read_text()
    if "cf.assert" in content:
        cleaned = re.sub(r'^\s*cf\.assert\b.*\n', '', content, flags=re.MULTILINE)
        # 同时移除仅被 assert 使用的 cmpi
        cleaned = re.sub(r'^\s*%\w+ = arith\.cmpi eq,.*\n(?=\s*$|\s*%)', '', cleaned, flags=re.MULTILINE)
        step7b.write_text(cleaned)
        print(f"  [step7c] 移除 cf.assert（动态 shape broadcast 检查）")

    # ── Stage 8: C++ Codegen (CANN standard) ──
    afir_translate = _find_tool("afir-translate")
    if afir_translate is None:
        print(f"  [step8] afir-translate 不在 PATH 中，跳过 codegen")
        return False

    step8_cpp = work_dir / "step8_kernel.cpp"
    step8_tiling = work_dir / "step8_kernel.tiling_space.json"
    print(f"  [step8] afir-translate -mlir-to-cann")
    cmd = [afir_translate, "-mlir-to-cann", str(step7b),
           "-o", str(step8_cpp),
           "--tiling-space-out", str(step8_tiling)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  失败: afir-translate -mlir-to-cann")
        print(f"  stderr: {proc.stderr[:500]}")
        return False

    print(f"  C++ kernel: {step8_cpp}")
    print(f"  tiling space: {step8_tiling}")
    return True


# ================================================================
# Autotuner: stage 9-10 (compile + tune + verify)
# ================================================================

def _run_autotuner(work_dir: Path, num_inputs: int, shape_str: str) -> bool:
    """
    调用 autotuner 完成 compile + tiling 搜索 + 数值验证。

    autotuner 内部自动编译 kernel，搜索最优 tiling 参数，并验证数值正确性。
    """
    # 入口校验
    for i in range(num_inputs):
        npy = work_dir / f"input_{i}.npy"
        assert npy.exists(), f"缺少 {npy}"
    expected_npy = work_dir / "expected_0.npy"
    assert expected_npy.exists(), f"缺少 {expected_npy}"
    assert (work_dir / "step8_kernel.cpp").exists(), "缺少 step8_kernel.cpp"
    assert (work_dir / "step8_kernel.tiling_space.json").exists(), \
        "缺少 step8_kernel.tiling_space.json"
    assert shape_str, "shape_str 不能为空"

    # 填充空 values 的搜索参数（afir-translate 生成的 tiling_space 缺少搜索范围）
    _patch_tiling_space(work_dir / "step8_kernel.tiling_space.json")

    autotuner = _find_tool("autotuner")
    if autotuner is None:
        print("  autotuner 不在 PATH 中，跳过")
        return False

    input_npys = ",".join(
        str(work_dir / f"input_{i}.npy") for i in range(num_inputs)
    )
    artifact_dir = work_dir / "build_e2e"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        autotuner,
        "--space", str(work_dir / "step8_kernel.tiling_space.json"),
        "--kernel", str(work_dir / "step8_kernel.cpp"),
        "--inputs", input_npys,
        "--expected", str(expected_npy),
        "--shape", shape_str,
        "--output", str(work_dir / "best_config.json"),
        "--profile-out", str(work_dir / "perf_out"),
    ]
    print(f"  cmd: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # 清理模拟器 dump 文件
    _cleanup_sim_dumps()

    # Always convert expected to txt for inspection
    _npy_to_txt(expected_npy, artifact_dir / "expected.txt")

    if proc.returncode != 0:
        print(f"  失败: autotuner (精度验证失败)")
        print(f"  stdout: {proc.stdout[-500:]}")
        print(f"  stderr: {proc.stderr[:500]}")
        return False

    print(f"  autotuner stdout:\n{proc.stdout[-300:]}")
    return True
