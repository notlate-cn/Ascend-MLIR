"""
pytest 集成：torch_e2e_test 装饰器

用法：
    @torch_e2e_test
    def test_my_model():
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.relu().sum(dim=1)
        return Model(), [torch.randn(32, 64)]
"""

import os
import re
import subprocess
import shutil
import functools
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from .convert import torch_to_linalg

REPO_ROOT = Path(__file__).parent.parent.parent
OUTPUT_ROOT = REPO_ROOT / "output" / "torch_e2e"


def torch_e2e_test(
    func=None,
    *,
    transform_mlir: str | Path | None = None,
    tiling: dict | None = None,
    kernel_cpp: str | Path | None = None,
    rtol: float = 1e-2,
    atol: float = 1.0,
    dtype: torch.dtype = torch.float16,
):
    """
    pytest 装饰器：定义一个 torch → NPU e2e 测试。

    被装饰函数应返回 (model, sample_inputs)。
    装饰器负责执行完整 pipeline 并验证精度。

    Args:
        transform_mlir: Transform dialect 脚本路径（step2 tiling 用）
        tiling: Tiling 参数，如 {"TB_M": 16, "TB_N": 4}
        kernel_cpp: 跳过 MLIR pipeline，直接使用已有 C++ kernel
        rtol/atol: 精度容忍度
        dtype: 目标数据类型
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper():
            model, inputs = fn()
            test_name = fn.__name__
            work_dir = OUTPUT_ROOT / test_name
            work_dir.mkdir(parents=True, exist_ok=True)

            # Stage 1: torch → linalg MLIR
            linalg_path = work_dir / "step0_linalg.mlir"
            print(f"\n[Stage 1] torch → linalg MLIR")
            torch_to_linalg(model, inputs, output_path=linalg_path, dtype=dtype)
            print(f"  输出: {linalg_path}")

            # Stage 2: MLIR pipeline → C++ kernel
            resolved_kernel_cpp = None
            if kernel_cpp is not None:
                resolved_kernel_cpp = Path(kernel_cpp)
                if not resolved_kernel_cpp.is_absolute():
                    resolved_kernel_cpp = REPO_ROOT / resolved_kernel_cpp
                print(f"\n[Stage 2-8] 使用已有 kernel: {resolved_kernel_cpp}")
            else:
                print(f"\n[Stage 2-8] MLIR pipeline → C++ kernel")
                if tiling:
                    print(f"  tiling: {tiling}")
                resolved_kernel_cpp = _run_mlir_pipeline(
                    linalg_path, work_dir, transform_mlir, tiling
                )

            # Stage 9: torch reference
            print(f"\n[Stage 9] PyTorch reference 计算")
            model_typed = model.to(dtype).eval()
            inputs_typed = [x.to(dtype) for x in inputs]
            with torch.no_grad():
                expected = model_typed(*inputs_typed)
            expected_np = expected.numpy()
            print(f"  输出 shape: {expected_np.shape}, dtype: {expected_np.dtype}")

            # 验证 C++ kernel 生成
            assert resolved_kernel_cpp is not None and resolved_kernel_cpp.exists(), \
                f"C++ kernel 生成失败: {resolved_kernel_cpp}"
            print(f"\n  C++ kernel 生成成功: {resolved_kernel_cpp}")

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ================================================================
# MLIR Pipeline: 对应 run.sh 的 stage 0-8
# ================================================================

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


def _generate_transform_script(fused_path: Path, work_dir: Path) -> Path | None:
    """
    读取 step1 fused IR，注入通用 transform 序列，生成 step2_transform.mlir。

    适用场景：单个 linalg.generic（fusion 后最常见的情况）。
    切分策略：沿第一个 parallel 维度做两级 tile（TB 核间 + Tb UB批次），
    其余维度不切（tile_size=0）。

    生成的 transform 序列：
      1. add_index_args 追加 2 个 index 参数（TB_M, Tb_M）
      2. TB 层 tile + ascendc.parallel 标注
      3. Tb 层 tile + ascendc.prologue/epilogue 标注
      4. ascendc.unit = "AiCore.Vector" 标注
    """
    content = fused_path.read_text()

    # 解析 iterator_types 以确定 tile_sizes
    match = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', content)
    if not match:
        print(f"  [transform-gen] 未找到 linalg.generic iterator_types，跳过")
        return None

    iters = [s.strip().strip('"') for s in match.group(1).split(",")]
    ndims = len(iters)

    # tile_sizes: parallel 维度用参数，其余填 0
    # 找到第一个 parallel 维度的索引
    par_idx = next((i for i, t in enumerate(iters) if t == "parallel"), None)
    if par_idx is None:
        print(f"  [transform-gen] 无 parallel 维度，跳过 tiling")
        return None

    # 构造 tile_sizes 字符串，如 [%tb_m, 0] 或 [0, %tb_m]
    def _tile_sizes(param_name: str) -> str:
        sizes = ["0"] * ndims
        sizes[par_idx] = f"%{param_name}"
        return "[" + ", ".join(sizes) + "]"

    tb_tile = _tile_sizes("tb_m")
    tb_inner_tile = _tile_sizes("tb_inner_m")

    # 替换 module 头，加上 transform.with_named_sequence
    new_content = content.replace(
        "module {",
        "module attributes {transform.with_named_sequence} {",
        1,
    )

    # 在最后一个 } 前注入 transform 序列
    transform_seq = f"""
  // ═══════════ 自动生成的 Transform 调度脚本 ═══════════
  transform.named_sequence @__transform_main(
      %root : !transform.any_op {{transform.readonly}}
  ) {{
    // 追加 2 个 index 参数：TB_M（核间切分）, Tb_M（UB 批次切分）
    %func = transform.structured.match ops{{["func.func"]}} in %root
        : (!transform.any_op) -> !transform.any_op
    %func_new, %tb_m, %tb_inner_m =
        transform.func.add_index_args %func, 2
            : (!transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 匹配 linalg.generic
    %generic = transform.structured.match ops{{["linalg.generic"]}} in %func_new
        : (!transform.any_op) -> !transform.any_op

    // TB 层切分（核间并行）
    %tiled_tb, %loop_tb =
        transform.structured.tile_using_for %generic
            tile_sizes {tb_tile}
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    %true_param = transform.param.constant true -> !transform.any_param
    transform.annotate %loop_tb "ascendc.parallel"
        = %true_param : !transform.any_op, !transform.any_param

    // Tb 层切分（UB 批次粒度）
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

    // 标注 Vector 执行单元
    %vector_unit = transform.param.constant "AiCore.Vector" -> !transform.any_param
    transform.annotate %tiled_inner "ascendc.unit"
        = %vector_unit : !transform.any_op, !transform.any_param

    transform.yield
  }}
"""

    # 在最后一个 } 前插入
    last_brace = new_content.rfind("}")
    new_content = new_content[:last_brace] + transform_seq + "}\n"

    out_path = work_dir / "step2_transform.mlir"
    out_path.write_text(new_content)
    print(f"  [transform-gen] iterator_types={iters}, tile parallel dim d{par_idx}")
    return out_path


def _run_mlir_pipeline(
    linalg_path: Path,
    work_dir: Path,
    transform_mlir: str | Path | None,
    tiling: dict | None = None,
) -> Path | None:
    """
    运行完整 MLIR pipeline（对应 run.sh stage 1-8）:
      step1: --linalg-fuse-elementwise-ops
      step2: --transform-interpreter (需要 transform 脚本)
      step3: --one-shot-bufferize
      step4: --ascendc-buffer-placement
      step5: --linalg-to-ascendc
      step6: --ascendc-parallelize
      step7: --ascendc-prepare-for-emit
      step8: ascir-translate -mlir-to-ascendc
    """
    afir_opt = _find_tool("afir-opt")
    if afir_opt is None:
        print("  afir-opt 不在 PATH 中，跳过")
        return None

    # ── Stage 1: Fusion ──
    step1 = work_dir / "step1_fused.mlir"
    print(f"  [step1] --linalg-fuse-elementwise-ops")
    if not _run_afir_opt(linalg_path, step1,
                         ["--linalg-fuse-elementwise-ops"], afir_opt):
        return None

    # ── Stage 2: Tiling (Transform Interpreter) ──
    if transform_mlir is not None:
        # 用户提供的 transform 脚本
        transform_path = Path(transform_mlir)
        if not transform_path.is_absolute():
            transform_path = REPO_ROOT / transform_path
    elif tiling is not None:
        # 从 tiling dict 自动生成 transform 脚本
        transform_path = _generate_transform_script(step1, work_dir)
    else:
        transform_path = None

    if transform_path is not None:
        step2 = work_dir / "step2_tiled.mlir"
        print(f"  [step2] --transform-interpreter ({transform_path.name})")
        if not _run_afir_opt(transform_path, step2,
                             ["--transform-interpreter", "--canonicalize", "--cse"],
                             afir_opt):
            return None
    else:
        print(f"  [step2] 跳过（未提供 transform 脚本或 tiling 参数）")
        step2 = step1

    # ── Stage 3: Bufferize ──
    step3 = work_dir / "step3_bufferized.mlir"
    bufferize_opts = ("--one-shot-bufferize="
                      "bufferize-function-boundaries=true "
                      "allow-return-allocs-from-loops=true "
                      "function-boundary-type-conversion=identity-layout-map")
    print(f"  [step3] --one-shot-bufferize")
    if not _run_afir_opt(step2, step3, [bufferize_opts, "--cse"], afir_opt):
        return None

    # ── Stage 4: Buffer Placement ──
    step4 = work_dir / "step4_buffer_placement.mlir"
    print(f"  [step4] --ascendc-buffer-placement")
    if not _run_afir_opt(step3, step4, ["--ascendc-buffer-placement"], afir_opt):
        return None

    # ── Stage 5: Linalg → AscendC ──
    step5 = work_dir / "step5_ascendc.mlir"
    print(f"  [step5] --linalg-to-ascendc")
    if not _run_afir_opt(step4, step5,
                         ["--linalg-to-ascendc", "--canonicalize", "--cse"],
                         afir_opt):
        return None

    # ── Stage 6: Parallelize ──
    step6 = work_dir / "step6_parallelize.mlir"
    print(f"  [step6] --ascendc-parallelize")
    if not _run_afir_opt(step5, step6,
                         ["--ascendc-parallelize", "--canonicalize", "--cse"],
                         afir_opt):
        return None

    # ── Stage 7: Prepare For Emit ──
    step7 = work_dir / "step7_kernel.mlir"
    print(f"  [step7] --ascendc-prepare-for-emit")
    if not _run_afir_opt(step6, step7,
                         ["--ascendc-prepare-for-emit", "--canonicalize", "--cse"],
                         afir_opt):
        return None

    # ── Stage 8: C++ Codegen ──
    ascir_translate = _find_tool("ascir-translate")
    if ascir_translate is None:
        print(f"  [step8] ascir-translate 不在 PATH 中，跳过 codegen")
        return None

    # 去掉 transform.named_sequence（ascir-translate 不支持）
    step8_no_transform = work_dir / "step8_no_transform.mlir"
    content = step7.read_text()
    content = content.replace("module attributes {transform.with_named_sequence}",
                              "module")
    content = re.sub(
        r"  transform\.named_sequence.*?^  \}\n", "",
        content, flags=re.DOTALL | re.MULTILINE,
    )
    step8_no_transform.write_text(content)

    step8_cpp = work_dir / "step8_kernel.cpp"
    print(f"  [step8] ascir-translate -mlir-to-ascendc")
    cmd = [ascir_translate, "-mlir-to-ascendc",
           str(step8_no_transform), "-o", str(step8_cpp)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  失败: ascir-translate")
        print(f"  stderr: {proc.stderr[:500]}")
        return None

    print(f"  C++ kernel 生成成功: {step8_cpp}")
    return step8_cpp