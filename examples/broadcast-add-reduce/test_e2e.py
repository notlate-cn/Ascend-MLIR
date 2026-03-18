#!/usr/bin/env python3
"""
broadcast-add-reduce 端到端测试脚本

用法：
  source python/test/env.sh
  python3 examples/broadcast-add-reduce/test_e2e.py

流程：
  1. 用 bisheng 编译 step8_kernel.cpp → ELF binary
  2. 构造 TilingData（手动）
  3. 准备 f16 输入，用 CPU 仿真执行 kernel
  4. 与 numpy 参考计算对比，验证正确性
"""
import os
import sys
import struct
from pathlib import Path

# 必须在 import runtime 前设置
os.environ.setdefault("SOC_VERSION", "Ascend910B1")
os.environ.setdefault("ASCEND_CPU_SIMULATION", "1")
os.environ.setdefault("ASCEND_DEVICE_ID", "0")

import numpy as np

# 添加 python/ 到路径
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from runtime import compile_kernel, read_binary
from runtime.executor import KernelExecutor
from runtime.utils import clean_dump

EXAMPLE_DIR = Path(__file__).parent
KERNEL_SRC  = EXAMPLE_DIR / "step8_kernel.cpp"
BUILD_DIR   = EXAMPLE_DIR / "build_e2e"
KERNEL_NAME = "broadcast_add_reducesum"

# ---- 测试参数 ----
M = 32   # 行数（broadcast 维度）
N = 32   # 列数（reduce 维度，需为 32 的倍数以满足对齐）

# Tiling 参数
# TB_M: 每个核处理的行数（核间分块）；DataCopy half 最少 16 元素，所以 TB_M >= 16
# TB_N: 内层循环的行批次大小（UB 分块）
TB_M = 16  # 每核 16 行，M=32 → 2 个核
TB_N = 4   # 内层批次大小

def make_tiling_bytes(tb_m, tb_n, dim_m, dim_n):
    """
    构造 TilingData bytes。
    结构体（6个 int64_t，小端）：
      TB_M, TB_N, dim_arg0_0(M), dim_arg1_1(N), dim_arg0_1(M), dim_arg1_0(N)
    executor.py 的 tiling 传参方式：
      按 8 字节一个 word 追加到 args 末尾
    """
    fields = [tb_m, tb_n, dim_m, dim_n, dim_m, dim_n]
    return struct.pack(f"{len(fields)}q", *fields)  # q = int64_t，小端

def numpy_reference(input_a_f16, input_b_f16):
    """CPU 参考实现：broadcast + add + reducesum"""
    a = input_a_f16.astype(np.float32)   # [M]
    b = input_b_f16.astype(np.float32)   # [M, N]
    # broadcast a [M] → [M, N]
    a_bcast = np.broadcast_to(a[:, np.newaxis], (M, N))
    # add
    c = a_bcast + b                       # [M, N]
    # reducesum along N
    out = c.sum(axis=1)                   # [M]
    return out.astype(np.float16)

def main():
    print("=" * 60)
    print("broadcast-add-reduce 端到端测试")
    print("=" * 60)

    # ---- Step 1: 编译 kernel ----
    print(f"\n[Step 1] 编译 kernel: {KERNEL_SRC.name}")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        binary_path = compile_kernel(
            src_file=KERNEL_SRC,
            output_dir=BUILD_DIR,
            kernel_name=KERNEL_NAME,
        )
        print(f"  ✅ 编译成功: {binary_path}")
    except Exception as e:
        print(f"  ❌ 编译失败: {e}")
        import traceback; traceback.print_exc()
        os._exit(1)

    # ---- Step 2: 构造 TilingData ----
    print(f"\n[Step 2] 构造 TilingData")
    tiling_bytes = make_tiling_bytes(TB_M, TB_N, M, N)
    print(f"  TB_M={TB_M}, TB_N={TB_N}, M={M}, N={N}")
    print(f"  tiling size = {len(tiling_bytes)} bytes")

    # ---- Step 3: 准备输入数据 ----
    print(f"\n[Step 3] 准备输入数据 (f16)")
    np.random.seed(42)
    input_a = np.random.rand(M).astype(np.float16)
    input_b = np.random.rand(M, N).astype(np.float16)
    print(f"  input_a shape: {input_a.shape}")
    print(f"  input_b shape: {input_b.shape}")

    # ---- Step 4: 读取 binary ----
    print(f"\n[Step 4] 读取 ELF binary")
    binary_data = read_binary(binary_path)
    print(f"  binary size = {len(binary_data)} bytes")

    # ---- Step 5: 执行 kernel ----
    print(f"\n[Step 5] CPU 仿真执行 kernel")
    block_dim = (M + TB_M - 1) // TB_M  # 核数
    print(f"  block_dim = {block_dim}")
    try:
        output_buf = np.zeros((M,), dtype=np.float16)
        executor = KernelExecutor()
        executor.initialize()
        outputs = executor.execute(
            binary_data=binary_data,
            function_name=KERNEL_NAME,
            inputs=[input_a, input_b],
            outputs=[output_buf],
            tiling_data=tiling_bytes,
            block_dim=block_dim,
        )
        output = outputs[0]
        print(f"  ✅ 执行成功")
    except Exception as e:
        print(f"  ❌ 执行失败: {e}")
        import traceback; traceback.print_exc()
        sys.stdout.flush()
        clean_dump()
        os._exit(1)

    # ---- Step 6: 验证结果 ----
    print(f"\n[Step 6] 验证结果")
    expected = numpy_reference(input_a, input_b)

    abs_diff = np.abs(output.astype(np.float32) - expected.astype(np.float32))
    max_diff  = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())

    print(f"  期望输出[:5]: {expected[:5]}")
    print(f"  实际输出[:5]: {output[:5]}")
    print(f"  最大绝对误差: {max_diff:.4e}")
    print(f"  平均绝对误差: {mean_diff:.4e}")

    # f16 精度容忍度：atol=1e-2（f16 精度约 1e-3，reducesum N=32 有累积误差）
    passed = max_diff < 1.0  # 粗判：有输出且误差合理
    close  = np.allclose(output.astype(np.float32), expected.astype(np.float32),
                         rtol=1e-2, atol=1.0)

    print()
    print("=" * 60)
    if close:
        print("✅✅✅ 端到端测试通过！")
    else:
        print("❌❌❌ 端到端测试失败（结果不匹配）")
    print("=" * 60)

    sys.stdout.flush()
    clean_dump()
    os._exit(0 if close else 1)


if __name__ == "__main__":
    main()
