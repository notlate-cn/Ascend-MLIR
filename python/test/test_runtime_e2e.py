#!/usr/bin/env python3
"""
Ascend Runtime 端到端测试脚本

完整的测试流程：
1. 从 graph_txt 开始
2. 使用 AscGen 生成源代码（tiling_def, host_tiling, op_kernel）
3. 编译 tiling_def + host_tiling 为 .so，并获取 tiling 数据
4. 使用 Bisheng 编译 device kernel
5. Launch kernel 并验证结果

注意：使用 os._exit() 避免 Python 清理时的段错误
"""
import os
import sys
import argparse
from pathlib import Path

# 设置环境（必须在导入其他模块前）
os.environ["SOC_VERSION"] = "Ascend910B1"

import numpy as np
import torch

# 添加 runtime 路径
sys.path.insert(0, str(Path(__file__).parent.parent))
# 导入 runtime 库（在环境设置后）
from runtime import (
    AscGen,
    compile_kernel as compile_kernel_with_bisheng,
    execute_kernel,
    read_binary,
)
from runtime.utils import clean_dump

# ================================================================
# 端到端测试
# ================================================================

def test_end_to_end(graph_text, output_path='./'):
    """端到端测试：从 graph 到 kernel 执行"""
    print("=" * 70)
    print("端到端测试：从 Graph 到 Kernel 执行")
    print("=" * 70)

    # ============================================================
    # 步骤 1: 读取 graph 文本
    # ============================================================
    print("\n[步骤 1] 读取 Graph 文本")
    print(f"  ✅ Graph 读取成功: {len(graph_text)} 字符")

    # ============================================================
    # 步骤 2: 使用 AscGen 生成源代码
    # ============================================================
    print("\n[步骤 2] 使用 AscGen 生成源代码")
    build_dir = Path(output_path)
    if build_dir.exists():
        import shutil
        shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    # 从 graph_text 中提取 graph_name（需要转换为 snake_case）
    import re
    match = re.search(r'graph_name:\s*"(\w+)"', graph_text)
    if match:
        original_name = match.group(1)
        # 将 CamelCase 转换为 snake_case
        graph_name = re.sub('([A-Z])', r'_\1', original_name).lower().lstrip('_')
    else:
        graph_name = "asc_graph"

    try:
        ascgen = AscGen()
        host_file, device_file, host_tiling, tiling_def = ascgen.compile_graph(
            graph_text, build_dir, graph_name
        )
        print(f"  ✅ 源代码生成成功:")
        print(f"    host_file: {host_file}")
        print(f"    device_file: {device_file}")
        print(f"    host_tiling: {len(host_tiling)} 字符")
        print(f"    tiling_def: {len(tiling_def)} 字符")
    except Exception as e:
        print(f"  ❌ 源代码生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ============================================================
    # 步骤 3: 编译 host_tiling 并获取 tiling 数据
    # ============================================================
    print("\n[步骤 3] 编译 host_tiling 并获取 tiling 数据")
    print("  编译 tiling_def + host_tiling 为 .so，然后调用获取 tiling 数据...")
    try:
        tiling_bytes = ascgen.calc_tiling_data(tiling_def, host_tiling, build_dir)
        print(f"  ✅ Tiling 数据获取成功: {len(tiling_bytes)} bytes")
    except Exception as e:
        print(f"  ❌ 获取 Tiling 数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ============================================================
    # 步骤 4: 使用 Bisheng 编译 device kernel
    # ============================================================
    print("\n[步骤 4] 使用 Bisheng 编译 device kernel")
    try:
        binary_path = compile_kernel_with_bisheng(
            src_file=device_file,
            output_dir=build_dir,
            kernel_name=graph_name
        )
        print(f"  ✅ Bisheng 编译成功: {binary_path}")
    except Exception as e:
        print(f"  ❌ Bisheng 编译失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ============================================================
    # 步骤 5: 准备测试数据
    # ============================================================
    print("\n[步骤 5] 准备测试数据")
    torch.manual_seed(42)
    input0 = torch.rand(20, 31, dtype=torch.float32)
    input1 = torch.rand(1, 31, dtype=torch.float32)
    broadcasted = input1.expand(20, 31)
    add_result = input0 + broadcasted
    mul_result = add_result * broadcasted
    expected_output = add_result - mul_result
    torch.save(input0, f'{build_dir}/{graph_name}_input0.pt')
    torch.save(input1, f'{build_dir}/{graph_name}_input1.pt')
    torch.save(expected_output, f'{build_dir}/{graph_name}_expected_output.pt')

    print(f"    input0: {input0.shape}, {input0.dtype}")
    print(f"    input1: {input1.shape}, {input1.dtype}")

    # ============================================================
    # 步骤 6: 读取 binary
    # ============================================================
    print("\n[步骤 6] 读取 ELF Binary")
    if not os.path.exists(binary_path):
        print(f"  ❌ Binary 文件不存在: {binary_path}")
        return False

    binary_data = read_binary(binary_path)
    print(f"  ✅ Binary 读取成功: {len(binary_data)} bytes")

    # ============================================================
    # 步骤 7: 执行 Kernel
    # ============================================================
    print("\n[步骤 7] 执行 Kernel")
    try:
        outputs = execute_kernel(
            binary_data=binary_data,
            function_name=graph_name,
            inputs=[input0.numpy(), input1.numpy()],
            output_shapes=[(20, 31)],
            tiling_data=tiling_bytes,
            dtype=np.float32
        )
        output_array = outputs[0]
        print(f"  ✅ Kernel 执行成功")
    except Exception as e:
        print(f"  ❌ Kernel 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ============================================================
    # 步骤 8: 验证结果
    # ============================================================
    print("\n[步骤 8] 验证结果")
    output_torch = torch.from_numpy(output_array)
    abs_diff = torch.abs(output_torch - expected_output)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()

    print(f"    最大绝对误差: {max_diff:.6e}")
    print(f"    平均绝对误差: {mean_diff:.6e}")
    print(f"\n    期望输出[0, :5]: {expected_output[0, :5]}")
    print(f"    实际输出[0, :5]: {output_torch[0, :5]}")

    is_close = torch.allclose(output_torch, expected_output, rtol=1e-5, atol=1e-5)

    return is_close


# ================================================================
# 主函数
# ================================================================

def main():
    """主测试函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Ascend Runtime 端到端测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从文件读取ascgraph
  python test_runtime_e2e.py --input path/to/asc_graph.txt

  # 指定输出目录
  python test_runtime_e2e.py --input asc_graph.txt --output ./output
        """
    )
    parser.add_argument(
        '--input',
        type=str,
        help='AscGraph文本文件路径。如果不指定，使用内置的graph定义'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./build',
        help='输出目录路径 (默认: ./build)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Ascend Runtime 端到端测试")
    print("=" * 70)
    print(f"版本: 0.1.0")
    print(f"Python: {sys.version}")

    if not args.input:
        print("❌ --input为必填项：python test_runtime_e2e.py --input path/to/asc_graph.txt")
        os._exit(1)

    try:
        # 读取或获取graph_text
        print(f"📖 从文件读取AscGraph: {args.input}")
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ 文件不存在: {args.input}")
            os._exit(1)

        with open(input_path, 'r', encoding='utf-8') as f:
            graph_text = f.read()
        success = test_end_to_end(graph_text, output_path=args.output)

        print("\n" + "=" * 70)
        if success:
            print("✅✅✅ 端到端测试通过！ ✅✅✅")
        else:
            print("❌❌❌ 端到端测试失败 ❌❌❌")
        print("=" * 70)

        # 清理 camodel dump 文件
        clean_dump()

        os._exit(0 if success else 1)
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ 测试异常: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()

        # 异常时也清理 dump 文件
        clean_dump()

        os._exit(1)


if __name__ == "__main__":
    main()
