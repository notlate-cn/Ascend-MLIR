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
from codecs import ignore_errors
from pathlib import Path

# 设置环境（必须在导入其他模块前）
os.environ["SOC_VERSION"] = "Ascend910B1"

import numpy as np
import torch

# 添加 runtime 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def get_asc_graph_text():
    """获取 ascgraph 的文本定义"""
    return """asc_graph_attr {
  tiling_key: -1
  axis {
    name: "z0"
    size: "20"
    align: "1"
    allow_unaligned_tail: true
  }
  axis {
    id: 1
    name: "z1"
    size: "31"
    align: "1"
    allow_unaligned_tail: true
  }
}
asc_node {
  outputs {
    attr {
      axis_ids: 0
      axis_ids: 1
      repeats: "20"
      repeats: "31"
      strides: "31"
      strides: "1"
      mem {
        tensor_id: -1
      }
      que {
        id: -1
        depth: -1
        buf_num: -1
      }
      buf {
        id: -1
      }
      opt {
        reuse_id: -1
        ref_tensor: -1
        merge_scope: -1
      }
    }
  }
  attr {
    name: "HashCopyAscGraph/Data_0"
    type: "Data"
    sched {
      axis: 0
      axis: 1
      loop_axis: -1
    }
    api {
      compute_type: 11
    }
    ir_attr_def {
      attr {
        key: "index"
        value {
          i: 0
        }
      }
    }
  }
  ir_def {
    output_names: "y"
    output_ir_type: 0
    type: "Data"
    output_nums: 1
  }
}
asc_node {
  input_src {
    src_node_name: "HashCopyAscGraph/Data_0"
  }
  outputs {
    attr {
      axis_ids: 0
      axis_ids: 1
      repeats: "20"
      repeats: "31"
      strides: "31"
      strides: "1"
      mem {
        tensor_id: -1
      }
      que {
        id: -1
        depth: -1
        buf_num: -1
      }
      buf {
        id: -1
      }
      opt {
        reuse_id: -1
        ref_tensor: -1
        merge_scope: -1
      }
    }
  }
  attr {
    name: "HashCopyAscGraph/Load_1"
    type: "Load"
    sched {
      exec_order: 1
      axis: 0
      axis: 1
      loop_axis: -1
    }
    api {
      type: 1
      unit: 2
    }
    ir_attr_def {
      attr {
        key: "offset"
        value {
          expression: "0"
        }
      }
    }
  }
  ir_def {
    input_names: "x"
    output_names: "y"
    input_ir_type: 0
    output_ir_type: 0
    type: "Load"
    input_nums: 1
    output_nums: 1
  }
}
asc_node {
  outputs {
    attr {
      axis_ids: 0
      axis_ids: 1
      repeats: "1"
      repeats: "31"
      strides: "0"
      strides: "1"
      mem {
        tensor_id: -1
      }
      que {
        id: -1
        depth: -1
        buf_num: -1
      }
      buf {
        id: -1
      }
      opt {
        reuse_id: -1
        ref_tensor: -1
        merge_scope: -1
      }
    }
  }
  attr {
    name: "HashCopyAscGraph/Data_2"
    type: "Data"
    sched {
      exec_order: 2
      axis: 0
      axis: 1
      loop_axis: -1
    }
    api {
      compute_type: 11
    }
    ir_attr_def {
      attr {
        key: "index"
        value {
          i: 1
        }
      }
    }
  }
  ir_def {
    output_names: "y"
    output_ir_type: 0
    type: "Data"
    output_nums: 1
  }
}
asc_node {
  input_src {
    src_node_name: "HashCopyAscGraph/Data_2"
  }
  outputs {
    attr {
      axis_ids: 0
      axis_ids: 1
      repeats: "1"
      repeats: "31"
      strides: "0"
      strides: "1"
      mem {
        tensor_id: -1
      }
      que {
        id: -1
        depth: -1
        buf_num: -1
      }
      buf {
        id: -1
      }
      opt {
        reuse_id: -1
        ref_tensor: -1
        merge_scope: -1
      }
    }
  }
  attr {
    name: "HashCopyAscGraph/Load_3"
    type: "Load"
    sched {
      exec_order: 3
      axis: 0
      axis: 1
      loop_axis: -1
    }
    api {
      type: 1
      unit: 2
    }
    ir_attr_def {
      attr {
        key: "offset"
        value {
          expression: "0"
        }
      }
    }
  }
  ir_def {
    input_names: "x"
    output_names: "y"
    input_ir_type: 0
    output_ir_type: 0
    type: "Load"
    input_nums: 1
    output_nums: 1
  }
}
asc_node {
  input_src {
    src_node_name: "HashCopyAscGraph/Load_3"
  }
  outputs {
    attr {
      axis_ids: 0
      axis_ids: 1
      repeats: "20"
      repeats: "31"
      strides: "31"
      strides: "1"
      mem {
        tensor_id: -1
      }
      que {
        id: -1
        depth: -1
        buf_num: -1
      }
      buf {
        id: -1
      }
      opt {
        reuse_id: -1
        ref_tensor: -1
        merge_scope: -1
      }
    }
  }
  attr {
    name: "HashCopyAscGraph/Broadcast_4"
    type: "Broadcast"
    sched {
      exec_order: -1
      axis: 0
      axis: 1
      loop_axis: -1
    }
    api {
      type: 2
      compute_type: 11
      unit: 7
    }
  }
  ir_def {
    input_names: "x"
    output_names: "y"
    input_ir_type: 0
    output_ir_type: 0
    type: "Broadcast"
    input_nums: 1
    output_nums: 1
  }
}
asc_node {
  input_src {
    src_node_name: "HashCopyAscGraph/Load_1"
  }
  input_src {
    src_node_name: "HashCopyAscGraph/Broadcast_4"
  }
  outputs {
    attr {
      axis_ids: 0
      axis_ids: 1
      repeats: "20"
      repeats: "31"
      strides: "31"
      strides: "1"
      mem {
        tensor_id: -1
      }
      que {
        id: -1
        depth: -1
        buf_num: -1
      }
      buf {
        id: -1
      }
      opt {
        reuse_id: -1
        ref_tensor: -1
        merge_scope: -1
      }
    }
  }
  attr {
    name: "HashCopyAscGraph/Add_5"
    type: "Add"
    sched {
      exec_order: 4
      axis: 0
      axis: 1
      loop_axis: -1
    }
    api {
      type: 1
      compute_type: 3
      unit: 5
    }
  }
  ir_def {
    input_names: "x1"
    input_names: "x2"
    output_names: "y"
    input_ir_type: 0
    input_ir_type: 0
    output_ir_type: 0
    type: "Add"
    input_nums: 1
    input_nums: 1
    output_nums: 1
  }
}
asc_node {
  input_src {
    src_node_name: "HashCopyAscGraph/Add_5"
  }
  input_src {
    src_node_name: "HashCopyAscGraph/Broadcast_4"
  }
  outputs {
    attr {
      axis_ids: 0
      axis_ids: 1
      repeats: "20"
      repeats: "31"
      strides: "31"
      strides: "1"
      mem {
        tensor_id: -1
      }
      que {
        id: -1
        depth: -1
        buf_num: -1
      }
      buf {
        id: -1
      }
      opt {
        reuse_id: -1
        ref_tensor: -1
        merge_scope: -1
      }
    }
  }
  attr {
    name: "HashCopyAscGraph/Mul_6"
    type: "Mul"
    sched {
      exec_order: 4
      axis: 0
      axis: 1
      loop_axis: -1
    }
    api {
      type: 1
      compute_type: 3
      unit: 5
    }
  }
  ir_def {
    input_names: "x1"
    input_names: "x2"
    output_names: "y"
    input_ir_type: 0
    input_ir_type: 0
    output_ir_type: 0
    type: "Mul"
    input_nums: 1
    input_nums: 1
    output_nums: 1
  }
}
asc_node {
  input_src {
    src_node_name: "HashCopyAscGraph/Add_5"
  }
  input_src {
    src_node_name: "HashCopyAscGraph/Mul_6"
  }
  outputs {
    attr {
      axis_ids: 0
      axis_ids: 1
      repeats: "20"
      repeats: "31"
      strides: "31"
      strides: "1"
      mem {
        tensor_id: -1
      }
      que {
        id: -1
        depth: -1
        buf_num: -1
      }
      buf {
        id: -1
      }
      opt {
        reuse_id: -1
        ref_tensor: -1
        merge_scope: -1
      }
    }
  }
  attr {
    name: "HashCopyAscGraph/Sub_7"
    type: "Sub"
    sched {
      exec_order: 5
      axis: 0
      axis: 1
      loop_axis: -1
    }
    api {
      type: 1
      compute_type: 3
      unit: 5
    }
  }
  ir_def {
    input_names: "x1"
    input_names: "x2"
    output_names: "y"
    input_ir_type: 0
    input_ir_type: 0
    output_ir_type: 0
    type: "Sub"
    input_nums: 1
    input_nums: 1
    output_nums: 1
  }
}
asc_node {
  input_src {
    src_node_name: "HashCopyAscGraph/Sub_7"
  }
  outputs {
    attr {
      axis_ids: 0
      axis_ids: 1
      repeats: "20"
      repeats: "31"
      strides: "31"
      strides: "1"
      mem {
        tensor_id: -1
      }
      que {
        id: -1
        depth: -1
        buf_num: -1
      }
      buf {
        id: -1
      }
      opt {
        reuse_id: -1
        ref_tensor: -1
        merge_scope: -1
      }
    }
  }
  attr {
    name: "HashCopyAscGraph/Store_8"
    type: "Store"
    sched {
      exec_order: 6
      axis: 0
      axis: 1
      loop_axis: -1
    }
    api {
      type: 1
      compute_type: 1
      unit: 2
    }
    ir_attr_def {
    }
  }
  ir_def {
    input_names: "x"
    output_names: "y"
    input_ir_type: 0
    output_ir_type: 0
    type: "Store"
    input_nums: 1
    output_nums: 1
  }
}
asc_node {
  input_src {
    src_node_name: "HashCopyAscGraph/Store_8"
  }
  outputs {
    attr {
      mem {
        tensor_id: -1
      }
      que {
        id: -1
        depth: -1
        buf_num: -1
      }
      buf {
        id: -1
      }
      opt {
        reuse_id: -1
        ref_tensor: -1
        merge_scope: -1
      }
    }
  }
  attr {
    name: "HashCopyAscGraph/Output_9"
    type: "Output"
    sched {
      exec_order: 7
      axis: 0
      axis: 1
      loop_axis: -1
    }
    api {
      compute_type: 11
    }
    ir_attr_def {
      attr {
        key: "index"
        value {
          i: 0
        }
      }
    }
  }
  ir_def {
    input_names: "x"
    output_names: "y"
    input_ir_type: 0
    output_ir_type: 0
    type: "Output"
    input_nums: 1
    output_nums: 1
  }
}
graph_name: "HashCopyAscGraph"
"""

# ================================================================
# 端到端测试
# ================================================================

def test_end_to_end(graph_text, output_path='./'):
    """端到端测试：从 graph 到 kernel 执行"""
    print("=" * 70)
    print("端到端测试：从 Graph 到 Kernel 执行")
    print("=" * 70)

    # 导入 runtime 库（在环境设置后）
    from runtime import (
        AscGen,
        compile_kernel as compile_kernel_with_bisheng,
        execute_kernel,
        read_binary,
    )

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
    print("=" * 70)
    print("Ascend Runtime 端到端测试")
    print("=" * 70)
    print(f"版本: 0.1.0")
    print(f"Python: {sys.version}")
    print()

    try:
        graph_text = get_asc_graph_text()
        success = test_end_to_end(graph_text, output_path='./build')

        print("\n" + "=" * 70)
        if success:
            print("✅✅✅ 端到端测试通过！ ✅✅✅")
        else:
            print("❌❌❌ 端到端测试失败 ❌❌❌")
        print("=" * 70)

        os._exit(0 if success else 1)
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ 测试异常: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        os._exit(1)


if __name__ == "__main__":
    main()
