import sys
import os
import ctypes


def verify_ascend_environment():
    """验证 Ascend 环境变量和安装路径"""
    print("=" * 60)
    print("Ascend 环境初始化")
    print("=" * 60)

    # 检查 ASCEND_INSTALL_PATH 环境变量
    if 'ASCEND_INSTALL_PATH' not in os.environ:
        print("❌ 错误: 环境变量 ASCEND_INSTALL_PATH 未设置")
        print("请先执行: source env.sh")
        sys.exit(1)

    ascend_install_path = os.environ['ASCEND_INSTALL_PATH']
    print(f"✅ ASCEND_INSTALL_PATH: {ascend_install_path}")

    # 检查安装路径是否存在
    if not os.path.exists(ascend_install_path):
        print(f"❌ 错误: Ascend 安装路径不存在: {ascend_install_path}")
        sys.exit(1)
    print(f"✅ 安装路径存在")

    return ascend_install_path


def load_libgraph(ascend_install_path):
    """加载 libgraph.so 动态库"""
    libgraph_path = os.path.join(ascend_install_path, 'lib64/libgraph.so')

    # 检查 libgraph.so 是否存在
    if not os.path.exists(libgraph_path):
        print(f"❌ 错误: libgraph.so 不存在: {libgraph_path}")
        print("请检查 Ascend 安装是否完整")
        sys.exit(1)

    # 加载 libgraph.so
    try:
        ctypes.CDLL(libgraph_path, mode=ctypes.RTLD_GLOBAL)
        print("✅ 成功加载 libgraph.so")
    except OSError as e:
        print(f"❌ 加载 libgraph.so 失败: {e}")
        print(f"\n可能的原因：")
        print(f"  1. CANN包安装失败，缺少依赖库（如 libplatform.so）")
        print(f"  2. LD_LIBRARY_PATH 未正确设置，需要先执行: source env.sh")
        print(f"\n当前 LD_LIBRARY_PATH:")
        print(f"  {os.environ.get('LD_LIBRARY_PATH', '(未设置)')}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        sys.exit(1)


def import_autofuse_modules():
    """导入 autofuse 相关模块"""
    try:
        from autofuse.pyautofuse import ascir, Autofuser, AutofuserOptions, Schedule, CodeGen
        from autofuse import ascir_api
        return ascir, Autofuser, AutofuserOptions
    except ImportError as e:
        print(f"❌ 导入 autofuse 模块失败: {e}")
        sys.exit(1)


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


def deserialize_asc_graph(ascir, graph_text):
    """反序列化 ascgraph 文本为图对象"""
    graph = ascir.utils.deserialize("asc_graph", graph_text)
    debug_str = ascir.utils.debug_str(graph)

    if debug_str:
        print("✅ 反序列化 ascgraph 成功")
        print()
        print(debug_str)
    else:
        print("❌ 反序列化 ascgraph 失败")
        sys.exit(1)

    return graph


def run_autofuse_pipeline(Autofuser, AutofuserOptions, graph):
    """执行 autofuse 流程：schedule 和 codegen"""
    print()
    print("=" * 60)
    print("开始执行 Autofuse 任务")
    print("=" * 60 + "\n")

    options = AutofuserOptions()
    fuser = Autofuser(options)
    sched_result = fuser.schedule(graph)
    tiling_def, host_tiling, op_kernel = fuser.codegen(sched_result)

    if tiling_def and host_tiling and op_kernel:
        print("✅ 成功生成 tiling_def, host_tiling, op_kernel")
    else:
        print("❌ 生成 tiling_def, host_tiling, op_kernel 失败")
        sys.exit(1)

    return tiling_def, host_tiling, op_kernel


def setup_cmake_environment(ascend_install_path):
    """设置 CMake 编译环境变量"""
    # CMake 需要找到 ASC 包的配置文件
    cmake_prefix_path = os.path.join(ascend_install_path, 'lib64/cmake')
    if not os.path.exists(cmake_prefix_path):
        # 尝试备选路径
        cmake_prefix_path = os.path.join(ascend_install_path, 'aarch64-linux/lib64/cmake')

    if os.path.exists(cmake_prefix_path):
        os.environ['CMAKE_PREFIX_PATH'] = cmake_prefix_path
        print(f"✅ 设置 CMAKE_PREFIX_PATH: {cmake_prefix_path}")
    else:
        print(f"⚠️  警告: 找不到 CMake 配置目录，编译可能失败")
        print(f"   尝试的路径: {cmake_prefix_path}")

    # 设置 ASCEND_CANN_PACKAGE_PATH（CMake需要）
    os.environ['ASCEND_CANN_PACKAGE_PATH'] = ascend_install_path
    print(f"✅ 设置 ASCEND_CANN_PACKAGE_PATH: {ascend_install_path}")


def run_jit_compile(tiling_def, host_tiling, op_kernel, graph_name="HashCopyAscGraph"):
    """执行 JIT 编译，生成 .so 文件"""
    from autofuse.compile_adapter import jit_compile

    print("\n" + "=" * 60)
    print("开始 JIT 编译")
    print("=" * 60 + "\n")

    compile_args = [
        f"--graph_name={graph_name}",
        f"--output_file=./{graph_name}.so",
        "--output_path=./build-ascir",
        "--force_unknown=True"
    ]

    try:
        jit_compile(tiling_def, host_tiling, op_kernel, compile_args)
        print("\n✅ JIT 编译成功！")
        print(f"   输出文件: ./build-ascir/{graph_name}.so")
    except Exception as e:
        print(f"\n❌ JIT 编译失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """主函数：协调整个工作流程"""
    # 1. 验证 Ascend 环境
    ascend_install_path = verify_ascend_environment()

    # 2. 加载 libgraph.so
    load_libgraph(ascend_install_path)

    # 3. 导入 autofuse 模块
    ascir, Autofuser, AutofuserOptions = import_autofuse_modules()

    # 4. 获取并反序列化 ascgraph
    graph_text = get_asc_graph_text()
    graph = deserialize_asc_graph(ascir, graph_text)

    # 5. 执行 autofuse 流程
    tiling_def, host_tiling, op_kernel = run_autofuse_pipeline(
        Autofuser, AutofuserOptions, graph
    )

    # 6. 设置 CMake 环境
    setup_cmake_environment(ascend_install_path)

    # 7. 执行 JIT 编译
    run_jit_compile(tiling_def, host_tiling, op_kernel)

if __name__ == "__main__":
    main()