import sys
import os
import ctypes

# ========== 环境变量校验 ==========
print("=" * 60)
print("Ascend 环境初始化")
print("=" * 60)

# 检查 ASCEND_INSTALL_PATH 环境变量
if 'ASCEND_INSTALL_PATH' not in os.environ:
    print("❌ 错误: 环境变量 ASCEND_INSTALL_PATH 未设置")
    print("请先执行: source env.sh")
    sys.exit(1)

ASCEND_INSTALL_PATH = os.environ['ASCEND_INSTALL_PATH']
print(f"✅ ASCEND_INSTALL_PATH: {ASCEND_INSTALL_PATH}")

# 检查安装路径是否存在
if not os.path.exists(ASCEND_INSTALL_PATH):
    print(f"❌ 错误: Ascend 安装路径不存在: {ASCEND_INSTALL_PATH}")
    sys.exit(1)
print(f"✅ 安装路径存在")

# ========== 加载 libgraph.so ==========
libgraph_path = os.path.join(ASCEND_INSTALL_PATH, 'lib64/libgraph.so')

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

print()
print("=" * 60)
print("开始执行 Autofuse 任务")
print("=" * 60 + "\n")

# 导入 autofuse 模块
try:
    from autofuse.pyautofuse import ascir, Autofuser, AutofuserOptions, Schedule, CodeGen
    from autofuse import ascir_api
except ImportError as e:
    print(f"❌ 导入 autofuse 模块失败: {e}")
    sys.exit(1)

txt = """asc_graph_attr {
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
graph = ascir.utils.deserialize("asc_graph", txt)
ss = ascir.utils.debug_str(graph)
if ss:
    print("✅ 反序列化 ascgraph 成功")
    print()
    print(ss)
else:
    print("❌ 反序列化 ascgraph 失败")
    sys.exit(1)

options = AutofuserOptions()
fuser = Autofuser(options)
sched_result = fuser.schedule(graph)
tiling_def, host_tiling, op_kernel = fuser.codegen(sched_result)
if tiling_def and host_tiling and op_kernel:
    print("✅ 成功生成 tiling_def, host_tiling, op_kernel")
else:
    print("❌ 生成 tiling_def, host_tiling, op_kernel 失败")

# print(tiling_def)
# print(host_tiling)
# print(op_kernel)