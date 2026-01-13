#!/usr/bin/env python3
"""
Ascend Runtime 库测试脚本

独立实现完整测试流程，避免导入导致的段错误
"""
import os
import sys

# 设置环境（必须在其他导入前）
os.environ["ASCEND_HOME_PATH"] = "/home/niu/Ascend/latest"
os.environ["SOC_VERSION"] = "Ascend910B1"

import numpy as np
import torch
import ctypes

# ================================================================
# DevBinary 结构体
# ================================================================

class DevBinary(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("data", ctypes.c_char_p),
        ("length", ctypes.c_uint64),
    ]


# ================================================================
# 简化的 Runtime 类
# ================================================================

class SimpleRuntime:
    def __init__(self):
        self.runtime_path = "/home/niu/Ascend/latest/aarch64-linux/simulator/Ascend910B1/lib/libruntime_camodel.so"
        self.runtime = ctypes.CDLL(self.runtime_path, mode=ctypes.RTLD_GLOBAL)
        self._setup_api()

    def _setup_api(self):
        self.runtime.rtSetDevice.argtypes = [ctypes.c_int32]
        self.runtime.rtSetDevice.restype = ctypes.c_int

        self.runtime.rtDevBinaryRegister.argtypes = [
            ctypes.POINTER(DevBinary),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.runtime.rtDevBinaryRegister.restype = ctypes.c_int

        self.runtime.rtFunctionRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.runtime.rtFunctionRegister.restype = ctypes.c_int

        self.runtime.rtMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint16,
        ]
        self.runtime.rtMalloc.restype = ctypes.c_int

        self.runtime.rtFree.argtypes = [ctypes.c_void_p]
        self.runtime.rtFree.restype = ctypes.c_int

        self.runtime.rtMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
        ]
        self.runtime.rtMemcpy.restype = ctypes.c_int

        self.runtime.rtCtxGetCurrentDefaultStream.argtypes = []
        self.runtime.rtCtxGetCurrentDefaultStream.restype = ctypes.c_void_p

        self.runtime.rtKernelLaunch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.runtime.rtKernelLaunch.restype = ctypes.c_int

        self.runtime.rtDeviceSynchronize.argtypes = []
        self.runtime.rtDeviceSynchronize.restype = ctypes.c_int

    def set_device(self, device_id=0):
        return self.runtime.rtSetDevice(ctypes.c_int32(device_id))

    def register_binary(self, binary_data):
        magic = 0x41415246  # RT_DEV_BINARY_MAGIC_ELF_AIVEC
        device_binary = DevBinary(
            magic=ctypes.c_uint32(magic),
            version=ctypes.c_uint32(0),
            data=ctypes.c_char_p(binary_data),
            length=ctypes.c_uint64(len(binary_data)),
        )
        binary_handle = ctypes.c_void_p()
        ret = self.runtime.rtDevBinaryRegister(
            ctypes.byref(device_binary),
            ctypes.byref(binary_handle),
        )
        return binary_handle.value if ret == 0 else None

    def register_function(self, kernel_handle, function_name):
        fn_name_bytes = function_name.encode("utf-8")
        name_ptr = ctypes.c_char_p(fn_name_bytes)
        name_void_ptr = ctypes.cast(name_ptr, ctypes.c_void_p)

        ret = self.runtime.rtFunctionRegister(
            ctypes.c_void_p(kernel_handle),
            name_void_ptr,
            name_ptr,
            name_void_ptr,
            ctypes.c_uint32(0),
        )
        return name_void_ptr.value if ret == 0 else None

    def malloc(self, size):
        c_memory_p = ctypes.c_void_p()
        ret = self.runtime.rtMalloc(
            ctypes.byref(c_memory_p),
            ctypes.c_uint64(size),
            ctypes.c_uint32(0),
            ctypes.c_uint16(33),
        )
        return c_memory_p.value if ret == 0 else None

    def free(self, ptr):
        self.runtime.rtFree(ctypes.c_void_p(ptr))

    def memcpy(self, dst, src, size, kind):
        return self.runtime.rtMemcpy(
            ctypes.c_void_p(dst),
            ctypes.c_uint64(size),
            ctypes.c_void_p(src),
            ctypes.c_uint64(size),
            ctypes.c_uint32(kind),
        )

    def get_stream(self):
        return self.runtime.rtCtxGetCurrentDefaultStream()

    def synchronize(self):
        return self.runtime.rtDeviceSynchronize()

    def launch_kernel(self, func_handle, block_dim, args_array, stream):
        args_size = len(args_array) * 8
        return self.runtime.rtKernelLaunch(
            ctypes.c_void_p(func_handle),
            ctypes.c_uint32(block_dim),
            ctypes.byref(args_array),
            ctypes.c_uint32(args_size),
            ctypes.c_void_p(0),
            stream,
        )


# ================================================================
# Tiling 计算（简化版）
# ================================================================

def calculate_tiling():
    """计算 tiling 数据"""
    block_dim = 1
    z1t_size = 31
    z0z1Tb_size = 20

    tiling_values = [
        block_dim,   # block_dim
        1,           # corenum
        1264,        # ub_size
        0,           # hbm_size
        0,           # tiling_key
        z1t_size,    # z1t_size
        z0z1Tb_size, # z0z1Tb_size
        0,           # z0t_size
        0,           # z0Tb_size
        128,         # q0_size
        128,         # q1_size
        128,         # q2_size
        128,         # b0_size
        0,           # b1_size
        8192,        # tmp_tbuf_size
    ]

    import struct
    return b''.join(struct.pack('<I', v) for v in tiling_values)


# ================================================================
# 主测试函数
# ================================================================

def run_test():
    """运行测试"""
    print("=" * 70)
    print("Ascend Runtime 库测试")
    print("=" * 70)

    # 步骤 1: 读取 ELF Binary
    print("\n[步骤 1] 读取 ELF Binary")
    binary_file = "./hash_copy_asc_graph.bin"
    if not os.path.exists(binary_file):
        print(f"  ❌ 文件不存在: {binary_file}")
        return False

    with open(binary_file, "rb") as f:
        kernel_binary = f.read()
    print(f"  ✅ 读取成功: {len(kernel_binary)} bytes")

    # 步骤 2: 准备数据
    print("\n[步骤 2] 准备数据")
    torch.manual_seed(42)
    input0 = torch.rand(20, 31, dtype=torch.float32)
    input1 = torch.rand(1, 31, dtype=torch.float32)
    broadcasted = input1.expand(20, 31)
    add_result = input0 + broadcasted
    mul_result = add_result * broadcasted
    expected_output = add_result - mul_result
    print(f"  ✅ 生成测试数据")

    # 步骤 3: 初始化 Runtime
    print("\n[步骤 3] 初始化 Runtime")
    runtime = SimpleRuntime()
    print(f"  ✅ Runtime 加载成功")

    # 步骤 4: 设置设备
    print("\n[步骤 4] 设置设备")
    ret = runtime.set_device(0)
    print(f"  ✅ 设备设置成功 (返回码: {ret})")

    # 步骤 5: 注册 Kernel
    print("\n[步骤 5] 注册 Kernel")
    kernel_handle = runtime.register_binary(kernel_binary)
    print(f"  ✅ Kernel 注册成功: 0x{kernel_handle:X}")

    # 步骤 6: 注册函数
    print("\n[步骤 6] 注册函数")
    function_handle = runtime.register_function(kernel_handle, "hash_copy_asc_graph")
    print(f"  ✅ 函数注册成功: 0x{function_handle:X}")

    # 步骤 7: 分配内存
    print("\n[步骤 7] 分配内存")
    input0_bytes = input0.numpy().tobytes()
    input1_bytes = input1.numpy().tobytes()
    output_bytes = np.zeros((20, 31), dtype=np.float32).tobytes()

    input0_addr = runtime.malloc(len(input0_bytes))
    input1_addr = runtime.malloc(len(input1_bytes))
    output_addr = runtime.malloc(len(output_bytes))
    print(f"  ✅ 内存分配成功")

    # 步骤 8: 拷贝数据到设备
    print("\n[步骤 8] 拷贝数据")
    input0_ptr = (ctypes.c_char * len(input0_bytes)).from_buffer_copy(input0_bytes)
    input1_ptr = (ctypes.c_char * len(input1_bytes)).from_buffer_copy(input1_bytes)

    runtime.memcpy(input0_addr, ctypes.addressof(input0_ptr), len(input0_bytes), 1)
    runtime.memcpy(input1_addr, ctypes.addressof(input1_ptr), len(input1_bytes), 1)
    print(f"  ✅ 数据拷贝完成")

    # 步骤 9: 计算 Tiling
    print("\n[步骤 9] 计算 Tiling")
    tiling_bytes = calculate_tiling()
    print(f"  ✅ Tiling 数据: {len(tiling_bytes)} bytes")

    # 步骤 10: 构建参数
    print("\n[步骤 10] 构建参数")
    args_list = [input0_addr, input1_addr, output_addr, 0]
    for i in range(0, len(tiling_bytes), 8):
        word = tiling_bytes[i:i+8]
        word = word + b'\\x00' * (8 - len(word))
        val = int.from_bytes(word, 'little')
        args_list.append(val)

    args_array = (ctypes.c_uint64 * len(args_list))(*args_list)
    print(f"  ✅ 参数数量: {len(args_list)}")

    # 步骤 11: 启动 Kernel
    print("\n[步骤 11] 启动 Kernel")
    stream = runtime.get_stream()
    ret = runtime.launch_kernel(function_handle, 1, args_array, stream)
    print(f"  ✅ Kernel 启动成功 (返回码: {ret})")

    # 同步
    runtime.synchronize()
    print(f"  ✅ 同步完成")

    # 步骤 12: 拷贝输出
    print("\n[步骤 12] 拷贝输出")
    output_buffer = (ctypes.c_char * len(output_bytes))()
    runtime.memcpy(ctypes.addressof(output_buffer), output_addr, len(output_bytes), 2)
    output_array = np.frombuffer(output_buffer, dtype=np.float32).reshape(20, 31)
    print(f"  ✅ 输出拷贝完成")

    # 步骤 13: 验证结果
    print("\n[步骤 13] 验证结果")
    output_torch = torch.from_numpy(output_array)
    abs_diff = torch.abs(output_torch - expected_output)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()

    print(f"    最大绝对误差: {max_diff:.6e}")
    print(f"    平均绝对误差: {mean_diff:.6e}")

    is_close = torch.allclose(output_torch, expected_output, rtol=1e-5, atol=1e-5)

    # 清理
    runtime.free(input0_addr)
    runtime.free(input1_addr)
    runtime.free(output_addr)

    return is_close


# ================================================================
# 主入口
# ================================================================

if __name__ == "__main__":
    print()
    try:
        success = run_test()
        print()
        print("=" * 70)
        if success:
            print("✅✅✅ 测试通过！Kernel 执行成功！ ✅✅✅")
        else:
            print("❌❌❌ 测试失败 ❌❌❌")
        print("=" * 70)
        # 立即退出，避免清理时的段错误
        os._exit(0 if success else 1)
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ 测试异常: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        os._exit(1)
