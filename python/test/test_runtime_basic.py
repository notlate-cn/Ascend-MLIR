#!/usr/bin/env python3
"""
Ascend Runtime 库测试脚本

独立实现完整测试流程，避免导入导致的段错误
"""
import os
import sys
from pathlib import Path

# 添加 runtime 路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from runtime.utils import (
    find_runtime_library
)

# 设置环境（必须在其他导入前）
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
        self.runtime_path = find_runtime_library()
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
        # 参考pyasc实现，内存需要512字节对齐
        real_size = size + 512
        c_memory_p = ctypes.c_void_p()
        ret = self.runtime.rtMalloc(
            ctypes.byref(c_memory_p),
            ctypes.c_uint64(real_size),
            ctypes.c_uint32(0),
            ctypes.c_uint16(33),
        )
        if ret != 0:
            return None
        # 对齐到512字节边界
        raw_addr = c_memory_p.value
        # 512字节对齐：地址应该是512的整数倍
        # 向上取整到下一个512的倍数
        aligned_addr = ((raw_addr + 512 - 1) // 512) * 512
        # 保存原始地址用于释放
        if not hasattr(self, '_alloc_map'):
            self._alloc_map = {}
        self._alloc_map[aligned_addr] = raw_addr
        # 调试输出
        print(f"    [DEBUG malloc] size={size}, raw=0x{raw_addr:x}, aligned=0x{aligned_addr:x}, aligned%512={aligned_addr%512}")
        return aligned_addr

    def free(self, ptr):
        # 获取原始地址用于释放
        raw_addr = self._alloc_map.get(ptr, ptr)
        self.runtime.rtFree(ctypes.c_void_p(raw_addr))

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
        block_dim,  # block_dim
        1,  # corenum
        1264,  # ub_size
        0,  # hbm_size
        0,  # tiling_key
        z1t_size,  # z1t_size
        z0z1Tb_size,  # z0z1Tb_size
        0,  # z0t_size
        0,  # z0Tb_size
        128,  # q0_size
        128,  # q1_size
        128,  # q2_size
        128,  # b0_size
        0,  # b1_size
        8192,  # tmp_tbuf_size
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
    workspace_addr = runtime.malloc(8192)  # workspace是必需的
    print(f"  ✅ 内存分配成功")

    # 步骤 8: 拷贝数据到设备（使用分块拷贝）
    print("\n[步骤 8] 拷贝数据")
    # 验证输入数据
    import struct
    first_val = struct.unpack('<f', input0_bytes[:4])[0]
    print(f"    [DEBUG] input0[0] (host): {first_val}")

    # 分块拷贝input0（256字节每块）
    chunk_size = 256
    for offset in range(0, len(input0_bytes), chunk_size):
        chunk_bytes = min(chunk_size, len(input0_bytes) - offset)
        chunk_ptr = (ctypes.c_char * chunk_bytes).from_buffer_copy(input0_bytes[offset:offset+chunk_bytes])
        ret0 = runtime.memcpy(input0_addr + offset, ctypes.addressof(chunk_ptr), chunk_bytes, 1)

    # 拷贝input1（小数据，直接拷贝）
    input1_ptr = (ctypes.c_char * len(input1_bytes)).from_buffer_copy(input1_bytes)
    ret1 = runtime.memcpy(input1_addr, ctypes.addressof(input1_ptr), len(input1_bytes), 1)
    print(f"  ✅ 数据拷贝完成 (ret1={ret1})")

    # 验证：手动将一些数据写入output地址，然后读回验证memcpy是否工作
    test_pattern = struct.pack('<f', 123.456)
    test_ptr = (ctypes.c_char * 4).from_buffer_copy(test_pattern)
    runtime.memcpy(output_addr, ctypes.addressof(test_ptr), 4, 1)  # H2D
    read_back = (ctypes.c_char * 4)()
    runtime.memcpy(ctypes.addressof(read_back), output_addr, 4, 2)  # D2H
    read_val = struct.unpack('<f', bytes(read_back))[0]
    print(f"    [DEBUG] round-trip test: write=123.456, read={read_val}")
    if abs(read_val - 123.456) < 0.001:
        print(f"    [DEBUG] ✓ memcpy工作正常！")
    else:
        print(f"    [DEBUG] ✗ memcpy失败！")

    # 步骤 9: 计算 Tiling
    print("\n[步骤 9] 计算 Tiling")
    tiling_bytes = calculate_tiling()
    print(f"  ✅ Tiling 数据: {len(tiling_bytes)} bytes")

    # 步骤 10: 构建参数
    print("\n[步骤 10] 构建参数")
    args_list = [input0_addr, input1_addr, output_addr, workspace_addr]
    for i in range(0, len(tiling_bytes), 8):
        word = tiling_bytes[i:i + 8]
        word = word + b'\\x00' * (8 - len(word))
        val = int.from_bytes(word, 'little')
        args_list.append(val)

    args_array = (ctypes.c_uint64 * len(args_list))(*args_list)
    print(f"  ✅ 参数数量: {len(args_list)}")
    # 打印前4个参数（地址参数）
    for i in range(min(4, len(args_list))):
        print(f"    args[{i}] = 0x{args_list[i]:016x}")

    # 步骤 11: 启动 Kernel
    print("\n[步骤 11] 启动 Kernel")
    stream = runtime.get_stream()
    ret = runtime.launch_kernel(function_handle, 1, args_array, stream)
    print(f"  ✅ Kernel 启动成功 (返回码: {ret})")

    # 同步
    runtime.synchronize()
    print(f"  ✅ 同步完成")

    # 验证：检查kernel执行后output地址的内容
    check_ptr = (ctypes.c_char * 4)()
    runtime.memcpy(ctypes.addressof(check_ptr), output_addr, 4, 2)  # D2H
    check_val = struct.unpack('<f', bytes(check_ptr))[0]
    print(f"    [DEBUG] output[0] immediately after kernel: {check_val}")

    # 步骤 12: 拷贝输出
    print("\n[步骤 12] 拷贝输出")

    # 再次验证output地址的内容（确保数据还在）
    verify_ptr = (ctypes.c_char * 4)()
    runtime.memcpy(ctypes.addressof(verify_ptr), output_addr, 4, 2)
    verify_val = struct.unpack('<f', bytes(verify_ptr))[0]
    print(f"    [DEBUG] output[0] before final copy: {verify_val}")

    # 分块拷贝输出数据
    # 使用与verify_ptr相同的方式：每个4字节单独分配缓冲区
    print(f"    正在使用逐元素拷贝...")
    output_list = []
    for i in range(0, len(output_bytes), 4):
        elem_ptr = (ctypes.c_char * 4)()
        ret = runtime.memcpy(
            ctypes.addressof(elem_ptr),
            output_addr + i,
            4,
            2  # DEVICE_TO_HOST
        )
        if ret != 0:
            print(f"    [DEBUG] Element {i//4} memcpy failed with ret={ret}")
        output_list.append(bytes(elem_ptr))
        # 验证前几个元素
        if i == 0:
            first_val = struct.unpack('<f', bytes(elem_ptr))[0]
            print(f"    [DEBUG] Element 0: {first_val}, ret={ret}")
        elif i == 4:
            second_val = struct.unpack('<f', bytes(elem_ptr))[0]
            print(f"    [DEBUG] Element 1: {second_val}, ret={ret}")
        elif i == 620 - 4:  # 最后一个元素
            last_val = struct.unpack('<f', bytes(elem_ptr))[0]
            print(f"    [DEBUG] Element 619: {last_val}, ret={ret}")
    output_buffer_bytes = b''.join(output_list)
    print(f"  ✅ 输出拷贝完成 (逐元素拷贝, {len(output_list)} elements)")
    # 转换为numpy array
    output_array = np.frombuffer(output_buffer_bytes, dtype=np.float32).reshape(20, 31)

    # 检查
    print(f"    [DEBUG] output[0] (from d2h): {output_array[0, 0]}")

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
    runtime.free(workspace_addr)

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
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if success else 1)
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ 测试异常: {e}")
        print("=" * 70)
        import traceback
        import sys

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
