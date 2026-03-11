"""
Ascend Kernel 执行器 (无 PyAsc 依赖)

使用 ctypes 直接调用 libruntime_camodel.so 执行 Bisheng 编译的 ELF binary。

主要功能:
1. AscendRuntime - Runtime API 封装类
2. KernelExecutor - Kernel 执行器类
3. execute_kernel - 便捷的执行函数
"""
import ctypes
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np

from .utils import (
    find_ascend_root,
    find_runtime_library,
    setup_environment,
    logger,
    ExecutorError,
    MemoryAllocationError,
    KernelLaunchError,
    InvalidBinaryError,
    ASCEND_A2
)


# ================================================================
# DevBinary 结构体
# ================================================================

class DevBinary(ctypes.Structure):
    """
    Device Binary 结构体

    对应 runtime 的 DevBinary 结构，用于注册 ELF binary。
    字段顺序必须正确！
    """
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("data", ctypes.c_char_p),
        ("length", ctypes.c_uint64),
    ]


# ================================================================
# Ascend Runtime 封装类
# ================================================================

class AscendRuntime:
    """
    Ascend Runtime 封装类

    使用 ctypes 直接调用 libruntime_camodel.so 的 API。
    """

    # ELF Magic 值
    MAGIC_ELF_AIVEC = 0x41415246  # "BARF" - VectorCore kernel
    MAGIC_ELF_AICUBE = 0x41494343  # "AICC" - AiCore kernel
    MAGIC_ELF_AICPU = 0x41415243  # "BARC" - AICPU kernel

    def __init__(self,
                 ascend_root: Optional[Path] = None,
                 soc_version: str = ASCEND_A2,
                 simulation_mode: bool = True):
        """
        初始化 Runtime

        Args:
            ascend_root: Ascend 根目录
            soc_version: SOC 版本
            simulation_mode: 是否使用 CPU 仿真模式（默认 True）

        Raises:
            ExecutorError: Runtime 初始化失败
        """
        # 设置环境
        if ascend_root is None:
            ascend_root = find_ascend_root()
            if ascend_root is None:
                raise ExecutorError("Cannot find Ascend installation")

        setup_environment(ascend_root, soc_version, simulation_mode)

        self.ascend_root = ascend_root
        self.soc_version = soc_version
        self._simulation_mode = simulation_mode
        self._initialized = False
        self._device_set = False
        self._context = None

        # 加载 runtime 库
        self._load_runtime()

    def _load_runtime(self):
        """加载 libruntime_camodel.so"""
        runtime_path = find_runtime_library(self.ascend_root, self.soc_version)
        if runtime_path is None:
            raise ExecutorError(
                f"Cannot find libruntime_camodel.so for {self.soc_version}"
            )

        # # 对于 CPU 仿真模式，需要预加载驱动库
        # if self._simulation_mode:
        #     simulator_lib_dir = runtime_path.parent
        #     driver_libs = [
        #         "libnpu_drv_camodel.so",
        #         "libnpu_drv.so",
        #     ]
        #
        #     for lib_name in driver_libs:
        #         lib_path = simulator_lib_dir / lib_name
        #         if lib_path.exists():
        #             try:
        #                 logger.info(f"Preloading driver library: {lib_path}")
        #                 ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        #             except Exception as e:
        #                 logger.warning(f"Failed to preload {lib_name}: {e}")

        logger.info(f"Loading runtime: {runtime_path}")
        self.runtime = ctypes.CDLL(str(runtime_path), mode=ctypes.RTLD_GLOBAL)
        self._setup_api()
        self._initialized = True

    def _setup_api(self):
        """设置 API 函数签名"""
        # rtSetDevice
        self.runtime.rtSetDevice.argtypes = [ctypes.c_int32]
        self.runtime.rtSetDevice.restype = ctypes.c_int

        # rtDevBinaryRegister
        self.runtime.rtDevBinaryRegister.argtypes = [
            ctypes.POINTER(DevBinary),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.runtime.rtDevBinaryRegister.restype = ctypes.c_int

        # rtFunctionRegister
        self.runtime.rtFunctionRegister.argtypes = [
            ctypes.c_void_p,  # binHandle
            ctypes.c_void_p,  # stubFunc
            ctypes.c_char_p,  # stubName
            ctypes.c_void_p,  # kernelInfoExt
            ctypes.c_uint32,  # funcMode
        ]
        self.runtime.rtFunctionRegister.restype = ctypes.c_int

        # rtKernelLaunch
        self.runtime.rtKernelLaunch.argtypes = [
            ctypes.c_void_p,  # stubFunc
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # args
            ctypes.c_uint32,  # argsSize
            ctypes.c_void_p,  # smDesc
            ctypes.c_void_p,  # stream
        ]
        self.runtime.rtKernelLaunch.restype = ctypes.c_int

        # rtMalloc
        self.runtime.rtMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint16,
        ]
        self.runtime.rtMalloc.restype = ctypes.c_int

        # rtFree
        self.runtime.rtFree.argtypes = [ctypes.c_void_p]
        self.runtime.rtFree.restype = ctypes.c_int

        # rtMemcpy - 签名: (dst, destMax, src, cnt, kind)
        self.runtime.rtMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
        ]
        self.runtime.rtMemcpy.restype = ctypes.c_int

        # rtCtxCreate
        self.runtime.rtCtxCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # pctx
            ctypes.c_uint32,  # flags
            ctypes.c_int32,  # device
        ]
        self.runtime.rtCtxCreate.restype = ctypes.c_int

        # rtCtxDestroy
        self.runtime.rtCtxDestroy.argtypes = [ctypes.c_void_p]
        self.runtime.rtCtxDestroy.restype = ctypes.c_int

        # rtCtxSetCurrent
        self.runtime.rtCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self.runtime.rtCtxSetCurrent.restype = ctypes.c_int

        # rtStreamCreate - 创建stream (不直接使用get_default_stream)
        self.runtime.rtStreamCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int32,
        ]
        self.runtime.rtStreamCreate.restype = ctypes.c_int

        # rtStreamDestroy - 销毁stream
        self.runtime.rtStreamDestroy.argtypes = [ctypes.c_void_p]
        self.runtime.rtStreamDestroy.restype = ctypes.c_int

        # rtCtxGetCurrentDefaultStream (仅作为备用，通常不直接使用)
        self.runtime.rtCtxGetCurrentDefaultStream.argtypes = []
        self.runtime.rtCtxGetCurrentDefaultStream.restype = ctypes.c_void_p

        # rtDeviceSynchronize
        self.runtime.rtDeviceSynchronize.argtypes = []
        self.runtime.rtDeviceSynchronize.restype = ctypes.c_int

    # ============================================================
    # 设备管理
    # ============================================================

    def set_device(self, device_id: int = 0):
        """
        设置设备

        Args:
            device_id: 设备 ID

        Note:
            参考 test_runtime_basic.py 的实现：
            在 camodel 仿真模式下，rtSetDevice 可能返回错误码，但这不影响后续操作。
            模拟器会自动初始化必要的上下文，无需手动创建 context。
            关键是不要因为 rtSetDevice 的错误而抛出异常或尝试手动创建 context。
        """
        ret = self.runtime.rtSetDevice(ctypes.c_int32(device_id))

        # 不检查返回值，参考 test_runtime_basic.py 的实现
        # 即使返回错误码，camodel 也会自动处理，后续操作仍然可以正常工作
        if ret != 0:
            logger.info(f"rtSetDevice returned {ret} (normal in simulation mode)")
        else:
            logger.info(f"Device {device_id} set successfully")

        self._device_set = True

    # ============================================================
    # 内存管理
    # ============================================================

    def malloc(self, size: int) -> int:
        """
        分配设备内存（512字节对齐）

        Args:
            size: 大小（字节）

        Returns:
            设备内存地址（已对齐）

        Raises:
            MemoryAllocationError: 内存分配失败

        Note:
            参考pyasc实现，内存需要512字节对齐
        """
        # 多分配512字节用于对齐
        real_size = size + 512
        c_memory_p = ctypes.c_void_p()
        ret = self.runtime.rtMalloc(
            ctypes.byref(c_memory_p),
            ctypes.c_uint64(real_size),
            ctypes.c_uint32(0),  # RT_MEMORY_DEFAULT
            ctypes.c_uint16(33),  # moduleId
        )

        if ret != 0:
            raise MemoryAllocationError(f"rtMalloc failed: {ret}")

        # 对齐到512字节边界
        raw_addr = c_memory_p.value
        aligned_addr = 512 * ((raw_addr + 512 - 1) // 512)

        # 保存原始地址用于释放
        if not hasattr(self, '_alloc_map'):
            self._alloc_map = {}
        self._alloc_map[aligned_addr] = raw_addr

        return aligned_addr

    def free(self, ptr: int):
        """
        释放设备内存

        Args:
            ptr: 设备内存地址（对齐后的地址）
        """
        # 获取原始地址用于释放
        raw_addr = self._alloc_map.get(ptr, ptr)
        self.runtime.rtFree(ctypes.c_void_p(raw_addr))

    def memcpy(self, dst: int, src: int, size: int, kind: int = 1):
        """
        内存拷贝

        Args:
            dst: 目标地址
            src: 源地址
            size: 大小（字节）
            kind: 拷贝类型
                0 = HOST_TO_HOST
                1 = HOST_TO_DEVICE
                2 = DEVICE_TO_HOST
                3 = DEVICE_TO_DEVICE

        Raises:
            ExecutorError: 拷贝失败
        """
        ret = self.runtime.rtMemcpy(
            ctypes.c_void_p(dst),
            ctypes.c_uint64(size),
            ctypes.c_void_p(src),
            ctypes.c_uint64(size),
            ctypes.c_uint32(kind),
        )
        if ret != 0:
            raise ExecutorError(f"rtMemcpy failed: {ret}")

    def memcpy_h2d(self, dst: int, src: Union[bytes, np.ndarray], size: int):
        """Host 到 Device 内存拷贝

        Note:
            对于大块数据，使用分块拷贝以避免rtMemcpy的问题
            经测试，256字节的chunk size是安全的
        """
        if isinstance(src, np.ndarray):
            # 将 numpy 数组转换为 bytes
            src_bytes = src.tobytes()

            # 使用分块拷贝（每次最多256字节）
            chunk_size = 256
            for offset in range(0, len(src_bytes), chunk_size):
                chunk_bytes = min(chunk_size, len(src_bytes) - offset)
                src_ptr = (ctypes.c_char * chunk_bytes).from_buffer_copy(src_bytes[offset:offset+chunk_bytes])
                self.memcpy(dst + offset, ctypes.addressof(src_ptr), chunk_bytes, kind=1)
        else:
            src_ptr = ctypes.create_string_buffer(src, size)
            self.memcpy(dst, ctypes.addressof(src_ptr), size, kind=1)

    def memcpy_d2h(self, dst: Union[np.ndarray, bytearray], src: int, size: int):
        """
        Device 到 Host 内存拷贝

        Args:
            dst: 目标缓冲区（numpy array 或 bytearray）
            src: 源设备地址
            size: 拷贝大小（字节）

        Note:
            对于大块数据，使用逐元素拷贝以避免rtMemcpy的问题
            经测试，4字节的元素拷贝是安全的
        """
        import struct
        output_list = []
        for i in range(0, size, 4):
            elem_ptr = (ctypes.c_char * 4)()
            self.memcpy(
                ctypes.addressof(elem_ptr),
                src + i,
                4,
                kind=2
            )
            output_list.append(bytes(elem_ptr))

        if isinstance(dst, np.ndarray):
            # 拷贝到 numpy array
            dst_bytes = b''.join(output_list)
            dst_view = np.frombuffer(dst_bytes, dtype=dst.dtype).reshape(dst.shape)
            dst[:] = dst_view
        else:
            # 拷贝到 bytearray
            dst[:] = bytearray(b''.join(output_list))

    # ============================================================
    # Kernel 管理
    # ============================================================

    def register_binary(self, binary_data: bytes, magic: int = MAGIC_ELF_AIVEC) -> int:
        """
        注册 ELF binary

        Args:
            binary_data: ELF binary 数据
            magic: ELF magic 值

        Returns:
            Kernel handle

        Raises:
            ExecutorError: 注册失败
        """
        kernel_size = len(binary_data)
        if kernel_size <= 0:
            raise InvalidBinaryError("Kernel size must be greater than 0")

        device_binary = DevBinary(
            magic=ctypes.c_uint32(magic),
            version=ctypes.c_uint32(0),
            data=ctypes.c_char_p(binary_data),
            length=ctypes.c_uint64(kernel_size),
        )

        binary_handle = ctypes.c_void_p()
        ret = self.runtime.rtDevBinaryRegister(
            ctypes.byref(device_binary),
            ctypes.byref(binary_handle),
        )

        if ret != 0:
            raise ExecutorError(f"rtDevBinaryRegister failed: {ret}")

        logger.info(f"Binary registered: 0x{binary_handle.value:X}")
        return binary_handle.value

    def register_function(self, kernel_handle: int, function_name: str, mode: int = 0) -> int:
        """
        注册 kernel 函数

        Args:
            kernel_handle: Kernel handle
            function_name: 函数名
            mode: 模式

        Returns:
            函数 handle

        Raises:
            ExecutorError: 注册失败
        """
        fn_name_bytes = function_name.encode("utf-8")
        name_ptr = ctypes.c_char_p(fn_name_bytes)
        name_void_ptr = ctypes.cast(name_ptr, ctypes.c_void_p)

        ret = self.runtime.rtFunctionRegister(
            ctypes.c_void_p(kernel_handle),
            name_void_ptr,
            name_ptr,
            name_void_ptr,
            ctypes.c_uint32(mode),
        )

        if ret != 0:
            raise ExecutorError(f"rtFunctionRegister failed: {ret}")

        logger.info(f"Function registered: {function_name} -> 0x{name_void_ptr.value:X}")
        return name_void_ptr.value

    # ============================================================
    # Kernel 执行
    # ============================================================

    def create_stream(self, priority: int = 0):
        """
        创建stream

        Args:
            priority: Stream 优先级

        Returns:
            Stream handle
        """
        stream_handle = ctypes.c_void_p()
        ret = self.runtime.rtStreamCreate(
            ctypes.byref(stream_handle),
            ctypes.c_int32(priority),
        )

        if ret != 0:
            raise ExecutorError(f"rtStreamCreate failed: {ret}")

        return stream_handle

    def destroy_stream(self, stream: ctypes.c_void_p):
        """
        销毁stream

        Args:
            stream: Stream handle
        """
        self.runtime.rtStreamDestroy(stream)

    def get_stream(self) -> ctypes.c_void_p:
        """
        获取默认stream - 在仿真模式下使用create_stream替代

        注意：rtCtxGetCurrentDefaultStream在某些情况下会段错误
        使用create_stream创建新stream作为替代
        """
        return self.create_stream()

    def synchronize(self):
        """同步设备"""
        ret = self.runtime.rtDeviceSynchronize()
        if ret != 0:
            raise ExecutorError(f"rtDeviceSynchronize failed: {ret}")

    def launch_kernel(self,
                      function_handle: int,
                      block_dim: int,
                      args: List[int],
                      stream: Optional[ctypes.c_void_p] = None):
        """
        启动 kernel

        Args:
            function_handle: 函数 handle
            block_dim: Block 维度
            args: 参数列表（将被转换为 uint64 数组）
            stream: Stream（默认使用默认 stream）

        Raises:
            KernelLaunchError: 启动失败
        """
        # 转换参数为 uint64 数组
        args_array = (ctypes.c_uint64 * len(args))(*args)
        args_size = len(args) * 8

        ret = self.runtime.rtKernelLaunch(
            ctypes.c_void_p(function_handle),
            ctypes.c_uint32(block_dim),
            ctypes.byref(args_array),
            ctypes.c_uint32(args_size),
            ctypes.c_void_p(0),  # smDesc = NULL
            stream if stream is not None else ctypes.c_void_p(0),
        )

        if ret != 0:
            raise KernelLaunchError(f"rtKernelLaunch failed: {ret}")

        logger.debug(f"Kernel launched: block_dim={block_dim}, args={len(args)}")


# ================================================================
# Kernel 执行器类
# ================================================================

class KernelExecutor:
    """
    Kernel 执行器

    提供简化的 kernel 执行接口。
    """

    def __init__(self,
                 ascend_root: Optional[Path] = None,
                 soc_version: str = ASCEND_A2,
                 device_id: int = 0,
                 simulation_mode: bool = True):
        """
        初始化执行器

        Args:
            ascend_root: Ascend 根目录
            soc_version: SOC 版本
            device_id: 设备 ID
            simulation_mode: 是否使用 CPU 仿真模式（默认 True）
        """
        self.runtime = AscendRuntime(ascend_root, soc_version, simulation_mode)
        self.device_id = device_id
        self._initialized = False

    def initialize(self):
        """初始化执行器"""
        self.runtime.set_device(self.device_id)
        self._initialized = True

    def execute(self,
                binary_data: bytes,
                function_name: str,
                inputs: List[np.ndarray],
                outputs: List[np.ndarray],
                tiling_data: Optional[bytes] = None,
                block_dim: int = 1,
                magic: int = AscendRuntime.MAGIC_ELF_AIVEC) -> List[np.ndarray]:
        """
        执行 kernel

        Args:
            binary_data: ELF binary 数据
            function_name: 函数名
            inputs: 输入张量列表
            outputs: 输出张量列表（用于分配内存）
            tiling_data: Tiling 数据
            block_dim: Block 维度
            magic: ELF magic 值

        Returns:
            输出张量列表

        Raises:
            ExecutorError: 执行失败
        """
        if not self._initialized:
            self.initialize()

        # 注册 binary
        kernel_handle = self.runtime.register_binary(binary_data, magic)

        # 注册函数
        function_handle = self.runtime.register_function(kernel_handle, function_name)

        # 分配并拷贝输入内存
        input_addrs = []
        for i, inp in enumerate(inputs):
            addr = self.runtime.malloc(inp.nbytes)
            self.runtime.memcpy_h2d(addr, inp, inp.nbytes)
            input_addrs.append(addr)

        # 分配输出内存
        output_addrs = []
        for out in outputs:
            addr = self.runtime.malloc(out.nbytes)
            output_addrs.append(addr)

        # 分配workspace内存
        workspace_addr = self.runtime.malloc(8192)

        # 构建参数
        args = []
        args.extend(input_addrs)
        args.extend(output_addrs)
        args.append(workspace_addr)  # workspace

        # 添加 tiling data
        if tiling_data:
            for i in range(0, len(tiling_data), 8):
                word = tiling_data[i:i + 8]
                # 填充到 8 字节
                word = word + b'\x00' * (8 - len(word))
                val = int.from_bytes(word, 'little')
                args.append(val)

        # 启动 kernel
        stream = self.runtime.get_stream()
        self.runtime.launch_kernel(function_handle, block_dim, args, stream)

        # 同步
        self.runtime.synchronize()

        # 拷贝输出
        for i, (out, addr) in enumerate(zip(outputs, output_addrs)):
            self.runtime.memcpy_d2h(out, addr, out.nbytes)

        # 清理
        for addr in input_addrs + output_addrs:
            self.runtime.free(addr)
        self.runtime.free(workspace_addr)

        return outputs


# ================================================================
# 便捷函数
# ================================================================

def execute_kernel(binary_data: bytes,
                   function_name: str,
                   inputs: List[np.ndarray],
                   output_shapes: List[Tuple[int, ...]],
                   tiling_data: Optional[bytes] = None,
                   ascend_root: Optional[Path] = None,
                   soc_version: str = ASCEND_A2,
                   device_id: int = 0,
                   dtype: np.dtype = np.float32,
                   simulation_mode: bool = True) -> List[np.ndarray]:
    """
    执行 kernel（便捷函数）

    Args:
        binary_data: ELF binary 数据
        function_name: 函数名
        inputs: 输入张量列表
        output_shapes: 输出形状列表
        tiling_data: Tiling 数据
        ascend_root: Ascend 根目录
        soc_version: SOC 版本
        device_id: 设备 ID
        dtype: 输出数据类型
        simulation_mode: 是否使用 CPU 仿真模式（默认 True）

    Returns:
        输出张量列表

    Example:
        >>> outputs = execute_kernel(
        ...     binary_data=binary,
        ...     function_name="my_kernel",
        ...     inputs=[input_array],
        ...     output_shapes=[(20, 31)],
        ...     tiling_data=tiling_bytes
        ... )
    """
    # 分配输出缓冲区
    outputs = [np.zeros(shape, dtype=dtype) for shape in output_shapes]

    # 创建执行器（默认使用 CPU 仿真模式）
    executor = KernelExecutor(ascend_root, soc_version, device_id, simulation_mode)

    # 执行
    return executor.execute(binary_data, function_name, inputs, outputs, tiling_data)
