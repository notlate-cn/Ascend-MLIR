"""
Ascend Kernel 执行器

通过 ctypes 调用 C++ Executor (lib/Runtime/Executor.cpp) 的 C API。
"""
import ctypes
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from .utils import (
    find_ascend_root,
    find_runtime_library,
    find_afirt_capi_lib,
    setup_environment,
    logger,
    ExecutorError,
    MemoryAllocationError,
    KernelLaunchError,
    InvalidBinaryError,
    ASCEND_A2
)


# ================================================================
# 懒加载 C API 库
# ================================================================

_lib = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        lib_path = find_afirt_capi_lib()
        _lib = ctypes.CDLL(str(lib_path))
        _setup_api(_lib)
    return _lib


def _setup_api(lib: ctypes.CDLL):
    """设置 C API 函数签名"""
    # afirt_executor_create / destroy
    lib.afirt_executor_create.argtypes = []
    lib.afirt_executor_create.restype  = ctypes.c_void_p

    lib.afirt_executor_destroy.argtypes = [ctypes.c_void_p]
    lib.afirt_executor_destroy.restype  = None

    # afirt_executor_initialize
    lib.afirt_executor_initialize.argtypes = [
        ctypes.c_void_p,  # executor
        ctypes.c_int,     # device_id
        ctypes.c_char_p,  # err_buf
        ctypes.c_size_t,  # err_len
    ]
    lib.afirt_executor_initialize.restype = ctypes.c_int

    # afirt_executor_run
    lib.afirt_executor_run.argtypes = [
        ctypes.c_void_p,                     # executor
        ctypes.POINTER(ctypes.c_uint8),      # binary_data
        ctypes.c_size_t,                     # binary_len
        ctypes.c_char_p,                     # function_name
        ctypes.c_int,                        # num_inputs
        ctypes.POINTER(ctypes.c_void_p),     # input_ptrs
        ctypes.POINTER(ctypes.c_size_t),     # input_bytes
        ctypes.c_int,                        # num_outputs
        ctypes.POINTER(ctypes.c_void_p),     # output_ptrs
        ctypes.POINTER(ctypes.c_size_t),     # output_bytes
        ctypes.POINTER(ctypes.c_uint8),      # tiling_data
        ctypes.c_size_t,                     # tiling_len
        ctypes.c_int,                        # block_dim
        ctypes.c_uint32,                     # magic
        ctypes.c_char_p,                     # err_buf
        ctypes.c_size_t,                     # err_len
    ]
    lib.afirt_executor_run.restype = ctypes.c_int

    # afirt_executor_run_file
    lib.afirt_executor_run_file.argtypes = [
        ctypes.c_void_p,                     # executor
        ctypes.c_char_p,                     # binary_path
        ctypes.c_char_p,                     # function_name
        ctypes.c_int,                        # num_inputs
        ctypes.POINTER(ctypes.c_void_p),     # input_ptrs
        ctypes.POINTER(ctypes.c_size_t),     # input_bytes
        ctypes.c_int,                        # num_outputs
        ctypes.POINTER(ctypes.c_void_p),     # output_ptrs
        ctypes.POINTER(ctypes.c_size_t),     # output_bytes
        ctypes.POINTER(ctypes.c_uint8),      # tiling_data
        ctypes.c_size_t,                     # tiling_len
        ctypes.c_int,                        # block_dim
        ctypes.c_uint32,                     # magic
        ctypes.c_char_p,                     # err_buf
        ctypes.c_size_t,                     # err_len
    ]
    lib.afirt_executor_run_file.restype = ctypes.c_int


# ================================================================
# Magic 常量（保持与 C++ 一致）
# ================================================================

MAGIC_ELF_AIVEC  = 0x41415246
MAGIC_ELF_AICUBE = 0x41494343
MAGIC_ELF_AICPU  = 0x41415243


# ================================================================
# DevBinary — 保留供外部代码引用（已不再需要直接使用）
# ================================================================

class DevBinary(ctypes.Structure):
    _fields_ = [
        ("magic",   ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("data",    ctypes.c_char_p),
        ("length",  ctypes.c_uint64),
    ]


# ================================================================
# AscendRuntime — 薄封装，调用 C API
# ================================================================

class AscendRuntime:
    """
    Ascend Runtime 封装类（通过 C API 调用 C++ Executor）
    """

    MAGIC_ELF_AIVEC  = MAGIC_ELF_AIVEC
    MAGIC_ELF_AICUBE = MAGIC_ELF_AICUBE
    MAGIC_ELF_AICPU  = MAGIC_ELF_AICPU

    def __init__(self,
                 ascend_root=None,
                 soc_version: str = ASCEND_A2,
                 simulation_mode: bool = True):
        if ascend_root is None:
            ascend_root = find_ascend_root()
            if ascend_root is None:
                raise ExecutorError("Cannot find Ascend installation")

        setup_environment(ascend_root, soc_version, simulation_mode)

        self.ascend_root    = ascend_root
        self.soc_version    = soc_version
        self._simulation    = simulation_mode
        self._initialized   = False

        lib = _get_lib()
        handle = lib.afirt_executor_create()
        if not handle:
            raise ExecutorError("Failed to create C++ Executor instance")
        self._handle = handle

    def __del__(self):
        if getattr(self, "_handle", None):
            try:
                _get_lib().afirt_executor_destroy(ctypes.c_void_p(self._handle))
            except Exception:
                pass
            self._handle = None

    def set_device(self, device_id: int = 0):
        lib    = _get_lib()
        err    = ctypes.create_string_buffer(1024)
        ret    = lib.afirt_executor_initialize(
            ctypes.c_void_p(self._handle), device_id, err, ctypes.c_size_t(len(err))
        )
        if ret != 0:
            logger.info(f"afirt_executor_initialize returned {ret}: {err.value.decode(errors='replace')}")
        self._initialized = True

    def run(self,
            binary_data: bytes,
            function_name: str,
            inputs: List[np.ndarray],
            outputs: List[np.ndarray],
            tiling_data: Optional[bytes] = None,
            block_dim: int = 1,
            magic: int = MAGIC_ELF_AIVEC) -> List[np.ndarray]:
        """
        执行 kernel（通过 C API 的 afirt_executor_run）
        """
        lib = _get_lib()

        # binary
        bin_arr = (ctypes.c_uint8 * len(binary_data)).from_buffer_copy(binary_data)

        # inputs: keep contiguous copies alive
        in_bufs   = [np.ascontiguousarray(x) for x in inputs]
        in_ptrs   = (ctypes.c_void_p * len(in_bufs))(
            *[x.ctypes.data_as(ctypes.c_void_p) for x in in_bufs]
        )
        in_bytes  = (ctypes.c_size_t * len(in_bufs))(*[x.nbytes for x in in_bufs])

        # outputs: pre-allocated writable buffers
        out_bufs  = [np.ascontiguousarray(x) for x in outputs]
        out_ptrs  = (ctypes.c_void_p * len(out_bufs))(
            *[x.ctypes.data_as(ctypes.c_void_p) for x in out_bufs]
        )
        out_bytes = (ctypes.c_size_t * len(out_bufs))(*[x.nbytes for x in out_bufs])

        # tiling
        if tiling_data and len(tiling_data) > 0:
            tiling_arr = (ctypes.c_uint8 * len(tiling_data)).from_buffer_copy(tiling_data)
            tiling_ptr = ctypes.cast(tiling_arr, ctypes.POINTER(ctypes.c_uint8))
            tiling_len = len(tiling_data)
        else:
            tiling_ptr = ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_uint8))
            tiling_len = 0

        err = ctypes.create_string_buffer(1024)
        ret = lib.afirt_executor_run(
            ctypes.c_void_p(self._handle),
            ctypes.cast(bin_arr, ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(len(binary_data)),
            function_name.encode(),
            ctypes.c_int(len(in_bufs)),
            in_ptrs,
            in_bytes,
            ctypes.c_int(len(out_bufs)),
            out_ptrs,
            out_bytes,
            tiling_ptr,
            ctypes.c_size_t(tiling_len),
            ctypes.c_int(block_dim),
            ctypes.c_uint32(magic),
            err,
            ctypes.c_size_t(len(err)),
        )
        if ret != 0:
            raise ExecutorError(err.value.decode(errors="replace"))

        # copy results back into the original output arrays
        for dst, src in zip(outputs, out_bufs):
            dst[:] = src
        return outputs


# ================================================================
# KernelExecutor — higher-level interface
# ================================================================

class KernelExecutor:
    """Kernel 执行器（调用 C++ Executor via C API）"""

    def __init__(self,
                 ascend_root=None,
                 soc_version: str = ASCEND_A2,
                 device_id: int = 0,
                 simulation_mode: bool = True):
        self.runtime  = AscendRuntime(ascend_root, soc_version, simulation_mode)
        self.device_id = device_id
        self._initialized = False

    def initialize(self):
        self.runtime.set_device(self.device_id)
        self._initialized = True

    def execute(self,
                binary_data: bytes,
                function_name: str,
                inputs: List[np.ndarray],
                outputs: List[np.ndarray],
                tiling_data: Optional[bytes] = None,
                block_dim: int = 1,
                magic: int = MAGIC_ELF_AIVEC) -> List[np.ndarray]:
        if not self._initialized:
            self.initialize()
        return self.runtime.run(
            binary_data, function_name, inputs, outputs,
            tiling_data, block_dim, magic
        )


# ================================================================
# 便捷函数
# ================================================================

def execute_kernel(binary_data: bytes,
                   function_name: str,
                   inputs: List[np.ndarray],
                   output_shapes: List[Tuple[int, ...]],
                   tiling_data: Optional[bytes] = None,
                   ascend_root=None,
                   soc_version: str = ASCEND_A2,
                   device_id: int = 0,
                   dtype: np.dtype = np.float32,
                   simulation_mode: bool = True) -> List[np.ndarray]:
    """执行 kernel（便捷函数）"""
    outputs  = [np.zeros(shape, dtype=dtype) for shape in output_shapes]
    executor = KernelExecutor(ascend_root, soc_version, device_id, simulation_mode)
    return executor.execute(binary_data, function_name, inputs, outputs, tiling_data)
