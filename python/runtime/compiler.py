"""
Bisheng 编译器接口

通过 ctypes 调用 C++ Compiler (lib/Runtime/Compiler.cpp) 的 C API。
"""
import ctypes
import os
from pathlib import Path
from typing import Optional

from .utils import (
    find_ascend_root,
    setup_environment,
    find_afirt_capi_lib,
    logger,
    CompilerError,
    BinaryNotFoundError,
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
    # afirt_compiler_create
    lib.afirt_compiler_create.argtypes = [
        ctypes.c_char_p,  # soc_version
        ctypes.c_char_p,  # arch
        ctypes.c_int,     # opt_level
    ]
    lib.afirt_compiler_create.restype = ctypes.c_void_p

    # afirt_compiler_destroy
    lib.afirt_compiler_destroy.argtypes = [ctypes.c_void_p]
    lib.afirt_compiler_destroy.restype = None

    # afirt_compiler_compile
    lib.afirt_compiler_compile.argtypes = [
        ctypes.c_void_p,  # compiler
        ctypes.c_char_p,  # src_file
        ctypes.c_char_p,  # output_dir
        ctypes.c_char_p,  # kernel_name
        ctypes.c_char_p,  # out_bin_path (output buffer)
        ctypes.c_size_t,  # buf_len
    ]
    lib.afirt_compiler_compile.restype = ctypes.c_int


# ================================================================
# BishengCompiler
# ================================================================

class BishengCompiler:
    """
    Bisheng 编译器封装类（通过 C API 调用 C++ Compiler）
    """

    ARCH_VEC  = "dav-c220-vec"
    ARCH_CUBE = "dav-c220-cube"

    def __init__(self,
                 ascend_root=None,
                 soc_version: str = ASCEND_A2):
        if ascend_root is None:
            ascend_root = find_ascend_root()
        setup_environment(ascend_root, soc_version)

        self.ascend_root = ascend_root
        self.soc_version = soc_version
        self._arch = self.ARCH_VEC
        self._opt_level = 3
        self._handle = None  # allocated lazily per compile call

    def _create_handle(self, arch: str, opt_level: int) -> int:
        lib    = _get_lib()
        handle = lib.afirt_compiler_create(
            self.soc_version.encode(),
            arch.encode(),
            opt_level,
        )
        if not handle:
            raise CompilerError("Failed to create C++ Compiler instance")
        return handle

    def compile_and_link(self,
                         src_file,
                         output_dir,
                         kernel_name: str,
                         arch: str = ARCH_VEC,
                         opt_level: int = 3,
                         verbose: bool = False) -> Path:
        """
        编译并链接生成 ELF binary

        Args:
            src_file: 源文件路径
            output_dir: 输出目录
            kernel_name: kernel 名称
            arch: 目标架构
            opt_level: 优化级别
            verbose: 是否显示详细输出（C++ 层 verbose 通过 CompilerConfig 控制，此处忽略）

        Returns:
            ELF binary 文件路径
        """
        src_file   = Path(src_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        lib    = _get_lib()
        handle = self._create_handle(arch, opt_level)
        try:
            buf     = ctypes.create_string_buffer(4096)
            ret     = lib.afirt_compiler_compile(
                ctypes.c_void_p(handle),
                str(src_file.resolve()).encode(),
                str(output_dir.resolve()).encode(),
                kernel_name.encode(),
                buf,
                ctypes.c_size_t(len(buf)),
            )
            if ret != 0:
                raise CompilerError(buf.value.decode(errors="replace"))
            bin_path = Path(buf.value.decode())
        finally:
            lib.afirt_compiler_destroy(ctypes.c_void_p(handle))

        if not bin_path.exists():
            raise BinaryNotFoundError(f"Binary file not generated: {bin_path}")

        logger.info(f"Generated binary: {bin_path} ({bin_path.stat().st_size} bytes)")
        return bin_path


# ================================================================
# 便捷函数
# ================================================================

def compile_kernel(src_file,
                   output_dir,
                   kernel_name: str,
                   ascend_root=None,
                   soc_version: str = ASCEND_A2,
                   arch: str = BishengCompiler.ARCH_VEC,
                   opt_level: int = 3,
                   verbose: bool = False) -> Path:
    """
    编译 AscendC kernel 为 ELF binary（调用 C++ Compiler via C API）
    """
    compiler = BishengCompiler(ascend_root, soc_version)
    return compiler.compile_and_link(
        src_file=src_file,
        output_dir=output_dir,
        kernel_name=kernel_name,
        arch=arch,
        opt_level=opt_level,
        verbose=verbose,
    )
