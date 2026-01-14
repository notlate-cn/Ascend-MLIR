"""
Ascend Runtime Library - 无 PyAsc 依赖的 Ascend NPU Kernel 执行库

这个库提供了完整的 Ascend NPU kernel 开发和执行流程：
1. 源代码生成 (AscenGen.save_source_files)
2. Host Tiling 编译执行 (compile_and_execute_host_tiling)
3. Bisheng 编译 (BishengCompiler)
4. Kernel 执行 (KernelExecutor, AscendRuntime)
5. Tiling 数据计算 (calculate_tiling)

主要特点:
- 完全移除 PyAsc 依赖
- 使用 ctypes 直接调用 libruntime_camodel.so
- 简洁易用的 API 设计
- 完整的编译-执行流程
- 支持动态编译 host tiling C++ 代码

Example:
    >>> from runtime import (
    ...     save_source_files,
    ...     compile_and_execute_host_tiling,
    ...     compile_kernel,
    ...     execute_kernel,
    ... )
    >>>
    >>> # 1. 从 graph 生成源代码（返回 4 个值）
    >>> host_file, device_file, _, tiling_def = save_source_files(
    ...     graph_text=graph_def,
    ...     output_dir="./build",
    ...     graph_name="my_kernel"
    ... )
    >>>
    >>> # tiling_def 包含 tiling 数据结构的 C++ 定义
    >>> # 例如: "struct AutofuseTilingData { uint32_t block_dim; ... };"
    >>>
    >>> # 2. 编译执行 host tiling 生成 tiling 数据
    >>> tiling_data = calc_tiling_data(
    ...     host_cpp_file=host_file,
    ...     output_shape=(20, 31)
    ... )
    >>>
    >>> # 3. 编译 device kernel
    >>> binary_path = compile_kernel(
    ...     str(device_file),
    ...     "./build",
    ...     "my_kernel"
    ... )
    >>>
    >>> # 4. 读取 binary
    >>> binary_data = read_binary(binary_path)
    >>>
    >>> # 5. 执行 kernel
    >>> outputs = execute_kernel(
    ...     binary_data=binary_data,
    ...     function_name="my_kernel",
    ...     inputs=[input_array],
    ...     output_shapes=[(20, 31)],
    ...     tiling_data=tiling_data
    ... )
"""

__version__ = "0.1.0"
__author__ = "Claude Code"

# ================================================================
# 版本信息
# ================================================================

from . import utils
from . import compiler
from . import executor
from . import ascgen

# ================================================================
# 主要导出
# ================================================================

# 工具模块
from .utils import (
    find_ascend_root,
    find_runtime_library,
    setup_environment,
    logger,
    Logger,
    ProgressBar,
    # 异常
    AscendRuntimeError,
    CompilerError,
    ExecutorError,
    BinaryNotFoundError,
    InvalidBinaryError,
    MemoryAllocationError,
    KernelLaunchError,
)

# 编译器模块
from .compiler import (
    BishengCompiler,
    compile_kernel as compile_kernel_with_bisheng,
)

# Ascgen编译模块
from .ascgen import (
    AscGen,
    save_source_files,
    calc_tiling_data,
)

# 执行器模块
from .executor import (
    DevBinary,
    AscendRuntime,
    KernelExecutor,
    execute_kernel,
)

# ================================================================
# 便捷函数
# ================================================================

def read_binary(file_path: str) -> bytes:
    """
    读取 ELF binary 文件

    Args:
        file_path: 文件路径

    Returns:
        Binary 数据
    """
    with open(file_path, "rb") as f:
        return f.read()


def compile_kernel(src_file: str,
                   output_dir: str,
                   kernel_name: str,
                   **kwargs) -> str:
    """
    编译 AscendC kernel（便捷函数）

    这是对 compiler.compile_kernel 的简化包装。

    Args:
        src_file: 源文件路径
        output_dir: 输出目录
        kernel_name: kernel 名称
        **kwargs: 其他参数传递给 BishengCompiler

    Returns:
        ELF binary 文件路径

    Example:
        >>> binary_path = compile_kernel(
        ...     src_file="device/kernel.cpp",
        ...     output_dir="./build",
        ...     kernel_name="my_kernel"
        ... )
    """
    from pathlib import Path
    result = compile_kernel_with_bisheng(
        src_file=Path(src_file),
        output_dir=Path(output_dir),
        kernel_name=kernel_name,
        **kwargs
    )
    return str(result)


# ================================================================
# 模块信息
# ================================================================

__all__ = [
    # 版本
    "__version__",
    "__author__",

    # 工具
    "find_ascend_root",
    "find_runtime_library",
    "setup_environment",
    "logger",
    "Logger",
    "ProgressBar",

    # 异常
    "AscendRuntimeError",
    "CompilerError",
    "ExecutorError",
    "BinaryNotFoundError",
    "InvalidBinaryError",
    "MemoryAllocationError",
    "KernelLaunchError",

    # 编译器
    "BishengCompiler",
    "compile_kernel_with_bisheng",

    # Ascgen编译
    "AscGen",
    "save_source_files",
    "calc_tiling_data",

    # 执行器
    "DevBinary",
    "AscendRuntime",
    "KernelExecutor",
    "execute_kernel",

    # 便捷函数
    "read_binary",
    "compile_kernel",
]
