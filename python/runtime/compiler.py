"""
Bisheng 编译器接口

提供使用 bisheng 编译器编译 AscendC kernel 的功能。

主要功能:
1. BishengCompiler - Bisheng 编译器封装类
2. compile_kernel - 便捷的编译函数
"""
import os
import subprocess
from pathlib import Path
from typing import Optional, List

from .utils import (
    find_ascend_root,
    find_bisheng_compiler,
    setup_environment,
    logger,
    CompilerError,
    BinaryNotFoundError,
)


# ================================================================
# Bisheng 编译器类
# ================================================================

class BishengCompiler:
    """
    Bisheng 编译器封装类

    提供 AscendC kernel 的编译和链接功能。
    """

    # 目标架构
    ARCH_VEC = "dav-c220-vec"  # VectorCore
    ARCH_CUBE = "dav-c220-cube"  # AiCore

    def __init__(self,
                 ascend_root: Optional[Path] = None,
                 soc_version: str = "Ascend910B1"):
        """
        初始化编译器

        Args:
            ascend_root: Ascend 根目录，如果为 None 则自动查找
            soc_version: SOC 版本

        Raises:
            CompilerError: 如果找不到编译器
        """
        # 设置环境
        if ascend_root is None:
            ascend_root = find_ascend_root()

        setup_environment(ascend_root, soc_version)

        self.ascend_root = ascend_root
        self.soc_version = soc_version

        # 查找编译器和链接器
        self.bisheng_path = self._find_compiler()
        self.linker_path = self._find_linker()

        # TikCpp 包路径
        self.tikcpp_path = ascend_root / "compiler" / "tikcpp"

        logger.info(f"Bisheng Compiler: {self.bisheng_path}")
        logger.info(f"Linker: {self.linker_path}")

    def _find_compiler(self) -> Path:
        """查找 bisheng 编译器"""
        candidates = [
            self.ascend_root / "compiler" / "ccec_compiler" / "bin" / "bisheng",
            self.ascend_root / "toolchain" / "bisheng_compiler" / "bisheng",
        ]

        for path in candidates:
            if path.exists():
                return path

        raise CompilerError(f"Cannot find bisheng compiler in {self.ascend_root}")

    def _find_linker(self) -> Path:
        """查找 ld.lld 链接器"""
        candidates = [
            self.ascend_root / "compiler" / "ccec_compiler" / "bin" / "ld.lld",
            self.ascend_root / "toolchain" / "bisheng_compiler" / "ld.lld",
        ]

        for path in candidates:
            if path.exists():
                return path

        raise CompilerError(f"Cannot find ld.lld linker in {self.ascend_root}")

    def _get_common_options(self) -> List[str]:
        """
        获取通用编译选项

        参考 PyAsc compiler.py 的编译选项配置
        """
        return [
            "-std=c++17",
            "--cce-disable-kernel-global-attr-check",
            "-mllvm", "-cce-aicore-stack-size=0x8000",
            "-mllvm", "-cce-aicore-function-stack-size=0x8000",
            "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
            "-I", str(self.tikcpp_path / "tikcfw"),
            "-I", str(self.tikcpp_path / "tikcfw" / "impl"),
            "-I", str(self.tikcpp_path / "tikcfw" / "interface"),
            "-DASCENDC_DUMP=0",
            "-D__NPU_TILING__",
            "-DTILING_KEY_VAR=0",
        ]

    def compile(self,
                src_file: Path,
                output_file: Path,
                arch: str = ARCH_VEC,
                opt_level: int = 3,
                verbose: bool = False) -> Path:
        """
        编译 AscendC 源文件

        Args:
            src_file: 源文件路径
            output_file: 输出 .o 文件路径
            arch: 目标架构 (ARCH_VEC 或 ARCH_CUBE)
            opt_level: 优化级别 (0-3)
            verbose: 是否显示详细输出

        Returns:
            输出文件路径

        Raises:
            CompilerError: 编译失败
        """
        src_file = Path(src_file)
        output_file = Path(output_file)

        if not src_file.exists():
            raise CompilerError(f"Source file not found: {src_file}")

        # 创建输出目录
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 构建编译命令
        opt = f"-O{opt_level}"
        compile_cmd = [
                          str(self.bisheng_path),
                          "-c",
                          "-x", "cce",
                          opt,
                          src_file.name,  # 使用文件名，因为 cwd 将被设置为 src_file.parent
                          f"--cce-aicore-arch={arch}",
                          "--cce-aicore-only",
                          "-o", str(output_file.resolve()),  # 使用绝对路径
                      ] + self._get_common_options()

        logger.info(f"Compiling: {src_file.name} -> {output_file.name}")

        if verbose:
            logger.debug(f"Command: {' '.join(compile_cmd)}")

        # 执行编译
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            cwd=str(src_file.parent)
        )

        if result.returncode != 0:
            error_msg = f"Bisheng compilation failed (code {result.returncode})"
            if verbose:
                error_msg += f"\nstdout: {result.stdout}\nstderr: {result.stderr}"
            else:
                error_msg += f"\nstderr: {result.stderr[:500]}"
            raise CompilerError(error_msg)

        logger.info(f"Compilation successful: {output_file}")

        if verbose and result.stdout:
            logger.debug(f"Compiler output:\n{result.stdout}")

        return output_file

    def link(self,
             obj_file: Path,
             output_file: Path,
             verbose: bool = False) -> Path:
        """
        链接生成 ELF binary

        Args:
            obj_file: 输入 .o 文件路径
            output_file: 输出 ELF binary 路径
            verbose: 是否显示详细输出

        Returns:
            输出文件路径

        Raises:
            CompilerError: 链接失败
        """
        obj_file = Path(obj_file)
        output_file = Path(output_file)

        if not obj_file.exists():
            raise CompilerError(f"Object file not found: {obj_file}")

        # 创建输出目录
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 构建链接命令
        link_cmd = [
            str(self.linker_path),
            "-m", "aicorelinux",
            "-Ttext=0",
            str(obj_file),
            "-static",
            "-o", str(output_file),
        ]

        logger.info(f"Linking: {obj_file.name} -> {output_file.name}")

        if verbose:
            logger.debug(f"Command: {' '.join(link_cmd)}")

        # 执行链接
        result = subprocess.run(
            link_cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            error_msg = f"Linking failed (code {result.returncode})"
            if verbose:
                error_msg += f"\nstdout: {result.stdout}\nstderr: {result.stderr}"
            else:
                error_msg += f"\nstderr: {result.stderr[:500]}"
            raise CompilerError(error_msg)

        logger.info(f"Linking successful: {output_file}")

        return output_file

    def compile_and_link(self,
                         src_file: Path,
                         output_dir: Path,
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
            verbose: 是否显示详细输出

        Returns:
            ELF binary 文件路径
        """
        src_file = Path(src_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 中间文件和最终输出
        obj_file = output_dir / f"{kernel_name}.o"
        bin_file = output_dir / f"{kernel_name}.bin"

        # 编译
        self.compile(src_file, obj_file, arch, opt_level, verbose)

        # 链接
        self.link(obj_file, bin_file, verbose)

        # 验证输出
        if not bin_file.exists():
            raise BinaryNotFoundError(f"Binary file not generated: {bin_file}")

        binary_size = bin_file.stat().st_size
        logger.info(f"Generated binary: {bin_file} ({binary_size} bytes)")

        return bin_file


# ================================================================
# 便捷函数
# ================================================================

def compile_kernel(src_file: Path,
                   output_dir: Path,
                   kernel_name: str,
                   ascend_root: Optional[Path] = None,
                   soc_version: str = "Ascend910B1",
                   arch: str = BishengCompiler.ARCH_VEC,
                   opt_level: int = 3,
                   verbose: bool = False) -> Path:
    """
    编译 AscendC kernel 为 ELF binary

    Args:
        src_file: 源文件路径
        output_dir: 输出目录
        kernel_name: kernel 名称
        ascend_root: Ascend 根目录
        soc_version: SOC 版本
        arch: 目标架构
        opt_level: 优化级别
        verbose: 是否显示详细输出

    Returns:
        ELF binary 文件路径

    Example:
        >>> binary_path = compile_kernel(
        ...     src_file="device/kernel.cpp",
        ...     output_dir="./build",
        ...     kernel_name="my_kernel"
        ... )
    """
    compiler = BishengCompiler(ascend_root, soc_version)
    return compiler.compile_and_link(
        src_file=src_file,
        output_dir=output_dir,
        kernel_name=kernel_name,
        arch=arch,
        opt_level=opt_level,
        verbose=verbose
    )
