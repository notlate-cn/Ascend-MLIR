"""
Ascend Runtime 工具函数

提供路径查找、环境设置等通用功能。
"""
import os
import sys
from pathlib import Path
from typing import Optional, List


# ================================================================
# 路径工具
# ================================================================

def find_ascend_root() -> Path:
    """
    查找 Ascend 安装根目录: ASCEND_HOME_PATH 环境变量

    Returns:
        Ascend 根目录路径，如果未找到返回 None
    """

    # 检查 ASCEND_HOME_PATH 环境变量
    if 'ASCEND_HOME_PATH' not in os.environ:
        print("❌ 错误: 环境变量 ASCEND_HOME_PATH 未设置")
        print("请先执行: source python/env.sh")
        sys.exit(1)

    ascend_home_path = os.environ['ASCEND_HOME_PATH']
    print(f"✅ ASCEND_HOME_PATH: {ascend_home_path}")

    # 检查安装路径是否存在
    if not os.path.exists(ascend_home_path):
        print(f"❌ 错误: Ascend 安装路径不存在: {ascend_home_path}")
        sys.exit(1)
    print(f"✅ 安装路径存在")

    return Path(ascend_home_path)


def find_runtime_library(ascend_root: Optional[Path] = None,
                         soc_version: str = "Ascend910B1",
                         simulation_mode: bool = True) -> Optional[Path]:
    """
    查找 runtime 库路径

    Args:
        ascend_root: Ascend 根目录，如果为 None 则自动查找
        soc_version: SOC 版本 (默认 Ascend910B1)
        simulation_mode: 是否为仿真模式

    Returns:
        runtime 库路径，如果未找到返回 None
    """
    ascend_root = find_ascend_root()

    # 仿真模式优先使用 tools/simulator 下的库（与 pyasc 一致）
    if simulation_mode:
        candidates = [
            f"tools/simulator/{soc_version}/lib/libruntime_camodel.so",
            f"aarch64-linux/simulator/{soc_version}/lib/libruntime_camodel.so",
            f"aarch64-linux/simulator/{soc_version}/lib/libruntime_cmodel.so",
        ]
    else:
        candidates = [
            f"aarch64-linux/lib64/libruntime.so",
        ]

    for rel_path in candidates:
        lib_path = ascend_root / rel_path
        if lib_path.exists():
            return lib_path

    return None


def find_bisheng_compiler(ascend_root: Optional[Path] = None) -> Optional[Path]:
    """
    查找 bisheng 编译器路径

    Args:
        ascend_root: Ascend 根目录，如果为 None 则自动查找

    Returns:
        bisheng 编译器路径，如果未找到返回 None
    """
    ascend_root = find_ascend_root()

    # 可能的路径
    candidates = [
        "toolchain/bisheng_compiler/bisheng",
        "compiler/bisheng",
        "bin/bisheng",
    ]

    for rel_path in candidates:
        compiler_path = ascend_root / rel_path
        if compiler_path.exists():
            return compiler_path

    return None


# ================================================================
# 环境设置
# ================================================================

def setup_environment(ascend_root: Optional[Path] = None,
                      soc_version: str = "Ascend910B1",
                      simulation_mode: bool = True) -> bool:
    """
    设置 Ascend 运行环境

    Args:
        ascend_root: Ascend 根目录，如果为 None 则自动查找
        soc_version: SOC 版本
        simulation_mode: 是否为仿真模式

    Returns:
        是否成功设置环境
    """
    if ascend_root is None:
        ascend_root = find_ascend_root()

    # 设置环境变量
    os.environ['ASCEND_HOME_PATH'] = str(ascend_root)
    os.environ['SOC_VERSION'] = soc_version

    # 仿真模式相关环境变量
    if simulation_mode:
        os.environ['ASCEND_CPU_SIMULATION'] = '1'
        os.environ['ASCEND_DEVICE_ID'] = '0'

    # 更新 LD_LIBRARY_PATH
    lib_paths = [
        ascend_root / 'lib64',
        ascend_root / 'aarch64-linux' / 'lib64',
        ascend_root / 'aarch64-linux' / 'devlib' / 'linux' / 'aarch64',
    ]

    # 添加 stub runtime 路径（用于无硬件环境）
    stub_path = ascend_root / 'runtime/lib64/stub/linux/aarch64'
    if stub_path.exists():
        lib_paths.append(stub_path)

    if soc_version:
        lib_paths.extend([
            ascend_root / f'aarch64-linux/simulator/{soc_version}/lib',
        ])

    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld = ':'.join([str(p) for p in lib_paths if p.exists()])
    os.environ['LD_LIBRARY_PATH'] = f"{new_ld}:{current_ld}"

    return True


# ================================================================
# ELF 工具
# ================================================================

def read_elf_binary(file_path: Path) -> bytes:
    """
    读取 ELF 二进制文件

    Args:
        file_path: 文件路径

    Returns:
        ELF 文件内容
    """
    with open(file_path, "rb") as f:
        return f.read()


def get_function_symbols(binary_data: bytes) -> List[str]:
    """
    从 ELF 二进制中提取函数符号

    Args:
        binary_data: ELF 二进制数据

    Returns:
        函数符号列表
    """
    # 简化实现：查找可打印字符串
    # 实际应用中应该使用 elftools 库解析 ELF
    import re

    # 查找以 null 结尾的字符串
    strings = re.findall(b'[\x20-\x7e]{4,}\x00', binary_data)
    return [s.decode('ascii', errors='ignore').rstrip('\x00') for s in strings]


# ================================================================
# 日志工具
# ================================================================

class Logger:
    """简单的日志工具"""

    LEVELS = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3}

    def __init__(self, level: str = 'INFO', prefix: str = '[Ascend Runtime]'):
        """
        初始化日志器

        Args:
            level: 日志级别
            prefix: 日志前缀
        """
        self.level = self.LEVELS.get(level, 1)
        self.prefix = prefix

    def _log(self, level: str, message: str):
        """内部日志函数"""
        if self.LEVELS.get(level, 1) >= self.level:
            print(f"{self.prefix} [{level}] {message}")

    def debug(self, message: str):
        """调试日志"""
        self._log('DEBUG', message)

    def info(self, message: str):
        """信息日志"""
        self._log('INFO', message)

    def warning(self, message: str):
        """警告日志"""
        self._log('WARNING', message)

    def error(self, message: str):
        """错误日志"""
        self._log('ERROR', message)


# 默认日志器实例
logger = Logger()


# ================================================================
# 验证工具
# ================================================================

def validate_binary(binary_data: bytes,
                    expected_magic: bytes = b'\\x7fELF') -> bool:
    """
    验证 ELF 二进制文件

    Args:
        binary_data: 二进制数据
        expected_magic: 期望的魔数 (默认 ELF)

    Returns:
        是否为有效的 ELF 文件
    """
    return binary_data[:4] == expected_magic


def validate_input_shape(shape: tuple) -> bool:
    """
    验证输入形状

    Args:
        shape: 输入形状

    Returns:
        是否有效
    """
    if not isinstance(shape, (tuple, list)):
        return False
    if len(shape) < 2:
        return False
    return all(isinstance(d, int) and d > 0 for d in shape)


# ================================================================
# 进度显示
# ================================================================

class ProgressBar:
    """简单的进度条"""

    def __init__(self, total: int, width: int = 50):
        """
        初始化进度条

        Args:
            total: 总数
            width: 进度条宽度
        """
        self.total = total
        self.width = width
        self.current = 0

    def update(self, n: int = 1):
        """
        更新进度

        Args:
            n: 增加的数量
        """
        self.current += n
        self._draw()

    def _draw(self):
        """绘制进度条"""
        if self.total == 0:
            return

        ratio = self.current / self.total
        filled = int(self.width * ratio)
        bar = '█' * filled + '░' * (self.width - filled)
        percent = int(ratio * 100)

        # 使用 \\r 让进度条在同一行更新
        sys.stdout.write(f'\\r[{bar}] {percent}%')
        sys.stdout.flush()

    def finish(self):
        """完成进度条"""
        self.current = self.total
        self._draw()
        print()  # 换行


# ================================================================
# 异常类
# ================================================================

class AscendRuntimeError(Exception):
    """Ascend Runtime 基础异常"""
    pass


class CompilerError(AscendRuntimeError):
    """编译器异常"""
    pass


class ExecutorError(AscendRuntimeError):
    """执行器异常"""
    pass


class BinaryNotFoundError(CompilerError):
    """二进制文件未找到异常"""
    pass


class InvalidBinaryError(CompilerError):
    """无效的二进制文件异常"""
    pass


class MemoryAllocationError(ExecutorError):
    """内存分配异常"""
    pass


class KernelLaunchError(ExecutorError):
    """Kernel 启动异常"""
    pass
