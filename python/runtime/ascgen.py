"""
AscendC 源代码生成接口

提供使用 Autofuse 生成源代码并编译执行 host tiling 的功能。

主要功能:
1. AscGen - 源代码生成器封装类
2. compile_and_execute_host_tiling - 编译并执行 host tiling C++ 代码
3. save_source_files - 保存源代码文件
"""
import ctypes
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from .utils import (
    get_platform,
    find_ascend_root,
    setup_environment,
    logger,
    CompilerError,
    ASCEND_A2,
)


# ================================================================
# 源代码生成器类
# ================================================================

class AscGen:
    """
    AscendC 源代码生成器封装类

    提供 Autofuse 源代码生成和 host tiling 编译执行功能。
    """

    def __init__(self,
                 ascend_root: Optional[Path] = None,
                 soc_version: str = ASCEND_A2):
        """
        初始化源代码生成器

        Args:
            ascend_root: Ascend 根目录
            soc_version: SOC 版本

        Raises:
            CompilerError: 初始化失败
        """
        # 设置环境
        if ascend_root is None:
            ascend_root = find_ascend_root()

        setup_environment(ascend_root, soc_version)

        self.ascend_root = ascend_root
        self.soc_version = soc_version
        self.platform = get_platform()

        # 加载 libgraph.so
        self._load_libgraph()

        # 导入 autofuse 模块
        self._import_autofuse()

    def _load_libgraph(self):
        """加载 libgraph.so"""
        libgraph_path = self.ascend_root / "lib64" / "libgraph.so"

        if not libgraph_path.exists():
            raise CompilerError(f"Cannot find libgraph.so: {libgraph_path}")

        try:
            ctypes.CDLL(str(libgraph_path), mode=ctypes.RTLD_GLOBAL)
            logger.info(f"Loaded libgraph.so: {libgraph_path}")
        except OSError as e:
            raise CompilerError(f"Failed to load libgraph.so: {e}")

    def _import_autofuse(self):
        """导入 autofuse 相关模块"""
        try:
            from autofuse.pyautofuse import ascir, Autofuser, AutofuserOptions
            self.ascir = ascir
            self.Autofuser = Autofuser
            self.AutofuserOptions = AutofuserOptions
            logger.info("Autofuse modules imported successfully")
        except ImportError as e:
            raise CompilerError(f"Failed to import autofuse modules: {e}")

    def compile_graph(self,
                      graph_text: str,
                      output_dir: Path,
                      graph_name: str = "kernel") -> Tuple[Path, Path, str, str]:
        """
        从 AscendC graph 生成源代码

        Args:
            graph_text: AscendC graph 文本定义
            output_dir: 输出目录
            graph_name: graph 名称

        Returns:
            (host 源文件路径, device 源文件路径, host_tiling 源代码, tiling_def 结构定义)

        Raises:
            CompilerError: 生成失败
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 反序列化 graph
        graph = self._deserialize_graph(graph_text)

        # 执行 autofuse
        tiling_def, host_tiling, op_kernel = self._run_autofuse(graph)

        # 保存源文件
        host_file, device_file = self._save_source_files(
            tiling_def, host_tiling, op_kernel,
            output_dir, graph_name
        )

        return host_file, device_file, host_tiling, tiling_def

    def _deserialize_graph(self, graph_text: str):
        """反序列化 ascgraph 文本"""
        graph = self.ascir.utils.deserialize("asc_graph", graph_text)
        debug_str = self.ascir.utils.debug_str(graph)

        if not debug_str:
            raise CompilerError("Failed to deserialize ascgraph")

        logger.debug(f"Graph deserialized: \n{debug_str}")
        return graph

    def _run_autofuse(self, graph):
        """执行 autofuse 流程"""
        options = self.AutofuserOptions()
        fuser = self.Autofuser(options)

        sched_result = fuser.schedule(graph)
        tiling_def, host_tiling, op_kernel = fuser.codegen(sched_result)

        if not all([tiling_def, host_tiling, op_kernel]):
            raise CompilerError("Autofuse codegen failed")

        logger.info("Autofuse pipeline completed")
        return tiling_def, host_tiling, op_kernel

    def _save_source_files(self, tiling_def, host_tiling, op_kernel,
                           output_dir: Path, graph_name: str) -> Tuple[Path, Path]:
        """
        保存源代码文件到输出目录

        Args:
            tiling_def: tiling 定义
            host_tiling: host tiling 源代码
            op_kernel: device kernel 源代码
            output_dir: 输出目录
            graph_name: graph 名称

        Returns:
            (host 源文件路径, device 源文件路径)
        """
        TILING_DATA_HEADER_NAME = 'autofuse_tiling_data.h'

        # 创建子目录
        host_dir = output_dir / "host"
        device_dir = output_dir / "device"
        host_dir.mkdir(parents=True, exist_ok=True)
        device_dir.mkdir(parents=True, exist_ok=True)

        # 保存 host tiling 源文件
        host_file = host_dir / f"{graph_name}_tiling.cpp"
        host_file.write_text(host_tiling, encoding="utf-8")
        tiling_def_file = host_dir / TILING_DATA_HEADER_NAME
        tiling_def_file.write_text(tiling_def, encoding="utf-8")
        logger.info(f"Saved host tiling: {host_file}")

        # 保存 device kernel 源文件
        device_file = device_dir / f"{graph_name}_op_kernel.cpp"
        device_file.write_text(op_kernel, encoding="utf-8")
        tiling_def_file = device_dir / TILING_DATA_HEADER_NAME
        tiling_def_file.write_text(tiling_def, encoding="utf-8")
        logger.info(f"Saved device kernel: {device_file}")

        return host_file, device_file

    def calc_tiling_data(
            self,
            tiling_def: str,
            host_tiling: str,
            build_dir: Optional[Path] = None,
    ) -> bytes:
        """
        编译并执行 host tiling C++ 代码，生成 tiling 数据

        Args:
            tiling_def: tiling 数据结构定义（C++ 代码）
            host_tiling: host tiling C++ 源代码
            build_dir: 编译输出目录，如果为 None 则使用临时目录（用于调试）

        Returns:
            tiling 数据 (bytes)
        """
        # 决定使用哪个目录
        if build_dir is None:
            # 使用临时目录（原有行为）
            work_dir = Path(tempfile.mkdtemp())
            cleanup = True
        else:
            # 使用用户指定的目录（便于调试）
            work_dir = Path(build_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            cleanup = False
            logger.info(f"Using build directory for debugging: {work_dir}")

        try:
            # 编译 host tiling 为 .so
            so_file = self._compile_host_tiling_to_so(
                tiling_def, host_tiling, work_dir
            )
            logger.info(f"Compiled host tiling to: {so_file}")
            if so_file is None:
                raise CompilerError("Failed to compile host tiling to .so")

            # 调用 .so 中的函数获取 tiling 数据
            tiling_data = self._call_tiling_functions(so_file)

            return tiling_data
        finally:
            # 清理临时目录
            if cleanup and work_dir.exists():
                import shutil
                shutil.rmtree(work_dir, ignore_errors=True)

    def _compile_host_tiling_to_so(
            self,
            tiling_def: str,
            host_tiling: str,
            output_dir: Path
    ) -> Optional[Path]:
        """
        编译 host tiling C++ 源代码为 .so 文件

        使用 CMake 构建系统，因为 host tiling 使用 ASC 语言，
        需要 Ascend 特定的编译器和库。

        Args:
            tiling_def: tiling 数据结构定义（C++ 代码）
            host_tiling: host tiling C++ 源代码
            output_dir: 输出目录

        Returns:
            编译后的 .so 文件路径，失败返回 None
        """
        output_dir = Path(output_dir)

        # 保存源文件
        source_file = output_dir / "host_tiling_combined.cpp"
        source_content = f"""// Auto-generated host tiling library
// Tiling data structure definition
{tiling_def}

// Host tiling implementation
{host_tiling}
"""
        source_file.write_text(source_content, encoding="utf-8")

        # 保存 tiling_def 头文件（可能在编译时需要）
        tiling_header = output_dir / "autofuse_tiling_data.h"
        tiling_header.write_text(tiling_def, encoding="utf-8")

        # 查找 ASC CMake 包路径
        asc_cmake_path = None
        possible_paths = [
            self.ascend_root / "tools" / "tikcpp" / "ascendc_kernel_cmake",
            self.ascend_root / "compiler" / "tikcpp" / "ascendc_kernel_cmake",
            self.ascend_root / "ascendc_devkit" / "tikcpp" / "samples" / "cmake",
        ]
        for path in possible_paths:
            if path.exists():
                asc_cmake_path = path
                break

        # 生成 CMakeLists.txt
        cmake_file = output_dir / "CMakeLists.txt"
        if asc_cmake_path is not None:
            logger.info(f"Using ASC CMake package from: {asc_cmake_path}")
            # 使用 ASC 语言支持的 CMake 配置
            cmake_content = f"""cmake_minimum_required(VERSION 3.16.0)
find_package(ASC REQUIRED HINTS {asc_cmake_path})
project(HostTiling LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOC_VERSION {self.soc_version} CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "{self.ascend_root}" CACHE PATH "ASCEND CANN package installation directory")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)

# 设置库搜索路径（必须在 add_library 之前）
link_directories(
    {self.ascend_root}/lib64
)

# 设置源文件属性为 ASC 语言
set_source_files_properties(
    {source_file.name}
    PROPERTIES LANGUAGE ASC
)

# 创建共享库
add_library(host_tiling SHARED
    {source_file.name}
)

# 设置输出名称（使用标准的 lib 前缀和 .so 后缀）
set_target_properties(host_tiling PROPERTIES
    OUTPUT_NAME host_tiling
)

# 链接必需的库
target_link_libraries(host_tiling PRIVATE
    c_sec
    ascendalog
    platform
    tiling_api
    graph
)

# 设置包含路径
target_include_directories(host_tiling PRIVATE
    {output_dir}
    {self.ascend_root}/include
    {self.ascend_root}/include/graph
    {self.ascend_root}/include/experiment
    {self.ascend_root}/pkg_inc/base
    {self.ascend_root}/{self.platform}-linux/include
    {self.ascend_root}/{self.platform}-linux/ascendc/include/highlevel_api/tiling/platform
)

# 设置编译选项
target_compile_options(host_tiling PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
    -O2
    -fno-common
    -Wextra
    -Wfloat-equal
    -fvisibility=default
    -DLOG_CPP
    -ffile-prefix-map=${{CMAKE_CURRENT_LIST_DIR}}/=
)
"""
        else:
            # Fallback: 不使用 ASC 语言，尝试使用标准 C++（可能失败）
            logger.warning(f"ASC CMake package not found, using fallback C++ mode (may not work)")
            cmake_content = f"""cmake_minimum_required(VERSION 3.14)
project(HostTiling CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ASCEND_ROOT "{self.ascend_root}")

# 设置库搜索路径
link_directories(${{ASCEND_ROOT}}/lib64)

# 创建共享库
add_library(host_tiling SHARED
    {source_file.name}
)

# 设置输出名称
set_target_properties(host_tiling PROPERTIES
    OUTPUT_NAME host_tiling
)

# 设置包含路径
target_include_directories(host_tiling PRIVATE
    {output_dir}
    ${{ASCEND_ROOT}}/include
    ${{ASCEND_ROOT}}/include/graph
    ${{ASCEND_ROOT}}/include/experiment
    ${{ASCEND_ROOT}}/pkg_inc/base
    ${{ASCEND_ROOT}}/{self.platform}-linux/include
    ${{ASCEND_ROOT}}/{self.platform}-linux/ascendc/include/highlevel_api/tiling/platform
)

# 链接必需的库
target_link_libraries(host_tiling PRIVATE
    c_sec
    ascendalog
    platform
    tiling_api
    graph
)

# 设置编译选项
target_compile_options(host_tiling PRIVATE
    -O2
    -fno-common
    -Wextra
    -fvisibility=default
    -DLOG_CPP
)
"""
        cmake_file.write_text(cmake_content, encoding="utf-8")

        # 创建构建目录
        build_dir = output_dir / "cmake_build"
        build_dir.mkdir(exist_ok=True)

        so_file = build_dir / "libhost_tiling.so"

        # 设置环境变量（参考 call_ascgen.py:setup_cmake_environment）
        env = os.environ.copy()
        env["ASCEND_HOME_PATH"] = str(self.ascend_root)
        env["SOC_VERSION"] = self.soc_version

        # CMake 需要的路径设置
        if asc_cmake_path is not None:
            env["CMAKE_PREFIX_PATH"] = str(asc_cmake_path)
            logger.info(f"Setting CMAKE_PREFIX_PATH: {asc_cmake_path}")

        env["ASCEND_CANN_PACKAGE_PATH"] = str(self.ascend_root)
        logger.info(f"Setting ASCEND_CANN_PACKAGE_PATH: {self.ascend_root}")

        # 设置 LD_LIBRARY_PATH
        ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{self.ascend_root}/lib64:{ld_path}"

        try:
            # 1. CMake 配置
            logger.info(f"Configuring CMake project...")
            configure_cmd = [
                "cmake",
                "-S", str(output_dir),
                "-B", str(build_dir),
                f"-DCMAKE_BUILD_TYPE=Release",
            ]

            result = subprocess.run(
                configure_cmd,
                capture_output=True,
                text=True,
                timeout=60,
                env=env
            )

            if result.returncode != 0:
                logger.error(f"CMake configuration failed: {result.stderr}")
                logger.error(f"stdout: {result.stdout}")
                return None

            logger.info(f"CMake configuration successful")

            # 2. CMake 构建
            logger.info(f"Building with CMake...")
            build_cmd = [
                "cmake",
                "--build", str(build_dir),
                "--parallel",
            ]

            result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=120,
                env=env
            )

            if result.returncode != 0:
                logger.error(f"CMake build failed: {result.stderr}")
                logger.error(f"stdout: {result.stdout}")
                return None

            # 检查输出文件是否存在
            if so_file.exists():
                logger.info(f"Build successful: {so_file}")
                return so_file
            else:
                # 查找可能的其他输出位置
                for possible_so in build_dir.glob("*.so"):
                    logger.info(f"Found .so file: {possible_so}")
                    return possible_so
                logger.error(f"No .so file found in {build_dir}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Build timeout")
            return None
        except FileNotFoundError as e:
            logger.error(f"CMake not found: {e}")
            logger.error("Please ensure CMake is installed and available in PATH")
            return None
        except Exception as e:
            logger.error(f"Build error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _call_tiling_functions(self, so_file: Path) -> bytes:
        """
        调用 .so 文件中的函数获取 tiling 数据

        Args:
            so_file: .so 文件路径

        Returns:
            tiling 数据 (bytes)
        """
        # 加载 .so 文件
        # 使用 RTLD_LOCAL 避免与 libruntime_camodel.so 的全局状态冲突
        try:
            so_lib = ctypes.CDLL(str(so_file), mode=ctypes.RTLD_LOCAL)
            logger.info(f"Loaded .so file: {so_file}")
        except OSError as e:
            raise CompilerError(f"Failed to load .so file: {e}")

        # 获取 GetTilingDataSize 函数
        try:
            get_tiling_size_func = so_lib.GetTilingDataSize
            get_tiling_size_func.restype = ctypes.c_size_t
            get_tiling_size_func.argtypes = []
        except AttributeError as e:
            raise CompilerError(f"GetTilingDataSize function not found in .so: {e}")

        # 获取 tiling 数据大小
        tiling_size = get_tiling_size_func()
        logger.info(f"Tiling data size: {tiling_size} bytes")

        # 分配 tiling 数据缓冲区
        tiling_buffer = ctypes.create_string_buffer(tiling_size)

        # 获取 AutofuseTiling 函数
        try:
            autofuse_tiling_func = so_lib.AutofuseTiling
            autofuse_tiling_func.restype = ctypes.c_int
            autofuse_tiling_func.argtypes = [
                ctypes.c_void_p,  # tiling_data
                ctypes.POINTER(ctypes.c_uint32),  # workspace_size
                ctypes.POINTER(ctypes.c_uint32),  # block_dim
                ctypes.c_void_p,  # unknown (can be nullptr)
            ]
        except AttributeError as e:
            raise CompilerError(f"AutofuseTiling function not found in .so: {e}")

        # 调用 AutofuseTiling 计算 tiling 数据
        workspace_size = ctypes.c_uint32(0)
        block_dim = ctypes.c_uint32(0)

        ret = autofuse_tiling_func(
            ctypes.addressof(tiling_buffer),  # tiling_data
            ctypes.byref(workspace_size),  # workspace_size
            ctypes.byref(block_dim),  # block_dim
            ctypes.c_void_p(0)  # unknown (nullptr)
        )

        if ret != 0:
            raise CompilerError(f"AutofuseTiling failed with return code: {ret}")

        logger.info(
            f"Tiling calculation successful: block_dim={block_dim.value}, workspace_size={workspace_size.value}")

        # 返回 tiling 数据
        tiling_data = tiling_buffer.raw[:tiling_size]

        # 解析并打印 tiling 数据
        self._parse_and_print_tiling_data(tiling_data)

        return tiling_data

    def _parse_and_print_tiling_data(self, tiling_data: bytes):
        """
        解析并打印 tiling 数据

        根据 AutofuseTilingData 结构解析二进制数据：
        struct AutofuseTilingData {
            uint32_t block_dim;
            uint32_t corenum;
            uint32_t ub_size;
            uint32_t hbm_size;
            uint32_t tiling_key;
            uint32_t z1t_size;
            uint32_t z0z1Tb_size;
            uint32_t z0t_size;
            uint32_t z0Tb_size;
            uint32_t q0_size;
            uint32_t q1_size;
            uint32_t q2_size;
            uint32_t b0_size;
            uint32_t b1_size;
            uint32_t tmp_tbuf_size;
        };

        Args:
            tiling_data: tiling 数据二进制
        """
        # AutofuseTilingData 包含 15 个 uint32_t 字段
        expected_size = 15 * 4  # 60 bytes
        if len(tiling_data) < expected_size:
            logger.warning(
                f"Tiling data size ({len(tiling_data)} bytes) is smaller than expected ({expected_size} bytes)")
            # 只解析可用的数据
            num_fields = len(tiling_data) // 4
        else:
            num_fields = 15

        # 使用 struct 模块解析 uint32_t 数组
        # 格式: 15 个 'I' (unsigned int, 4 bytes)
        fmt = f'{num_fields}I'
        fields = struct.unpack(fmt, tiling_data[:num_fields * 4])

        # 字段名称映射
        field_names = [
            "block_dim",
            "corenum",
            "ub_size",
            "hbm_size",
            "tiling_key",
            "z1t_size",
            "z0z1Tb_size",
            "z0t_size",
            "z0Tb_size",
            "q0_size",
            "q1_size",
            "q2_size",
            "b0_size",
            "b1_size",
            "tmp_tbuf_size",
        ]

        logger.info("\n" + "=" * 70)
        logger.info("Tiling Data (AutofuseTilingData):")
        logger.info("=" * 70)

        # 打印每个字段
        for i, (name, value) in enumerate(zip(field_names, fields)):
            logger.info(f"  [{i:2d}] {name:20s} = {value:10d} (0x{value:08x})")

        logger.info("=" * 70)
        logger.info(f"Total size: {len(tiling_data)} bytes")
        logger.info("=" * 70 + "\n")


# ================================================================
# 便捷函数
# ================================================================

def save_source_files(graph_text: str,
                      output_dir: Path,
                      graph_name: str = "kernel",
                      ascend_root: Optional[Path] = None,
                      soc_version: str = ASCEND_A2) -> Tuple[Path, Path, str, str]:
    """
    从 AscendC graph 生成源代码文件（便捷函数）

    Args:
        graph_text: AscendC graph 文本定义
        output_dir: 输出目录
        graph_name: graph 名称
        ascend_root: Ascend 根目录
        soc_version: SOC 版本

    Returns:
        (host 源文件路径, device 源文件路径, host_tiling 源代码, tiling_def 结构定义)

    Example:
        >>> host_file, device_file, host_tiling, tiling_def = save_source_files(
        ...     graph_text=graph_def,
        ...     output_dir="./build",
        ...     graph_name="my_kernel"
        ... )
    """
    compiler = AscGen(ascend_root, soc_version)
    return compiler.compile_graph(graph_text, output_dir, graph_name)


def calc_tiling_data(
        tiling_def: str,
        host_tiling: str,
        ascend_root: Optional[Path] = None,
        soc_version: str = ASCEND_A2,
        build_dir: Optional[Path] = None,
) -> bytes:
    """
    编译并执行 host tiling C++ 代码（便捷函数）

    Args:
        tiling_def: tiling 数据结构定义（C++ 代码）
        host_tiling: host tiling C++ 源代码
        ascend_root: Ascend 根目录
        soc_version: SOC 版本
        build_dir: 编译输出目录，如果为 None 则使用临时目录（用于调试）

    Returns:
        tiling 数据 (bytes)

    Example:
        >>> tiling_data = calc_tiling_data(
        ...     tiling_def=tiling_def_str,
        ...     host_tiling=host_tiling_str
        ... )
        >>>
        >>> # 指定构建目录以便调试
        >>> tiling_data = calc_tiling_data(
        ...     tiling_def=tiling_def_str,
        ...     host_tiling=host_tiling_str,
        ...     build_dir="./debug_build"
        ... )
    """
    compiler = AscGen(ascend_root, soc_version)
    return compiler.calc_tiling_data(tiling_def, host_tiling, build_dir)
