# -*- Python -*-

import os
import platform
import re
import subprocess
import sys
import tempfile

import lit.formats
import lit.util
import lit.llvm

from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.
if lit.llvm.llvm_config is None:
    lit.llvm.initialize(lit_config, config)

if not hasattr(config, 'afir_src_root'):
    config.afir_src_root = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
if not hasattr(config, 'afir_obj_root'):
    config.afir_obj_root = os.path.realpath(
        os.path.join(config.afir_src_root, 'build'))
if not hasattr(config, 'afir_tools_dir'):
    config.afir_tools_dir = os.path.join(config.afir_obj_root, 'bin')
if not hasattr(config, 'llvm_tools_dir'):
    config.llvm_tools_dir = os.path.join(
        os.environ.get('LLVM_BUILD_DIR', ''), 'bin') if os.environ.get('LLVM_BUILD_DIR') else ''
if not hasattr(config, 'llvm_shlib_ext'):
    config.llvm_shlib_ext = '.so'
if not getattr(config, 'python_executable', ''):
    config.python_executable = sys.executable
if not hasattr(config, 'enable_bindings_python'):
    config.enable_bindings_python = ''

# name: The name of this test suite.
config.name = 'AFIR'

config.test_format = lit.formats.ShTest(not lit.llvm.llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.afir_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

lit.llvm.llvm_config.with_system_environment([
    'HOME',
    'INCLUDE',
    'LIB',
    'TMP',
    'TEMP',
    'ASCEND_HOME_PATH',
    'ASCEND_TOOLKIT_HOME',
    'SOC_VERSION',
    'ASCEND_CPU_SIMULATION',
    'ASCEND_DEVICE_ID',
    'LD_LIBRARY_PATH',
    'PATH',
])

if (os.environ.get('ASCEND_HOME_PATH') or
        os.environ.get('ASCEND_TOOLKIT_HOME')):
    config.available_features.add('ascend_env')

lit.llvm.llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite.
config.excludes = [
    'CMakeLists.txt',
    'README.txt',
    'LICENSE.txt',
    'lit.cfg.py',
    'lit.site.cfg.py',
    'pypto_mix_public_breakdown.py',
    'cann-translate-mix-input.mlir',
    'cann-translate-mix-no-bias-input.mlir',
    'cann-translate-mix-relu-input.mlir',
    'cann-translate-mix-island-input.mlir',
    'cann-translate-mix-unsupported-vector-input.mlir',
    'cann-translate-mix-unsupported-cube-input.mlir',
    'cann-translate-gather-input.mlir',
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.afir_obj_root, 'test')

# Tweak the PATH to include the tools dir.
lit.llvm.llvm_config.with_environment('PATH', config.afir_tools_dir, append_path=True)
lit.llvm.llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

if config.enable_bindings_python:
    python_packages_dir = os.path.join(config.afir_obj_root, "python_packages")
    if os.path.exists(python_packages_dir):
        lit.llvm.llvm_config.with_environment(
            "PYTHONPATH",
            [python_packages_dir],
            append_path=True
        )
        config.suffixes.append('.py')
    else:
        config.enable_bindings_python = "false"

if not config.enable_bindings_python or config.enable_bindings_python == "false":
    config.excludes.append('python')

tool_dirs = [config.afir_tools_dir, config.llvm_tools_dir]
tools = [
    'aclnn-backend',
    'afir-opt',
    'afir-translate',
    ToolSubst('%PYTHON', config.python_executable, unresolved='ignore'),
]

lit.llvm.llvm_config.add_tool_substitutions(tools, tool_dirs)
