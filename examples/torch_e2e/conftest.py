import os
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python" / "torch"))

# Ascend CANN 环境变量（通过 ASCEND_HOME_PATH 或 ASCEND_TOOLKIT_HOME 指定）
ASCEND_HOME_PATH = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
if not ASCEND_HOME_PATH:
    raise EnvironmentError("请设置 ASCEND_HOME_PATH 或 ASCEND_TOOLKIT_HOME 环境变量")
os.environ["ASCEND_HOME_PATH"] = ASCEND_HOME_PATH
os.environ["PATH"] = os.environ.get("PATH", "") + ":" + str(REPO_ROOT / "build" / "bin")

_machine = platform.machine()
if _machine in ("x86_64", "amd64"):
    _arch_dir = "x86_64-linux"
    _devlib_arch = "x86_64"
elif _machine in ("aarch64", "arm64"):
    _arch_dir = "aarch64-linux"
    _devlib_arch = "aarch64"
else:
    raise EnvironmentError(f"不支持的架构: {_machine}")

os.environ["LD_LIBRARY_PATH"] = (
    f"{ASCEND_HOME_PATH}/{_arch_dir}/lib64"
    f":{ASCEND_HOME_PATH}/{_arch_dir}/simulator/Ascend910B1/lib"
    f":{ASCEND_HOME_PATH}/{_arch_dir}/devlib/linux/{_devlib_arch}"
    ":" + os.environ.get("LD_LIBRARY_PATH", "")
)
