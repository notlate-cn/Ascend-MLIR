import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python" / "torch"))

# Ascend CANN 环境变量（只需修改 ASCEND_HOME_PATH）
ASCEND_HOME_PATH = os.environ.get("ASCEND_HOME_PATH", "/home/gser/Ascend/cann")
os.environ["ASCEND_HOME_PATH"] = ASCEND_HOME_PATH
os.environ["PATH"] = os.environ.get("PATH", "") + ":" + str(REPO_ROOT / "build" / "bin")
os.environ["LD_LIBRARY_PATH"] = (
    f"{ASCEND_HOME_PATH}/x86_64-linux/lib64"
    f":{ASCEND_HOME_PATH}/x86_64-linux/simulator/Ascend910B1/lib"
    ":" + os.environ.get("LD_LIBRARY_PATH", "")
)

# 每次测试前自动编译
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build.sh"
if BUILD_SCRIPT.exists():
    print("[conftest] 编译项目 (scripts/build.sh --build-project) ...")
    result = subprocess.run(
        ["bash", str(BUILD_SCRIPT), "--build-project"],
    )
    if result.returncode != 0:
        print("[conftest] 编译失败")
        sys.exit(1)