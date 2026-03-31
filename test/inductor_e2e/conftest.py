import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

# inductor_backend 包
sys.path.insert(0, str(REPO_ROOT / "python"))

# framework.pipeline (避免与 PyTorch 的 torch 包冲突，直接插入 python/torch/)
sys.path.insert(0, str(REPO_ROOT / "python" / "torch"))

# 构建产物（afir-opt, afir-translate 等）
# worktree 的 REPO_ROOT 指向 .worktrees/..., build 产物在主 repo 的 build/bin/
# 同时检查 worktree 本地和主 repo 两个位置
_main_repo = REPO_ROOT
while not (_main_repo / "build" / "bin").exists() and _main_repo != _main_repo.parent:
    _main_repo = _main_repo.parent
os.environ["PATH"] = (os.environ.get("PATH", "")
                      + ":" + str(_main_repo / "build" / "bin")
                      + ":" + str(REPO_ROOT / "build" / "bin"))