# 开发环境

本地环境是MacOS，安装了OrbStack Docker虚拟机，名字为：xvm。

当前可用的 xvm 工作目录是 `/Users/niu/Code/Codex-Ascend-MLIR`，不是旧的 `/home/niu/code/Ascend-MLIR`。

如果本地修改代码后需要在 xvm 中验证，等待 1 秒左右再编译调试，避免同步延迟。

xvm容器SSH连接方式：ssh xvm@orb

推荐的 xvm 构建与验证流程：

```shell
ssh xvm@orb
cd /Users/niu/Code/Codex-Ascend-MLIR
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
./scripts/build.sh --build-project --llvm-build-dir "$LLVM_BUILD_DIR"
source examples/env.sh
afir-opt -h
```

task-graph runtime 的 focused verification 流程：

```shell
ssh xvm@orb
cd /Users/niu/Code/Codex-Ascend-MLIR
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
build/bin/runtime-session --help
```

说明：

- `test/tools/runtime/run_runtime.sh` 会执行 task-graph runtime 的 focused xvm 验证流程：构建 `AscendCRuntime` 和 `runtime-session`，然后做 `runtime-session --help`、planning/negative-path、`test_taskgraph_runtime` / `test_runtime` 检查。
- 如果 `build/CMakeCache.txt` 仍然指向旧的 `/home/niu/code/Ascend-MLIR`，脚本会自动删除并重新配置 `build/` 后再执行 focused runtime verification。
