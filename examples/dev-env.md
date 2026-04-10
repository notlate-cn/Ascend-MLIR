# 开发环境

本地环境是MacOS，安装了OrbStack Docker虚拟机，名字为：xvm。

本地写代码，会自动同步到xvm容器的/home/niu/code/Ascend-MLIR目录下，存在极短时间的延迟。

所以：请在本地修改代码后，等待1秒钟，再在xvm容器中编译调试。

xvm容器SSH连接方式：ssh xvm@orb

项目编译方式：

```shell
ssh xvm@orb
cd /home/niu/code/Ascend-MLIR
./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/build
source examples/env.sh
afir-opt -h
```