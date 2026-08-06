# baremix-test

`baremix-test` 是 `RuntimeMix` 的最小验证用例，目标是跑通：

`baremix_custom.cpp -> mix-compiler -> mix_runner -> mix-validator -> verify_result.py`

## 运行环境

只在 `xvm` 中运行。不要在 host 上执行。

默认依赖：

- `LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build`
- `SOC_VERSION=Ascend910B1`

## 一键验证

在 `xvm` 中执行：

```bash
cd /home/niu/code/Ascend-MLIR
bash examples/baremix-test/run.sh
```

或显式指定环境变量：

```bash
cd /home/niu/code/Ascend-MLIR
LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build \
SOC_VERSION=Ascend910B1 \
bash examples/baremix-test/run.sh
```

## Current ABI

当前 `baremix-test` 只稳定支持这一个接受面，不把它推广成通用 ABI。当前 runner 和 direct packed fallback 都从 `out/manifest.txt` 读取同一份 ABI metadata：

- `GM inputs`: 3 个，分别是 `x1_gm.bin`、`x2_gm.bin`、`bias.bin`
- `GM outputs`: 1 个，写到 `output.bin`
- `workspace`: 当前路径使用固定大小 workspace，不从输入推导动态大小
- `tiling`: 当前路径使用 artifact 中真实生成的 `out/tiling.bin`；runner 和 direct packed 读取同一份 tiling 来源
- `baremix` acceptance case: 只接受 3 inputs + 1 output 的这组约定

## 产物

- 编译产物：`build/runtime-mix-baremix`
- 测试数据：`build/runtime-mix-baremix/testdata`
- host runner：`build/runtime-mix-baremix/out/bin/mix_runner`
- packed kernel：`build/runtime-mix-baremix/out/libbaremix_custom_packed.so`

## 成功标准

- `mix-validator` 输出 `PASS`
- `scripts/verify_result.py` 输出 `test pass`

## Direct Packed Debug

如果要显式验证 `Executor::RunPackedMixFile(...)` 这条 direct packed 路径，可以在 `xvm` 中执行：

```bash
cd /home/niu/code/Ascend-MLIR
./build/runtime-mix-bootstrap/bin/mix-validator \
  --artifact-root build/runtime-mix-baremix \
  --input-dir build/runtime-mix-baremix/testdata/input \
  --golden build/runtime-mix-baremix/testdata/output/golden.bin \
  --output-file build/runtime-mix-baremix/testdata/output/direct-actual.bin \
  --soc Ascend910B1 \
  --force-direct-packed
```

这个入口会复用 artifact 中的 `out/tiling.bin`，确保 direct packed 和 runner 使用同源 tiling。
