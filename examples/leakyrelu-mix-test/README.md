# leakyrelu-mix-test

`leakyrelu-mix-test` 是 `RuntimeMix` 的第二个 mix 验证样例，用来验证当前 artifact-backed tiling 和 direct-packed 校验机制不只适用于 baremix。

内核来源：

- `examples/matmul-add-relu-sum/fc_leakyrelu_mix.cpp`

运行方式只在 `xvm` 中支持：

```bash
cd /home/niu/code/Ascend-MLIR
bash examples/leakyrelu-mix-test/run.sh
```

这条入口会完成：

- 自举或复用 `mix-compiler` 与 `mix-validator`
- 通过 `RuntimeMix` direct backend 编译 `fc_leakyrelu_mix.cpp`
- 生成 `out/tiling.bin`
- 调用 `gen_data_leakyrelu.py` 生成 `.npy`
- 把 `.npy` 转成当前 ABI metadata 约定的 `.bin`
- 先跑默认验证路径，再跑 `--force-direct-packed`

当前样例的源码文件名和实际 runtime kernel 名不完全一致。运行脚本会固定使用：

- source: `fc_leakyrelu_mix.cpp`
- runtime kernel name: `fc_leakyrelu`

输出目录在：

- artifact: `build/runtime-mix-leakyrelu`
- data: `build/runtime-mix-leakyrelu/testdata`

关键输出文件包括：

- `out/libfc_leakyrelu_packed.so`
- `out/bin/mix_runner`
- `out/tiling.bin`
- `testdata/output/golden.bin`
- `testdata/output/actual.bin`
- `testdata/output/direct-actual.bin`
