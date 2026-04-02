# relu-split-mix-test

`relu-split-mix-test` 是 `RuntimeMix` 的第二个 mix 验证样例，目标是验证当前 artifact-backed tiling 机制不只适用于 baremix。

## Dual-Source Host Pipeline

split-relu 采用 **dual-source** 编译策略（临时验证方案，非最终通用方案）：

- **Device 路径**：使用 `fc_relu_split_wrapperless.cpp` 作为 device 编译输入，绕开源码内手写 `auto_gen_*` wrapper 与工具链 auto-gen 的冲突。
- **Host 路径**：使用原始 `fc_relu_split_mix.cpp` 经 `host_bisheng` 编译，生成含 `aclrtlaunch_fc_relu_split` launcher 的 host object，再通过 `recompile_binary.py` 重新链接进最终 `.so`。

原始 mix 源码 `examples/matmul-add-relu-sum/fc_relu_split_mix.cpp` 仅用于 host launcher 生成；device 二进制由 wrapperless 路径独立编译产生。

## 当前接受标准

1. **host_dir 非空**：`build/runtime-mix-relu-split/host_dir/objects-Debug/host_bisheng_obj/` 下应有编译产出的 host object。
2. **launcher symbol 存在**：`nm -C .../libfc_relu_split_packed.so | grep aclrtlaunch_fc_relu_split` 应有输出。
3. **runtime 正确性**：最终验收门为 mix-validator 输出 `PASS`（仍在开发中）。

## 运行方式

只在 `xvm` 中支持：

```bash
cd /home/niu/code/Ascend-MLIR
bash examples/relu-split-mix-test/run.sh
```

`run.sh` 默认运行时 kernel 名为：

- `fc_relu_split`
