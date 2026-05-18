# 真实NPU环境和使用指导

## 1. SSH连接host信息
公网IP：<real-npu-host>
端口号：141
用户名：root
密码：<redacted-password>

## 2. Host工作目录和环境安装

### 2.1 工作目录
这是一台公共环境，只允许在 `/data/nyh` 目录下新增、修改、删除文件，务必谨记，务必谨记，务必谨记。

### 2.2 环境安装
若要使能NPU硬件，需要安装其依赖的驱动包和软件包。

#### 2.2.1 驱动包安装
目前驱动包已经安装好了，可以通过命令`npu-smi info`查看NPU硬件状态，能够看到是Ascend 910C，共16张卡，我们只允许使用7卡(`export ASCEND_DEVICE_ID=7`)。其安装在系统目录/usr/local/Ascend/driver目录下，后续每次建立shell会话，先执行命令`source /usr/local/Ascend/driver/bin/setenv.bash`，设置driver相关环境变量。

#### 2.2.2 软件包安装
软件包(toolkit工具包)只能安装到每位使用者的独立目录，比如我们的话就是第1点中的`/data/nyh/Ascend`目录下。

##### (1) 软件包下载
* 方式一：社区版（稳定但缺少最新功能），暂不使用此方法。
* 方式二：开发版（及时但不稳定），下载链接：https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/{时间戳}/ 。该目录下有x86和arm(aarch64)两种CPU架构的包。对于我们当前host环境(aarch64)，只下载安装两个包即可：
  * toolkit工具包：[Ascend-cann-toolkit_\${cann_version}_linux-\${arch}.run](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/20260513000324948/Ascend-cann-toolkit_9.1.0_linux-aarch64.run)
  * ops算子包：[Ascend-cann-\${soc_name}-ops_\${cann_version}_linux-\${arch}.run](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/20260513000324948/Ascend-cann-A3-ops_9.1.0_linux-aarch64.run)

##### (2) 软件包安装
针对方式二开发版，安装2个包即可。
```shell
cd /data/nyh
# 确保安装包具有可执行权限
chmod +x Ascend-cann-toolkit_9.1.0_linux-aarch64.run
chmod +x Ascend-cann-A3-ops_9.1.0_linux-aarch64.run
# 安装指定目录 --install-path=/data/nyh/Ascend
./Ascend-cann-toolkit_9.1.0_linux-aarch64.run --full --quiet --install-path=/data/nyh/Ascend
./Ascend-cann-A3-ops_9.1.0_linux-aarch64.run --install --type=toolkit --quiet --install-path=/data/nyh/Ascend

# 本次实测 9.1.0 开发包生成的是 /data/nyh/Ascend/cann/set_env.sh，
# 没有自动生成 /data/nyh/Ascend/latest/set_env.sh。
# 若 /data/nyh/env.sh 仍 source latest，可以补一个兼容软链接。
cd /data/nyh/Ascend
ln -sfn cann latest
```

##### (3) xvm软件包安装
xvm 也需要安装同版本 toolkit 和 A3 ops，用于在 xvm 上完成 artifact 编译、打包和
run-manifest-only `runtime-session` 构建。安装包下载链接与远端 host 相同，但安装根目录
使用 xvm 当前用户的 `~/Ascend`。

```shell
mkdir -p ~/Ascend/installers
cd ~/Ascend/installers
wget -O Ascend-cann-toolkit_9.1.0_linux-aarch64.run \
  https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/20260513000324948/Ascend-cann-toolkit_9.1.0_linux-aarch64.run
wget -O Ascend-cann-A3-ops_9.1.0_linux-aarch64.run \
  https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/20260513000324948/Ascend-cann-A3-ops_9.1.0_linux-aarch64.run

chmod +x Ascend-cann-toolkit_9.1.0_linux-aarch64.run
chmod +x Ascend-cann-A3-ops_9.1.0_linux-aarch64.run

./Ascend-cann-toolkit_9.1.0_linux-aarch64.run --full --quiet --install-path=$HOME/Ascend
./Ascend-cann-A3-ops_9.1.0_linux-aarch64.run --install --type=toolkit --quiet --install-path=$HOME/Ascend

# 本次 xvm 实测安装器会生成 ~/Ascend/cann -> ~/Ascend/cann-9.1.0，
# 但顶层 ~/Ascend/latest 仍可能保留为旧 9.0 指向，需要显式切到 9.1。
ln -sfn "$HOME/Ascend/ascend-toolkit/latest" "$HOME/Ascend/latest"

source ~/Ascend/latest/set_env.sh
```

## 3. 使用方式
按照上述步骤安装完成后，每次建立新的shell session，可以设置以下环境变量：
```shell
source /usr/local/Ascend/driver/bin/setenv.bash
source /data/nyh/Ascend/latest/set_env.sh
export ASCEND_DEVICE_ID=7
```
我把上述代码保存到了`/data/nyh/env.sh`，可以通过命令`cd /data/nyh; source /data/nyh/env.sh`一键设置。

### 3.1 远端 host 无 LLVM 时的运行方式
远端 host 只作为运行环境使用，不在远端构建 MLIR/LLVM 工程：

1. 在 xvm 上生成 artifact、输入数据、expected output 和 run manifest。
2. 在 xvm 上构建 run-manifest-only 的 `runtime-session`：
   ```shell
   cmake -G Ninja -S . -B build-runtime-session-run-only \
     -DLLVM_BUILD_DIR="${LLVM_BUILD}" \
     -DASCEND_RUNTIME_SESSION_RUN_ONLY=ON
   cmake --build build-runtime-session-run-only --target runtime-session -j2
   ```
3. 用以下命令检查该 runner 不会在进程启动时直接加载 CANN 编译/仿真库：
   ```shell
   bash test/tools/runtime/run_runtime_session_run_only_link_smoke.sh
   ```
4. 打包到 `/data/nyh/<case>`：
   - `bin/runtime-session`：来自 `build-runtime-session-run-only/bin/runtime-session`
   - `artifact/`：包含 `out/manifest.txt` 和 device binary
   - `data/`：输入、expected output、tiling schema 等运行数据
   - `run_manifest.json`
   - `out/`：若 `run_manifest.json` 的 output 写到 `./out/...`，需要预先创建该目录
   - `lib/libstdc++.so.6*`：若远端系统 `libstdc++` 版本过旧，可从 xvm 随包携带

远端运行时：
```shell
cd /data/nyh/<case>
source /data/nyh/env.sh
export LD_LIBRARY_PATH="$PWD/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_DEVICE_ID=7
export ASCEND_RUNTIME_TRACE_LAUNCH=1
./bin/runtime-session --run-manifest ./run_manifest.json --run
```

注意：不要随包覆盖远端系统 `libgcc_s.so.1`；xvm 的 `libgcc_s.so.1` 依赖 `GLIBC_2.35`，而当前远端 host 是 glibc 2.34。当前实测只需要随包携带 `libstdc++`，`libgcc_s` 使用远端系统版本即可。

### 3.2 real-NPU microcase 准备

`examples/real-npu-microcases` 提供第一组定位用 microcase：

- `const640`：只写 output，验证 launch、D2H 和 expected-output 校验基线。
- `copy640`：读一个 input 并写 output，验证 H2D、GM read 和基础 `DataCopy`。
- `relu_only`：GM -> UB -> `Max` -> GM，验证基础 vector compute。
- `broadcast_add`：`Broadcast + Add`，验证 broadcast primitive 和二输入向量路径。

在 xvm 上准备 artifacts 和 run manifests：

```shell
source /home/niu/Ascend/latest/set_env.sh
export RUNTIME_SESSION="$PWD/build/bin/runtime-session"
bash examples/real-npu-microcases/prepare.sh --out-dir /tmp/real-npu-microcases
```

打包到远端时，每个 case 目录至少包含：

- `artifact/`
- `run_manifest.json`
- 输入 `.npy`、`expected.npy`
- run-only `bin/runtime-session`
- 必要时附带 `lib/libstdc++.so.6*`

远端运行顺序：

```shell
for c in const640 copy640 relu_only broadcast_add; do
  echo "=== ${c} ==="
  cd "/data/nyh/real-npu-microcases/${c}"
  source /data/nyh/env.sh
  export LD_LIBRARY_PATH="$PWD/lib:${LD_LIBRARY_PATH:-}"
  export ASCEND_DEVICE_ID=7
  export ASCEND_RUNTIME_TRACE_LAUNCH=1
  ./bin/runtime-session --run-manifest ./run_manifest.json --run
done
```

判断边界：

- `const640` 失败：优先查 run-only runner、device binary 注册、launch 参数数量或 output GM。
- `copy640` 首次失败：优先查 H2D、input GM pointer、GM read `DataCopy`。
- `relu_only` 首次失败：优先查 UB buffer 或基础 vector primitive。
- `broadcast_add` 首次失败：优先查 `Broadcast` lowering/API 或二输入向量路径。
- microcase 全过但 full case 失败：继续做 shape/tile 二分和 generated kernel checkpoint。

### 3.3 当前实测结论
- run-only `runtime-session` 在 7 卡真机上可以完成最小 `const640` kernel 的 launch、D2H 和 expected-output 校验，结果为 `session.result=success` / `session.validation=pass`。
- `examples/relu-broadcast-transpose` 的 full-pipeline vec kernel 已在 7 卡真机通过：
  - xvm Ascend910B1 仿真先通过：`session.backend=sim` / `session.result=success` / `session.validation=pass`
  - 远端 run-only 包：`/data/nyh/relu-broadcast-transpose-final-real-20260518-190410`
  - 远端结果：`session.backend=npu` / `session.result=success` / `session.validation=pass`
- `examples/add-broadcast-concat` 的 full-pipeline vec kernel 已在 7 卡真机通过：
  - xvm Ascend910B1 仿真先通过：`session.backend=sim` / `session.result=success` / `session.validation=pass`
  - 远端 run-only 包：`/data/nyh/examples-real-add-broadcast-concat-tbn16-20260518-194533`
  - 远端 launch trace：`kernel=ewop_broadcast_concat` / `block_dim=10` / `tiling.word[0]=64` / `tiling.word[1]=16`
  - 远端结果：`session.backend=npu` / `session.result=success` / `session.validation=pass`
  - 定位结论：原 `TB_N=192` 在该 generated kernel 中相当于过大的 inner M tile；`N=500` 保持完整宽度进入 UB，导致 UB live set 超界并触发 `rtStreamSynchronize rc=507035`。当前默认配置改为 `TB_N=16`。
- `examples/split-relu-brc-add-mul` 的 full-pipeline vec kernel 已在 7 卡真机通过：
  - xvm Ascend910B1 仿真先通过：`session.backend=sim` / `session.result=success` / `session.validation=pass`
  - 远端 run-only 包：`/data/nyh/examples-real-split-relu-brc-add-mul-dead-tbuf-20260518-195902`
  - 远端 launch trace：`kernel=ewop_broadcast_split` / `block_dim=20` / `args_count=15` / `tiling_words=8`
  - 远端结果：`session.backend=npu` / `session.result=success` / `session.validation=pass`
  - 定位结论：仅把 tiling schema 从 12 字段裁到 CANN 签名的 8 字段仍失败；根因是 queue-backed memref alloc 同时保留了无实际用户的 TBuf `InitBuffer`，两条分支共享一个 `TPipe` 时 UB live set 超界。删除 dead TBuf initializer 后，generated kernel 的 `InitBuffer` 从 20 个降到 14 个并通过真机。
- 本次 `rtStreamSynchronize failed: rc=507035` 定位结论：
  - launch ABI、H2D/D2H、GM pointer 512B 对齐、workspace、tiling words 均正常
  - plog 主要报 `The address for the VEC instruction to read/write UB is out of bounds.The GM address accessed by scalar exceeds 48 bits.`
  - 未 hoist 的 `TB_N=16` 版本仍失败：`/data/nyh/relu-broadcast-transpose-tb16-real-20260518-183600`
  - hoist 后版本通过，说明根因是 generated kernel 的 UB/queue 生命周期与 tail alloc 尺寸上界处理：
    - VECOUT 输出 tensor 必须从 VECOUT queue `AllocTensor`
    - GM 输入经临时 VECIN queue `DeQue` 后必须 `FreeTensor`
    - `affine.min` / `memref.dim(subview)` 产生的 tail alloc 尺寸要解析为 loop-step 上界
    - loop-invariant `InitBuffer` / `InitQueue` 必须 hoist 到 inner tile loop 外
