# 真实NPU环境和使用指导

## 1. SSH连接host信息
真实 host 地址、端口、用户名和认证凭据不要写入仓库。
需要连接共享 real-NPU host 时，通过本地 shell、SSH config 或安全的密钥/密码管理器设置：

```shell
export ASCEND_MLIR_CI_REMOTE='<real host or SSH config alias>'
export ASCEND_MLIR_CI_REMOTE_PORT='<ssh port>'
export ASCEND_MLIR_CI_SSH_PASSWORD='<set locally only>'
```

优先使用 SSH key；如必须使用密码，只在本地环境变量中临时设置
`ASCEND_MLIR_CI_SSH_PASSWORD`，不要提交到文档、脚本或 shell history。

## 2. Host工作目录和环境安装

### 2.1 工作目录
这是一台公共环境，只允许在自己的 `/data/{username}` 目录下新增、修改、删除文件，务必谨记，务必谨记，务必谨记。
本文中的 `{username}` 是占位符，实际操作时替换为自己在该 host 上分配到的目录名。

### 2.2 环境安装
若要使能NPU硬件，需要安装其依赖的驱动包和软件包。

#### 2.2.1 驱动包安装
目前驱动包已经安装好了，可以通过命令`npu-smi info`查看NPU硬件状态，能够看到是Ascend 910C，共16张卡，我们只允许使用7卡(`export ASCEND_DEVICE_ID=7`)。其安装在系统目录/usr/local/Ascend/driver目录下，后续每次建立shell会话，先执行命令`source /usr/local/Ascend/driver/bin/setenv.bash`，设置driver相关环境变量。

#### 2.2.2 软件包安装
软件包(toolkit工具包)只能安装到每位使用者的独立目录，即第1点中的 `/data/{username}/Ascend` 目录下。

##### (1) 软件包下载
* 方式一：社区版（稳定但缺少最新功能），暂不使用此方法。
* 方式二：开发版（及时但不稳定），下载链接：https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/{时间戳}/ 。该目录下有x86和arm(aarch64)两种CPU架构的包。对于我们当前host环境(aarch64)，只下载安装两个包即可：
  * toolkit工具包：[Ascend-cann-toolkit_\${cann_version}_linux-\${arch}.run](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/20260513000324948/Ascend-cann-toolkit_9.1.0_linux-aarch64.run)
  * ops算子包：[Ascend-cann-\${soc_name}-ops_\${cann_version}_linux-\${arch}.run](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/20260513000324948/Ascend-cann-A3-ops_9.1.0_linux-aarch64.run)

##### (2) 软件包安装
针对方式二开发版，安装2个包即可。
```shell
cd /data/{username}
# 确保安装包具有可执行权限
chmod +x Ascend-cann-toolkit_9.1.0_linux-aarch64.run
chmod +x Ascend-cann-A3-ops_9.1.0_linux-aarch64.run
# 安装指定目录 --install-path=/data/{username}/Ascend
./Ascend-cann-toolkit_9.1.0_linux-aarch64.run --full --quiet --install-path=/data/{username}/Ascend
./Ascend-cann-A3-ops_9.1.0_linux-aarch64.run --install --type=toolkit --quiet --install-path=/data/{username}/Ascend

# 本次实测 9.1.0 开发包生成的是 /data/{username}/Ascend/cann/set_env.sh，
# 没有自动生成 /data/{username}/Ascend/latest/set_env.sh。
# 若 /data/{username}/env.sh 仍 source latest，可以补一个兼容软链接。
cd /data/{username}/Ascend
ln -sfn cann latest
```

## 3. 使用方式
按照上述步骤安装完成后，每次建立新的shell session，可以设置以下环境变量：
```shell
source /usr/local/Ascend/driver/bin/setenv.bash
source /data/{username}/Ascend/latest/set_env.sh
export ASCEND_DEVICE_ID=7
```
可以把上述代码保存到 `/data/{username}/env.sh`，后续通过命令 `cd /data/{username}; source /data/{username}/env.sh` 一键设置。

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
4. 打包到 `/data/{username}/<case>`：
   - `bin/runtime-session`：来自 `build-runtime-session-run-only/bin/runtime-session`
   - `artifact/`：包含 `out/manifest.txt` 和 device binary
   - `data/`：输入、expected output、tiling schema 等运行数据
   - `run_manifest.json`
   - `out/`：若 `run_manifest.json` 的 output 写到 `./out/...`，需要预先创建该目录
   - `lib/libstdc++.so.6*`：若远端系统 `libstdc++` 版本过旧，可从 xvm 随包携带

远端运行时：
```shell
cd /data/{username}/<case>
source /data/{username}/env.sh
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
- `copy_tbuf640` / `copy_params640` / `copy_wait640`：验证不同 `DataCopy`
  写法和显式 MTE 同步。
- `copy_scalar640`：验证 scalar GM read/write。
- `const_with_input640`：验证 input 绑定不会影响 write-only compute。
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
for c in const640 copy640 copy_tbuf640 copy_scalar640 copy_params640 \
         copy_wait640 const_with_input640 relu_only broadcast_add; do
  echo "=== ${c} ==="
  cd "/data/{username}/real-npu-microcases/${c}"
  source /data/{username}/env.sh
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

### 3.3 real-NPU 多 kernel 调度用例

`examples/real-npu-multikernel` 用于验证 `runtime-session` 在真实 NPU 上的多 task DAG 调度路径，而不是只验证单个 kernel launch。

当前包含两个用例：

- `serial-two-kernel`：`producer -> consumer`，`producer` 使用 `const640` 生成中间输出，`consumer` 通过 `task_output` 消费该中间输出并执行真机已验证通过的 `copy_scalar640`。该用例验证最小串行依赖、上游输出物化和下游输入绑定。
- `fork-join`：`producer_a` 和 `producer_b` 独立产生两个中间输出，`consumer` 等待二者完成后通过 `task_output` 读取并执行显式流水同步版 `add_wait640`。该用例验证 fork-join DAG、多个上游输出绑定和最终 expected-output 校验。

本地或容器内手动运行：

```shell
source /data/{username}/env.sh
export ASCEND_DEVICE_ID=7
export RUNTIME_SESSION="$PWD/build/bin/runtime-session"
export RUN_ONLY_RUNTIME_SESSION="$PWD/build-runtime-session-run-only/bin/runtime-session"

bash examples/real-npu-multikernel/run.sh --case all
```

`run.sh` 默认每个 case 先生成 `backend=sim` 的 manifest 并用完整 `runtime-session` 过仿真，再用 `RUN_ONLY_RUNTIME_SESSION` 跑 `backend=npu` 真机 manifest。若只做真机定位，可以临时加 `--skip-sim`，但不能用它证明 candidate fix readiness。

通过 `real-npu-ci` 一键跑：

```shell
scripts/sync-and-submit.sh \
  --remote-dir /data/{username}/Codex-Ascend-MLIR-current \
  --case real-npu-multikernel
```

通过标准：

- `serial-two-kernel` 的 sim 和 npu 都必须看到 `session.result=success` 与 `session.validation=pass`。
- `fork-join` 的 sim 和 npu 都必须看到 `session.result=success` 与 `session.validation=pass`。
- sim 日志中 `serial-two-kernel` 应为 `planned_task_count=2`、`serialized_launch_count=2`；`fork-join` 应为 `planned_task_count=3`、`serialized_launch_count=3`。
- npu 日志中 `serial-two-kernel` 应为 `planned_task_count=2`，且有 2 行 `device_id=7` 的 `[npu-launch] kernel=`；`fork-join` 应为 `planned_task_count=3`，且有 3 行 `device_id=7` 的 `[npu-launch] kernel=`。
- plog 中没有新的 `errorStr` 或 vector core exception。

### 3.4 真机调测原则

真机问题要按层收窄，不要直接猜修复点。

1. 先在 xvm 上生成 artifact、数据、expected output 和 run manifest。
2. 任何 candidate fix、generated-kernel 变体或 runtime artifact 上 910C 前，必须先用 xvm Ascend910B1 仿真跑同一个候选，并要求：
   - `session.backend=sim`
   - `session.result=success`
   - `session.validation=pass`
3. 原始 demo 的 xvm pass 只能说明原始 demo 在仿真中可运行，不等价于某个候选修复可上真机；候选修复必须单独过 xvm。
4. early-return checkpoint 等不完整 diagnostic kernel 可以上真机做定位，但不能作为 fix readiness 证明。
5. 真机 status 只报真机事实：case 名、远端目录、device id、pass/fail、错误码、关键 `errorStr`、launch trace 是否排除了 ABI/H2D/GM/tiling 问题。

### 3.5 真机 Debug Playbook

1. 在 `NativeExecutionRunner` launch assembly 附近打开或补充 launch tracing：
   - kernel name
   - binary path
   - block dim
   - input and output counts
   - each input/output byte size
   - GM pointer values and alignment
   - workspace size
   - tiling byte count and first 64-bit words
   - final launch argument count and byte size
2. 通过同一套 xvm-to-remote run-only 打包路径构造 microcase：
   - `const640`：write-only output baseline
   - `copy640`：read input and write output
   - `relu_only`：GM to local compute to GM
   - `broadcast_add`：Broadcast + Add
   - 如需要，再扩展 `transpose-only` 和 full generated kernel case
3. 做 shape 和 scheduling 二分：
   - `block_dim=1` vs 当前 multi-block launch
   - aligned shapes，例如 64 或 128
   - tail shapes，例如 65、127、500、640
4. 如果 full generated kernel 仍是第一个失败点，插入 early-return checkpoint：
   - entry 后立即返回
   - buffer/queue init 后返回
   - first `DataCopy` 后返回
   - compute primitive 后返回
   - final writeback 前后返回
5. 收集远端 CANN logs 和 `npu-smi` 状态，但以 microcase 和 checkpoint 结果作为主要定位信号。
6. 按证据修对应层：
   - trace data 与 generated CANN signature 不一致：修 runtime ABI/launch packing
   - kernel 收到错误 tiling：修 tiling/schema packing
   - 仅某个 primitive、tail path 或 block partition 失败：修 lowering/scheduling/codegen

### 3.6 plog 和 507035 triage

任何真机 `rtStreamSynchronize failed` 都先收集真实 plog `errorStr`，再判断修复点：

```shell
grep -R "errorStr" -n \
  /root/ascend/log/debug/plog \
  /var/log/npu/slog \
  /var/log/npu/plog \
  /root/ascend/log 2>/dev/null | tail -n 50
```

- `507035` / `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION` 只是症状类别，不是根因。必须结合 `errorStr` 和 launch trace 判断。
- 如果 `errorStr` 包含 `ADDR_MISALIGN` 或 `UB address ... is not aligned`，优先检查对应 AscendC API 的 alignment 约束，并打印或核对出错 primitive 附近的 UB offset。
- 如果 `errorStr` 包含 `VEC instruction to read/write UB is out of bounds`，先按 generated-kernel 的 UB lifetime、queue 使用、tile size、primitive shape misuse 定位。
- 如果同时出现 `GM address accessed by scalar exceeds 48 bits`，不要直接归因到 host GM allocation。若 launch trace 显示 H2D/D2H、GM pointer alignment、workspace、argument count、tiling words 都正常，它可能是 kernel 内 UB corruption 的后续症状。
- 在远端 CANN host 重新编译同一个 generated `step8_kernel.cpp` 可以排除 xvm compile artifact mismatch；如果错误不变，继续查 kernel ABI/tiling/lowering，不要停在 host dependency setup。

### 3.7 当前实测结论
- run-only `runtime-session` 在 7 卡真机上可以完成最小 `const640` kernel 的 launch、D2H 和 expected-output 校验，结果为 `session.result=success` / `session.validation=pass`。
- `examples/real-npu-microcases` 已通过 containerized run-only 真机验证：
  - 远端 job：`/data/{username}/real-npu-jobs/20260520-020245-2ecc4d4-real-microcases-microcases`
  - 覆盖 `const640`、copy variants、`relu_only` 和 `broadcast_add`
  - 每个 case 均为 `session.backend=npu` / `session.result=success` / `session.validation=pass`
- real-NPU containerized suite 已通过：
  - 命令：`scripts/sync-and-submit.sh --case all --ref 2ecc4d4-real-all --device-id 7 --jobs 6`
  - 覆盖 `add-broadcast-concat`、`broadcast-add-reduce`、`gather-elementwise-fusion`、
    `matmul-add-leakyrelu`、`relu-broadcast-transpose`、`split-relu-brc-add-mul`
    和 `real-npu-multikernel`
  - 每个 NPU case 均为 `session.backend=npu` / `session.result=success` /
    `session.validation=pass`
  - `real-npu-multikernel` 的两个 DAG case 均先过 `sim` 再过 `npu`
  - collected plog summaries 没有新的 `errorStr`
- `examples/relu-broadcast-transpose` 的 full-pipeline vec kernel 已在 7 卡真机通过：
  - xvm Ascend910B1 仿真先通过：`session.backend=sim` / `session.result=success` / `session.validation=pass`
  - 远端 run-only 包：`/data/{username}/relu-broadcast-transpose-final-real-20260518-190410`
  - 远端结果：`session.backend=npu` / `session.result=success` / `session.validation=pass`
- `examples/add-broadcast-concat` 的 full-pipeline vec kernel 已在 7 卡真机通过：
  - xvm Ascend910B1 仿真先通过：`session.backend=sim` / `session.result=success` / `session.validation=pass`
  - 远端 run-only 包：`/data/{username}/examples-real-add-broadcast-concat-tbn16-20260518-194533`
  - 远端 launch trace：`kernel=ewop_broadcast_concat` / `block_dim=10` / `tiling.word[0]=64` / `tiling.word[1]=16`
  - 远端结果：`session.backend=npu` / `session.result=success` / `session.validation=pass`
  - 定位结论：原 `TB_N=192` 在该 generated kernel 中相当于过大的 inner M tile；`N=500` 保持完整宽度进入 UB，导致 UB live set 超界并触发 `rtStreamSynchronize rc=507035`。当前默认配置改为 `TB_N=16`。
- `examples/split-relu-brc-add-mul` 的 full-pipeline vec kernel 已在 7 卡真机通过：
  - xvm Ascend910B1 仿真先通过：`session.backend=sim` / `session.result=success` / `session.validation=pass`
  - 远端 run-only 包：`/data/{username}/examples-real-split-relu-brc-add-mul-dead-tbuf-20260518-195902`
  - 远端 launch trace：`kernel=ewop_broadcast_split` / `block_dim=20` / `args_count=15` / `tiling_words=8`
  - 远端结果：`session.backend=npu` / `session.result=success` / `session.validation=pass`
  - 定位结论：仅把 tiling schema 从 12 字段裁到 CANN 签名的 8 字段仍失败；根因是 queue-backed memref alloc 同时保留了无实际用户的 TBuf `InitBuffer`，两条分支共享一个 `TPipe` 时 UB live set 超界。删除 dead TBuf initializer 后，generated kernel 的 `InitBuffer` 从 20 个降到 14 个并通过真机。
- 本次 `rtStreamSynchronize failed: rc=507035` 定位结论：
  - launch ABI、H2D/D2H、GM pointer 512B 对齐、workspace、tiling words 均正常
  - plog 主要报 `The address for the VEC instruction to read/write UB is out of bounds.The GM address accessed by scalar exceeds 48 bits.`
  - 未 hoist 的 `TB_N=16` 版本仍失败：`/data/{username}/relu-broadcast-transpose-tb16-real-20260518-183600`
  - hoist 后版本通过，说明根因是 generated kernel 的 UB/queue 生命周期与 tail alloc 尺寸上界处理：
    - VECOUT 输出 tensor 必须从 VECOUT queue `AllocTensor`
    - GM 输入经临时 VECIN queue `DeQue` 后必须 `FreeTensor`
    - `affine.min` / `memref.dim(subview)` 产生的 tail alloc 尺寸要解析为 loop-step 上界
    - loop-invariant `InitBuffer` / `InitQueue` 必须 hoist 到 inner tile loop 外

### 3.8 容器化 real-NPU runner

为了让 x86 和 aarch64 开发机使用同一套真机验证入口，新增 `scripts/real-npu-ci/`：

- 开发机只负责通过 SSH 或其它 CI 入口触发任务。
- 实际 clone/copy 源码、构建 Ascend-MLIR、生成 artifacts、运行 xvm-style sim pipeline、改写 manifest 到 `npu` backend、真实 NPU运行和 plog 收集都在 real-NPU aarch64 host 的 Docker 容器内完成。
- NPU driver 不打包进镜像，运行时从 host 挂载 `/usr/local/Ascend/driver` 和 `/dev/davinci*`。
- CANN toolkit 默认使用 host 上的 `/data/{username}/Ascend/latest`，通过挂载 `/data/{username}` 进入容器。
- LLVM/MLIR 依赖推荐通过 `build-aarch64-image.sh --with-llvm` 在本机 OrbStack、xvm、native arm64 Linux 或 CI 上自动构建，并提前 bake 到 builder image 的 `/opt/llvm/build`。
- real-NPU host 的常规职责是 pull/load 已构建好的 image 并运行验证 job；不要把镜像构建放进每次真机验证流程。

在 arm64 Docker builder 上构建基础镜像：

```shell
scripts/real-npu-ci/build-aarch64-image.sh \
  --tag ascend-mlir-builder:aarch64-ubuntu22.04
```

更推荐提前构建带 LLVM/MLIR 的共享镜像。版本 pin 记录在
`scripts/real-npu-ci/versions.env`，后续 LLVM/MLIR 版本变更时更新该文件并重建镜像。

首选方式：镜像构建时自动 clone/build pinned LLVM/MLIR：

```shell
scripts/real-npu-ci/build-aarch64-image.sh --with-llvm
```

如果已有确认可用的 arm64 LLVM build，也可以把它 bake 进去以加速镜像构建：

```shell
scripts/real-npu-ci/build-aarch64-image.sh \
  --tag ascend-mlir-builder:aarch64-ubuntu22.04-llvm21 \
  --embed-llvm-build-dir /opt/llvm/build
```

镜像构建完成后，把 image 推到 registry 让 real-NPU host pull，或者 `docker save`
后复制到 real-NPU host 并 `docker load`。`submit.sh` / `docker-run.sh`
只使用 real-NPU host 上已经存在的 image tag，不负责构建镜像。

PyAsc 跟随被验证的源码仓库 ref，不固定 bake 到通用镜像里；当 PyAsc 需要新的
LLVM/MLIR 或系统依赖时，再更新 `versions.env` / Dockerfile 并重建镜像。
镜像 tag 不编码 host CANN 版本；CANN 是运行时从 host 挂载进容器的依赖，而不是
镜像内置依赖。

默认 base image 是 Ubuntu 22.04。real-NPU host 本身不要求是 Ubuntu，因为容器自带
userland，只共享 host kernel。约束是运行也要留在容器内；如果把容器内编译出的
二进制拷到 host 裸跑，就需要单独检查 glibc 兼容性。

real-NPU host 上用当前源码树触发一个 case：

```shell
cd /data/{username}/Codex-Ascend-MLIR
scripts/real-npu-ci/docker-run.sh \
  --image ascend-mlir-builder:aarch64-ubuntu22.04-llvm21 \
  --source-dir "$PWD" \
  --ref "$(git rev-parse --short HEAD)" \
  --case relu-broadcast-transpose \
  --device-id 7
```

任意开发机远程触发同一流程：

```shell
scripts/real-npu-ci/submit.sh \
  --image ascend-mlir-builder:aarch64-ubuntu22.04-llvm21 \
  --repo-url git@example.com:team/Codex-Ascend-MLIR.git \
  --ref my-branch \
  --case relu-broadcast-transpose \
  --device-id 7
```

job 输出统一落在：

```text
/data/{username}/real-npu-jobs/<timestamp>-<ref>-<case>/
  job-env.txt
  logs/
    build-project.log
    <case>-sim.log
    <case>-npu.log
    plog/
      npu-smi.txt
      plog-errorStr.txt
      plog-files.txt
  out/
```

详细参数和运行方式见 `scripts/real-npu-ci/README.md`。

## 4. 真机问题经验总结

### 4.1 `relu-broadcast-transpose` 507035 lessons

- 之前 `examples/relu-broadcast-transpose` 真机失败不是 launch ABI 问题。H2D roundtrip、GM pointer 512B alignment、workspace、argument count 和 tiling words 都正常。
- 失败 plog 报 VEC UB out-of-bounds / scalar GM address over 48 bits，最终修复点在 generated kernel lowering 和 tile sizing。
- VECOUT outputs 必须先从 VECOUT queue `AllocTensor` 再 enqueue。把 VECCALC tensor enqueue 到 VECOUT queue 可能过仿真，但真机会触发 UB/MTE fault。
- 临时 GM-to-VECIN tensor 通过 `AllocTensor -> DataCopy -> EnQue -> DeQue` 创建后，必须在最后一次使用后 `FreeTensor`。
- all-parallel tail-tiled kernel 的 buffer allocation 应使用 enclosing loop-step upper bound；DataCopy/compute 仍使用 actual tail element count。这样一个 max-sized queue/tbuf 可以跨 tail iteration 复用。
- 计算 all-parallel VECOUT/VECIN/VECCALC buffer byte size 时，要把 `affine.min(remaining, step)` 和 `memref.dim(subview)` 解析到 loop-step upper bound。
- loop-invariant `InitBuffer` / `InitQueue` 要 hoist 到 inner tile loop 外。每轮循环重复 init 在 simulator 与真机上的 UB 消耗表现不同，真机可能触发 `507035`。
- 不要把同样的 max-size substitution 盲目套到 reduction output。rank-1 VECOUT reduction output 可能需要 exact tail size，因为 `ReduceSum2DL2` codegen 会从 source 和 destination tensor size 推导 rows/cols。
- `TB_N` 从 64 调到 16 降低了该 demo 的 UB live set，是当前真机通过配置的一部分；但 tile 修改本身不是根因修复，必须同时保证 queue 和 buffer lifetime 正确。

### 4.2 `add-broadcast-concat` 507035 lessons

- 之前 `examples/add-broadcast-concat` 真机失败是 tiling configuration 问题，不是 launch ABI 问题。launch trace 显示 argument count、H2D/D2H、512B-aligned GM pointers、workspace、tiling words 都正常。
- 在该 generated kernel 中，`TB_N` 当前实际表现为 inner M tile size；`N=500` 仍 full-width 进入每个 tile。`TB_N=192` 会让 UB live set 超界，并在 910C 上触发 `rtStreamSynchronize rc=507035`。
- 当前接受配置是 `TB_M=64, TB_N=16`。流程上先跑 xvm Ascend910B1 simulation，再跑 真实 NPU。

### 4.3 `split-relu-brc-add-mul` 507035 lessons

- 让 run manifest tiling fields 对齐 generated CANN `TilingData` signature 是必要 hygiene，但不是该 demo 的根因。8-field tiling probe 在 UB cleanup 前仍以同样 `507035` 失败。
- 根因是 queue-backed memref alloc 还保留了 standalone TBuf initializer。这些 TBuf 没有实际用户，只有 `TPipe.InitBuffer`，但 hoist 后仍会消耗真机 UB。
- data-move/compute conversion 后，应删除仅被 `TPipe.InitBuffer` 使用的 TBuf。该 demo 中 generated `InitBuffer` 从 20 个降到 14 个后，xvm simulation 保持通过，真实 NPU通过。
- `examples/split-relu-brc-add-mul` run manifest 要和当前 CANN signature 对齐：`TB_M`、`TB_N`、`dim_arg0_1`、`dim_arg1_0`、`dim_arg0_0`、`dim_arg3_0`、`dim_arg2_0`、`dim_arg4_0`。

### 4.4 microcase synchronization lessons

- `copy640`、`copy_tbuf640` 和 `copy_params640` 最初在真机上 validation fail，
  但 H2D roundtrip、launch 参数和 GM pointer alignment 均正常；根因是手写
  microcase kernel 缺少 MTE2 -> MTE3 同步，真机不会像仿真路径那样隐式掩盖。
- `relu_only` 在补 MTE2 -> V / V -> MTE3 后仍失败，说明问题不在 runtime ABI。
  将 `Duplicate(zero) + Max(x, zero)` 收敛为 `Maxs(x, 0)` 后真机通过，后续
  需要单独验证 `Duplicate + binary Max` 时不要把它混入基础 relu microcase。
- `scripts/real-npu-ci/run-real-npu-job.sh` 必须使用显式 microcase 顺序；glob
  字母序会让 `broadcast_add` 先运行，掩盖更基础的 `DataCopy` / vector 边界。
