# real-npu-ci 使用说明

这个目录提供共享真实 NPU 验证流程。目标是让没有本地 aarch64/NPU 环境的开发者也能按同一套步骤完成：

1. 在可构建 aarch64 Docker image 的机器上预构建 builder image。
2. 将 image 推送到 Huawei Cloud SWR，或离线 `docker save` 分发。
3. 在 real-NPU host 上 `docker pull/load` 已构建好的 image。
4. 在容器内构建当前 Ascend-MLIR 源码，先跑 sim gate，再用 run-only `runtime-session` 跑真机。

## 0. 总体边界

```text
本机 OrbStack / xvm / arm64 CI
  -> build aarch64 builder image
  -> push Huawei Cloud SWR 或 docker save

real-NPU host
  -> docker pull 或 docker load 已构建好的 image
  -> docker-run.sh
  -> docker run builder container
  -> 容器内 build Ascend-MLIR、sim gate、run-only runtime-session、NPU run
```

- real-NPU host 常规流程只 `pull/load/run` image，不在每次验证 job 里构建 image。
- CANN toolkit 和 driver 不打进 image，运行时从 real-NPU host 挂载。
- image tag 不编码 CANN 版本；CANN 是运行时挂载依赖，不是镜像内置依赖。
- 推荐使用带 LLVM/MLIR 的预构建 image，让每个验证 job 只构建当前 Ascend-MLIR 源码。
- 当前 real-NPU Docker 访问 NPU 需要 `--privileged`；`docker-run.sh` 已默认加入。

## 1. 文件说明

| 文件 | 作用 | 何时使用 |
| --- | --- | --- |
| `.dockerignore` | 控制 Docker build context，只把 Dockerfile、脚本、README、`versions.env` 和可选 `llvm-build/` 放进镜像构建上下文。 | `docker buildx` 自动使用，不手动执行。 |
| `Dockerfile.builder` | 定义 aarch64 builder image：安装基础工具、Python 依赖，并可选 clone/build LLVM/MLIR 到 `/opt/llvm/build`。 | 由 `build-aarch64-image.sh` 调用。 |
| `versions.env` | 集中记录默认 image tag、base image、LLVM repo/ref/build type/assertions。 | 升级 LLVM/MLIR 或默认 image 名时修改。 |
| `build-aarch64-image.sh` | 构建、push、save builder image 的主入口。默认关闭 provenance/SBOM，产出 SWR 可接受的单架构 image。 | 在本机 OrbStack、xvm、native arm64 Linux 或 CI 上运行。 |
| `docker-run.sh` | real-NPU host 侧 `docker run` 包装脚本。负责 `--privileged`、device、driver、CANN、job 目录挂载，并把参数传给容器。 | 在 real-NPU host 上运行，前提是 image 已存在。 |
| `run-real-npu-job.sh` | 容器 ENTRYPOINT。负责 clone/copy 源码、构建 Ascend-MLIR、跑 sim pipeline、构建 run-only `runtime-session`、改写 manifest 为 `npu`、执行真机验证。 | 通常不手动运行，由 Docker ENTRYPOINT 自动执行。 |
| `collect-plog.sh` | 收集 `npu-smi info`、近期 plog 文件列表和 `errorStr`。 | `run-real-npu-job.sh` 结束时自动调用，也可在容器内手动调用。 |
| `submit.sh` | 开发机侧 SSH 提交器。登录 real-NPU host 后调用远端 `docker-run.sh`。 | x86/aarch64 开发机远程触发真机验证时使用。 |
| `../sync-and-submit.sh` | 开发机侧一键入口。打包本地 git 工作树，同步到远端指定目录，然后用已有 image 跑指定用例。 | 需要验证本地未 push 改动时使用。 |
| `README.md` | 本说明文档。 | 维护此目录流程和约定。 |

## 2. 准备变量（首次配置或凭证变化时执行）

下面命令里的 token、密码、仓库 URL 请按实际环境替换。不要把真实 token 写进仓库。

```shell
export SWR_REGISTRY=swr.cn-east-2.myhuaweicloud.com
export SWR_NAMESPACE=ascendmlir
export IMAGE_NAME=ascend-mlir-builder
export IMAGE_TAG=aarch64-ubuntu22.04-llvm21
export SWR_IMAGE="${SWR_REGISTRY}/${SWR_NAMESPACE}/${IMAGE_NAME}:${IMAGE_TAG}"

export SWR_USER='<SWR login username>'
export SWR_TOKEN='<SWR temporary password or token>'
```

本次验证过的 tag 形态是：

```text
swr.cn-east-2.myhuaweicloud.com/ascendmlir/ascend-mlir-builder:aarch64-ubuntu22.04-llvm21
```

## 3. 构建 image（依赖栈变化时执行）

在 arm64-capable Docker 环境构建 image，例如本机 OrbStack、xvm、native arm64 Linux 或 arm64 CI。不要在 real-NPU host 的每次验证 job 中构建 image。
如果 LLVM、系统依赖、Python 依赖或本目录镜像脚本没有变化，不需要每次真机验证都重新构建 image。

### 3.1 标准构建

当前推荐构建带 LLVM/MLIR 的共享 image。LLVM 构建耗时较长，首次通常约 1.5 小时。

```shell
scripts/real-npu-ci/build-aarch64-image.sh \
  --with-llvm \
  --tag "${SWR_IMAGE}"
```

`build-aarch64-image.sh` 默认使用：

- `docker buildx build --load`
- `--platform linux/arm64`
- `--provenance=false`
- `--sbom=false`

关闭 provenance/SBOM 是为了兼容 SWR。若直接推送 BuildKit 默认 attestation manifest，SWR 可能报：

```text
Invalid image, fail to parse 'manifest.json'
```

### 3.2 本地自检

构建后先确认 image 是 arm64，且 LLVM/MLIR 和 Python 依赖存在。

```shell
docker run --rm --entrypoint bash "${SWR_IMAGE}" -lc '
  set -e
  uname -m
  /opt/llvm/build/bin/llvm-config --version --host-target
  test -d /opt/llvm/build/lib/cmake/mlir && echo "mlir-cmake-ok"
  python3 - <<PY
import numpy, pybind11, nanobind, yaml, pygments
print("python-deps-ok")
PY
  /opt/ascend-mlir-ci/build-aarch64-image.sh --help | sed -n "1,12p"
'
```

期望至少看到：

```text
aarch64
mlir-cmake-ok
python-deps-ok
```

## 4. 上传到 Huawei Cloud SWR（image 更新后执行）

只有新构建或重新打 tag 的 image 需要上传。日常运行已有 image 的真机验证 job 时，不需要重复执行本节。

### 4.1 登录

使用 `--password-stdin`，不要使用 `docker login -p ...`，避免 token 暴露在进程参数或 shell history 中。

```shell
printf '%s' "${SWR_TOKEN}" | \
  docker login -u "${SWR_USER}" --password-stdin "${SWR_REGISTRY}"
```

### 4.2 推送

```shell
docker push "${SWR_IMAGE}"
```

成功时会看到类似输出：

```text
${IMAGE_TAG}: digest: sha256:<digest> size: <size>
```

记录这个 digest，后续排查时可以确认 real-NPU host pull 到的是同一个 image。

### 4.3 登出

临时账号使用完后登出。

```shell
docker logout "${SWR_REGISTRY}"
```

## 5. real-NPU host 下载 image（首次或 image 更新后执行）

在 real-NPU host 上执行。当前公共 host 的工作目录约束见 `examples/real-npu.md`：只在 `/data/{username}` 下新增、修改、删除文件，默认使用 `ASCEND_DEVICE_ID=7`。
本文中的 `{username}` 是占位符，实际操作时替换为自己在该 host 上分配到的目录名。
host 上已经存在目标 image tag 时，不需要每次 job 都重新 `docker pull`。

### 5.1 登录 host

```shell
ssh -p 141 root@<real-npu-host>
```

### 5.2 登录 SWR 并 pull

```shell
export SWR_REGISTRY=swr.cn-east-2.myhuaweicloud.com
export SWR_NAMESPACE=ascendmlir
export IMAGE_NAME=ascend-mlir-builder
export IMAGE_TAG=aarch64-ubuntu22.04-llvm21
export SWR_IMAGE="${SWR_REGISTRY}/${SWR_NAMESPACE}/${IMAGE_NAME}:${IMAGE_TAG}"

export SWR_USER='<SWR login username>'
export SWR_TOKEN='<SWR temporary password or token>'

printf '%s' "${SWR_TOKEN}" | \
  docker login -u "${SWR_USER}" --password-stdin "${SWR_REGISTRY}"

docker pull "${SWR_IMAGE}"

docker image inspect "${SWR_IMAGE}" \
  --format 'tag={{index .RepoTags 0}} arch={{.Architecture}} os={{.Os}} id={{.Id}} size={{.Size}}'
```

确认 `arch=arm64` 或 `arch=aarch64` 对应的 arm64 image。使用临时账号后可以登出：

```shell
docker logout "${SWR_REGISTRY}"
```

## 6. 准备 real-NPU host 源码（按验证方式选择执行）

有两种方式，优先使用“已 push 分支/commit”方式；只有本地未提交改动需要实测时，才同步本地工作树。
如果使用 `scripts/sync-and-submit.sh`，脚本会自动同步本地工作树，通常不需要手动执行 6.2。

### 6.1 已 push 分支或 commit

在 real-NPU host 上准备仓库：

```shell
cd /data/{username}
git clone <repo-url> Codex-Ascend-MLIR
cd /data/{username}/Codex-Ascend-MLIR
git fetch --all --prune
git checkout <branch-or-commit>
```

后续运行时使用 `--repo-url/--ref` 或 `--source-dir` 都可以。若用 `--repo-url/--ref`，容器会自己 clone 指定 ref。

### 6.2 同步本地未提交工作树

在开发机执行。这个方式只用于临时验证本地未 push 的改动。

```shell
ssh -p 141 root@<real-npu-host> \
  'rm -rf /data/{username}/Codex-Ascend-MLIR-current && mkdir -p /data/{username}/Codex-Ascend-MLIR-current'

COPYFILE_DISABLE=1 tar --no-xattrs \
  --exclude='./.git' \
  --exclude='./build' \
  --exclude='./out' \
  -czf - . | \
ssh -p 141 root@<real-npu-host> \
  'tar --no-xattrs -C /data/{username}/Codex-Ascend-MLIR-current -xzf -'
```

`COPYFILE_DISABLE=1` 和 `--no-xattrs` 用于避免 macOS 的 `._*` 扩展属性文件混入远端源码树。

## 7. 在 real-NPU host 上运行真机验证（每次验证执行）

### 7.1 host 已有源码树模式

适合验证 `/data/{username}/Codex-Ascend-MLIR-current` 或 `/data/{username}/Codex-Ascend-MLIR` 中已有的源码。

```shell
cd /data/{username}/Codex-Ascend-MLIR-current

scripts/real-npu-ci/docker-run.sh \
  --image "${SWR_IMAGE}" \
  --source-dir /data/{username}/Codex-Ascend-MLIR-current \
  --ref "$(git rev-parse --short HEAD 2>/dev/null || echo local-tree)" \
  --case relu-broadcast-transpose \
  --device-id 7 \
  --jobs 6
```

### 7.2 repo/ref 模式

适合验证已经 push 的分支或 commit。容器会 clone 指定仓库和 ref。

```shell
cd /data/{username}/Codex-Ascend-MLIR

scripts/real-npu-ci/docker-run.sh \
  --image "${SWR_IMAGE}" \
  --repo-url <repo-url> \
  --ref <branch-or-commit> \
  --case relu-broadcast-transpose \
  --device-id 7 \
  --jobs 6
```

### 7.3 一键同步本地工作树并运行

适合验证本地未 push 的改动。脚本会：

1. 从当前 git 工作树选取 tracked files、已展开的 submodule files 和未被 `.gitignore` 忽略的 untracked files。
2. 替换远端 `--remote-dir` 指定的目录。
3. 默认保留该目录下的 `build/` 和 `build-runtime-session-run-only/`，用于增量编译。
4. SSH 到 real-NPU host，在该目录下调用 `docker-run.sh`。
5. 使用远端已经存在的 Docker image 跑指定 example case。

```shell
scripts/sync-and-submit.sh \
  --remote-dir /data/{username}/Codex-Ascend-MLIR-current \
  --case relu-broadcast-transpose
```

如果要执行任意命令，使用 `--cmd`。命令会在容器内完成项目构建后，从源码根目录执行，并预先设置：

- `PATH`：包含 `build/bin`
- `AFIR_OPT`
- `AFIR_TRANSLATE`
- `RUNTIME_SESSION`
- `RUN_ONLY_RUNTIME_SESSION`

例如跑全量 example pipeline：

```shell
scripts/sync-and-submit.sh \
  --remote-dir /data/{username}/Codex-Ascend-MLIR-current \
  --cmd 'bash test/tools/examples/example_pipelines.sh'
```

再比如跑 runtime baseline：

```shell
scripts/sync-and-submit.sh \
  --remote-dir /data/{username}/Codex-Ascend-MLIR-current \
  --cmd 'bash test/tools/runtime/run_runtime.sh'
```

默认值：

- `--image`：来自 `versions.env` 的 `ASCEND_MLIR_CI_DEFAULT_REMOTE_IMAGE`，当前是 `swr.cn-east-2.myhuaweicloud.com/ascendmlir/ascend-mlir-builder:aarch64-ubuntu22.04-llvm21`
- `--device-id`：`7`
- `--jobs`：`6`
- `--remote-dir`：建议显式传 `/data/{username}/Codex-Ascend-MLIR-current`，或一次性设置 `ASCEND_MLIR_CI_REMOTE_DIR`
- 增量编译：默认开启，直接在 `--remote-dir` 内复用 `build/` 和 `build-runtime-session-run-only/`
- ccache：默认开启，cache 目录按 `--remote-dir` 派生，例如 `/data/{username}/ccache/Codex-Ascend-MLIR-current`

如果需要覆盖这些默认值，再显式传参：

```shell
scripts/sync-and-submit.sh \
  --remote-dir /data/{username}/Codex-Ascend-MLIR-smoke \
  --image swr.cn-east-2.myhuaweicloud.com/ascendmlir/ascend-mlir-builder:aarch64-ubuntu22.04-llvm21 \
  --case add-broadcast-concat \
  --device-id 7 \
  --jobs 6 \
  --ccache-dir /data/{username}/ccache/my-worktree
```

怀疑增量目录或 ccache 被污染时，使用 `--clean` 做一次干净构建。它会删除该 `--remote-dir` 下的构建目录，并删除该 worktree 对应的 ccache 目录：

```shell
scripts/sync-and-submit.sh \
  --remote-dir /data/{username}/Codex-Ascend-MLIR-current \
  --case relu-broadcast-transpose \
  --clean
```

若需要恢复旧行为，即每个 job 复制源码到独立 job 目录并冷构建，可使用：

```shell
scripts/sync-and-submit.sh \
  --remote-dir /data/{username}/Codex-Ascend-MLIR-current \
  --case relu-broadcast-transpose \
  --no-incremental
```

`--case` 的普通取值来自源码树里的 `examples/<case>/run.sh`，也就是 `examples/` 下带 `run.sh` 的目录名。可以用下面命令查看当前本地可选值：

```shell
scripts/sync-and-submit.sh --list-cases
```

当前还有两个特殊值：

- `microcases`：运行 `examples/real-npu-microcases/prepare.sh` 生成的真机 microcases。
- `real-npu-multikernel`：运行 `examples/real-npu-multikernel/run.sh`，覆盖串行双 kernel 和 fork-join 三 kernel 的真实 NPU DAG 调度路径。

`all` 在 runner 中明确禁用，暂时不要使用。

`--case` 是保留的快捷方式，适合继续跑“example sim gate + 改 manifest 后真机 NPU run”的固定流程；`--cmd` 是通用入口，会覆盖 `--case`，适合全量脚本、临时排查命令或自定义验证流程。

注意：

- `--remote-dir` 路径必须在 `/data/{username}/` 下，不能传 `/data/{username}` 本身。
- 默认同步会替换源码文件，但保留 `build/` 和 `build-runtime-session-run-only/`；传 `--clean` 时才删除这些构建目录。
- ignored 产物不会同步，包括 `build/`、`out/`、生成的 `.npy`、`build_e2e/` 等。
- 远端 image 必须已经存在；脚本不会执行 `docker pull` 或 `docker login`。
- SSH 认证优先使用本机已有 ssh key；如果没有 key，脚本会读取 `ASCEND_MLIR_CI_SSH_PASSWORD`，或从 `examples/real-npu.md` 的密码行读取密码，并通过 `sshpass` 进行非交互登录。
- 如果不想使用 ccache，可加 `--no-ccache`。

### 7.4 开发机远程提交

开发机可以用 `submit.sh` 通过 SSH 触发远端 job。它只触发远端执行，不会同步本地未提交改动。

```shell
scripts/real-npu-ci/submit.sh \
  --image "${SWR_IMAGE}" \
  --remote-source-dir /data/{username}/Codex-Ascend-MLIR-current \
  --ref local-tree \
  --case relu-broadcast-transpose \
  --device-id 7 \
  --jobs 6
```

如果使用 `--repo-url --ref`，本地未 push 的改动不会被验证。

## 8. 查看结果

每个 job 输出到：

```text
/data/{username}/real-npu-jobs/<timestamp>-<ref>-<case>/
  commit.txt
  job-env.txt
  source-head.txt
  source-status.txt
  logs/
    build-project.log
    build-run-only.log
    <case>-sim.log
    <case>-npu.log
    microcases-prepare.log
    microcase-<name>-npu.log
    real-npu-multikernel.log
    plog/
      npu-smi.txt
      plog-errorStr.txt
      plog-files.txt
  out/
```

快速查看最新结果：

```shell
latest="$(cat /data/{username}/real-npu-jobs/latest-job.txt)"
echo "${latest}"

grep -E 'session.backend|session.result|session.validation' \
  "${latest}/logs/relu-broadcast-transpose-sim.log" \
  "${latest}/logs/relu-broadcast-transpose-npu.log"

cat "${latest}/logs/plog/plog-errorStr.txt"
```

通过时应看到：

```text
session.backend=sim
session.result=success
session.validation=pass
session.backend=npu
session.result=success
session.validation=pass
```

多 kernel 调度用例可以这样查看：

```shell
latest="$(cat /data/{username}/real-npu-jobs/latest-job.txt)"

grep -E 'session.backend|session.result|session.validation|planned_task_count|serialized_launch_count' \
  "${latest}/logs/real-npu-multikernel.log"
```

期望看到 `serial-two-kernel` 的 sim/npu 各 2 个 task，以及 `fork-join` 的 sim/npu 各 3 个 task，且每段都有 `session.result=success` 和 `session.validation=pass`。当前 sim summary 会打印 `serialized_launch_count`；真实 NPU summary 不一定打印该 counter，NPU launch 次数以 `device_id=7` 的 `[npu-launch] kernel=` 行数为准。

## 9. 失败排查

### 9.1 SWR namespace 不存在

症状：

```text
Image organization does not exist
Failed to find namespace in <name>
```

处理：确认 `SWR_NAMESPACE` 已在 SWR 控制台创建。本项目当前使用 `ascendmlir`。

### 9.2 SWR 拒绝 manifest

症状：

```text
Invalid image, fail to parse 'manifest.json'
```

处理：使用本目录的 `build-aarch64-image.sh` 重新构建。它默认关闭 `--provenance` 和 `--sbom`，并通过 `--load` 生成 SWR 可接受的单架构 image。不要直接 push 带 BuildKit attestation 的 manifest list。

### 9.3 push 网络中断

症状：

```text
failed commit on ref ... EOF
```

处理：直接重试同一个 `docker push "${SWR_IMAGE}"`。已经上传成功的 layer 会复用。

### 9.4 容器内 `rtSetDevice failed: rc=107001`

原因通常是容器权限不足。当前 real-NPU host 只映射 `/dev/davinci*` 不够，需要 `--privileged`。`docker-run.sh` 已默认加入：

```text
--privileged
```

若仍失败，检查：

```shell
ls -l /dev/davinci7 /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
source /usr/local/Ascend/driver/bin/setenv.bash
source /data/{username}/Ascend/latest/set_env.sh
export ASCEND_DEVICE_ID=7
```

### 9.5 真机阶段缺 simulator 动态库

症状：

```text
libnpu_drv_camodel.so: cannot open shared object file
```

原因是误用完整构建的 `runtime-session` 去跑真机。真机阶段必须使用：

```text
ASCEND_RUNTIME_SESSION_RUN_ONLY=ON
```

`run-real-npu-job.sh` 已在 sim gate 后单独构建 `build-runtime-session-run-only/bin/runtime-session`，并用它执行 NPU manifest。

### 9.6 job-env 里的 `npu-smi` 报 `libc_sec.so`

`job-env.txt` 里可能看到：

```text
npu-smi: error while loading shared libraries: libc_sec.so
```

这是采集 job-env 时 driver env 尚未完全 source 的早期输出；如果后续 NPU log 中 `session.result=success`，它不是本次 job 失败原因。真正失败以 `<case>-npu.log` 和 `plog-errorStr.txt` 为准。

## 10. 维护约定

- 更新 LLVM/MLIR 版本时，优先改 `versions.env`，再重新 build image。
- PyAsc 跟随被验证的源码 ref，不作为固定 artifact bake 到通用 image。
- CANN 和 driver 保持 host 挂载，不打进 image。
- 真机执行阶段使用 run-only `runtime-session`，避免进程启动时依赖 CANN compiler/simulator 库。
- 不要为每个验证 job 重建 image；按依赖栈变更频率预构建。
- 临时 SWR token 使用后执行 `docker logout "${SWR_REGISTRY}"`。
