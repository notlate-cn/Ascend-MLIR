# Runtime README

## 目标

`lib/Runtime` 是仓库内唯一运行时中心，负责：

- AscendC kernel 编译
- CPU 仿真执行与 profiling
- NPU 执行路径接线
- task-graph 执行
- 全局调度与后续资源化调度演进

`runtime-session` 是唯一通用 runtime CLI，负责：

- 统一 compile/run 入口
- 统一 summary/profile 输出
- 保持 CLI `main.cpp` 只承担参数解析、摘要打印和执行编排

## 目录分层

### `Artifact/`

负责 runtime artifact 的加载、编译、manifest 和 metadata 管理。

- `ArtifactCompiler` 是统一编译入口
- `mix` 走 `MixDirectBackend`
- `vec/cube` 走 `VecCubeArtifactBackend`

### `Execution/`

负责运行时执行主链路。

- `RuntimeFrontendCore`
- `ExecutionSession`
- `GlobalScheduler`
- `ResourceScheduler`
- `SimBackend`
- `NpuBackend`
- `NativeExecutionRunner`

### `Mix/`

负责 runtime-native mix 编译与 tiling。

- mix ABI 抽取
- direct tiling artifact 生成
- matmul 显式语义路径
- dual-backend matmul tiling

### `Profile/`

负责 profile trace、retained summary 和结果归一化。

- CLI 和 C API 使用同一套 profile/result contract

### `Support/`

负责 runtime 公共支撑代码。

- 文件、环境、路径和通用辅助逻辑统一收口

## 主执行链

### 外部入口

`runtime-session` 和 C API 负责外部接口解析，并调用共享的 frontend core。

### `RuntimeFrontendCore`

负责统一前端 compile/run 语义。

- compile request assembly
- run preparation
- normalized run result / summary interpretation

### `ArtifactCompiler`

负责按 `KernelKind` 分发 runtime-native 编译后端。

- `mix -> MixDirectBackend`
- `vec/cube -> VecCubeArtifactBackend`

### `ExecutionSession`

`ExecutionSession` 是单次运行的 facade。

- 提供 `submit()`、`plan()`、`run()`
- 串行模式下保留本地 ready-queue 执行
- 并发模式下通过 `GlobalScheduler` 执行 task graph
- 统一管理单次运行的工作目录、task output 绑定物化和 session 级 profile 汇总

### `ExecutionBackend`

`ExecutionBackend` 是具体 backend 执行入口，目前包含：

- `Simulation`
- `NPU`

这两条路径共用 capability contract 和 execution runner seam。

### `ProfileTrace` 与 retained session summary

统一承载执行结果、profile artifact 和 runtime 统计输出。

- CLI 与 C API 使用同一结果契约

## 调度与执行模型

### `TaskGraph`

`TaskGraph` 负责 runtime task DAG 的基本合法性和拓扑顺序。

- 检查重复 task id
- 检查未知依赖
- 检查环
- 为执行与调度提供稳定的 topological order

### `ExecutionSession`

`ExecutionSession` 已不再只是本地 ready-queue 执行器。

- 串行路径仍保留
- 并发路径已接到 `GlobalScheduler`
- task 间数据流统一通过工作目录中的物化输出文件衔接

### `GlobalScheduler`

`GlobalScheduler` 是进程内全局 DAG 调度中心。

- 完整持有提交进来的 DAG
- root task admission
- dependents 推进
- task fail 后取消 pending tasks
- session release 时回收 reservation 和调度状态

### `ResourceScheduler`

`ResourceScheduler` 是第一版全局资源仲裁器，目前只覆盖基础资源约束。

- `workspaceBytes`
- `requiresSerializedLaunch`
- `exclusiveDeviceAccess`
- `stream` 级资源占用
- `backendKind`

当前 admission 规则是：只有依赖满足且资源满足时，任务才会进入可执行状态。

- Stream-level resource accounting is now part of the baseline resource model.
- Current policy remains backend-agnostic: stream consumption is expressed through task resource requirements, not through per-backend scheduling policy branches.

### `BackendCapabilities`

backend/driver 通过 `BackendCapabilities` 显式声明调度能力。

- `supportsConcurrentDispatch`
- `supportsConcurrentExecution`
- `requiresSerializedLaunch`
- `maxConcurrentTasks`

当前主执行链已经改为 capability 驱动，但仍保留少量按 `backendKind` 提供默认 capability 的兜底路径。

## Simulation 与 NPU 路径

### `SimBackend`

`SimBackend` 已接入全局调度链，但它的实际执行约束需要单独理解。

- 并发调度与并发取任务路径已接入
- simulator 实际执行仍受受控串行边界约束
- `run_runtime.sh` 与 example pipelines 在 xvm 上稳定通过

这意味着 simulation 路径已经参与全局调度，但并不等价于 simulator 已具备真正的 task-level 并发执行。

### `NpuBackend`

`NpuBackend` 已接入 `GlobalScheduler` 和第一版资源化调度。

- bare NPU 按默认 capability 参与并发路径
- driver-backed NPU 尊重 driver 自身 capability
- `maxConcurrentTasks == 1` 时仍走全局调度路径，但退化为单任务/独占设备语义
- 真机仍待 real-device validation

## Mix Runtime-Native 路径

### matmul 显式语义契约

`abi_matmul_*` 是稳定的 matmul 显式语义契约。

- 上游 pass 显式标注
- `MixAbiExtractor` 读取
- `MixTilingGenerator` 优先消费
- 不再只靠 shape guessing

### 上游语义来源

`AnnotateMixMatmulSemanticsPass` 是上游稳定语义来源，目前覆盖保守 matmul family 子集。

- `matmul`
- `matmul_transpose_a/b`
- `batch_matmul`
- `batch_matmul_transpose_a/b`

### dual-backend matmul tiling

`MatmulTilingDispatcher` 提供 mix matmul 的 dual-backend 框架。

- `MatmulApiTilingBackend`
- `NativeMatmulTilingBackend`
- `auto/api/native` 分发

### `NativeMatmulPlanner`

`NativeMatmulPlanner` 是 runtime 自主 matmul planning 模块，目前重点在稳定 correctness。

- `traverse`
- `plannedBlockDim`
- `splitKEnabled`
- `tileM/N/K`

当前 matmul backend 状态：

- `2D matmul -> native planner + api materializer`
- `batch matmul` 保守子集 -> `api-backed tiling`
- 非 matmul mix op 仍保留 fallback 路径

## 当前状态

### runtime 主架构

runtime 已完成从单 session 执行器到全局调度雏形的跃迁。

- `GlobalScheduler` 已真实接管并发 sim/NPU mock 路径
- resource admission 已进入主执行链

### CPU 仿真验证

xvm 是当前权威开发验证环境。当前 CPU 仿真基线包括：

- `bash test/tools/runtime/run_runtime.sh` 通过
- `bash test/tools/examples/example_pipelines.sh` 通过
- 6 条 example pipeline 全通过

### NPU 状态

NPU 代码路径已接通。

- scheduler contract 已落地
- 当前主要缺口只剩 real-device validation

### 原始 runtime 任务完成度

原始 runtime 任务已经不再是“架构未成”的问题。

- AscendC kernel 编译：完成
- CPU 仿真执行与 profiling：完成
- NPU 执行路径接线：代码完成，待真机验证

## 后续演进

### `NPU real-device validation`

这是当前最高优先级缺口。

- 在有硬件时完成实机多任务执行验证
- 核对 summary、profile 和失败模式

### 更合理的资源化调度

当前 `ResourceScheduler` 仍是 first-version，后续演进重点包括：

- per-backend slot policy
- stream-level scheduling
- workspace / serialized launch 进一步细化

### cross-session fairness / quota

`GlobalScheduler` 现在已有 first-version cross-session fairness baseline。

- 当前使用 session round-robin admission baseline
- fairness 目前只作用于 cross-session admission 顺序
- session 内部 task 顺序仍保持稳定，不做额外 policy 分叉
- quota baseline 已支持 session admission quota
- priority baseline 已支持 static session priority
- 默认 policy 已通过内部 `GlobalSchedulerPolicy` 显式收口
- 当前配置面仍然是 runtime-internal / test-oriented，不是用户 CLI 配置接口

### runtime-native mix tiling

matmul 方向继续增强，但保持通用性和兼容性。

- 扩 `NativeMatmulPlanner` policy surface
- 继续削弱 `api materializer` 依赖
- 扩 `batch matmul` 验证覆盖
- 最后再考虑 broader non-matmul generalization

### execution/profile contract polish

执行与 profile 契约已经统一，但仍可继续打磨。

- 更清晰的 retained summary
- 更一致的 backend/runtime observability

## 验证基线

### 环境

`xvm` 是当前权威开发验证环境，承担：

- runtime focused 验证
- example pipelines CPU 仿真回归
- mix/runtime-native 验证

### 权威回归脚本

修改 runtime 内部实现时，以下脚本应保持绿色：

- `test/tools/runtime/run_runtime.sh`
- `test/tools/runtime/run_simbackend_examples.sh`
- `test/tools/examples/example_pipelines.sh`
