## 9. Pass 流水线参考与 Runtime 对接规范

本节作为开发参考补充，覆盖两个主题：

1. **Pass 流水线参考**：记录当前原型路径的 Pass 序列，以及目标 V2 自动化流水线的规划形态，供开发者在迁移过程中对照参考。
2. **Runtime 对接规范**：定义编译器产物与外部 Runtime 框架的集成协议，包括接口约定、动态 shape 支持、多 kernel DAG 调度和非 C++ 框架对接方式。

---

### 9.1 当前原型流水线（V1 路径）

> **工具说明**：本节命令行中出现的 `afir-opt` 是原型阶段的 driver 工具，功能等价于 MLIR 社区的 `mlir-opt`——承载当前所有 Pass 的注册与执行入口。它随 AFIR Dialect 一同存在于原型期代码库中。V2 规范（见 V2-1.4.2）不依赖 AFIR Dialect；V2 各层 Pass 全部完成后，统一 driver 将替换为 `ascend-mlir-opt`（见 9.2.1 节）。在此之前，开发者可将本节的 `afir-opt` 命令理解为"在当前工具链下的等价调用"。

当前原型阶段，Layers 1–3（Normalize / Kernelize / Schedule）尚未实现为自动化 Pass，由手写 Transform 脚本和人工挑选的融合策略代替。完整 Pass 序列如下。

#### 9.1.1 标准向量算子流水线

适用于：broadcast-add-reduce、relu-broadcast-transpose、add-broadcast-concat 等纯向量 kernel。

```
# 阶段 1：融合（由 linalg 社区 Pass 完成）
afir-opt --linalg-fuse-elementwise-ops \
         INPUT.mlir -o step1_fused.mlir

# 阶段 2：Tiling（由手写 Transform 脚本驱动）
afir-opt --transform-interpreter \
         --canonicalize --cse \
         step2_transform.mlir -o step2_tiled.mlir

# 阶段 3：Bufferize（Layer 4 前置，社区 Pass）
afir-opt '--one-shot-bufferize=bufferize-function-boundaries=true \
          allow-return-allocs-from-loops=true \
          function-boundary-type-conversion=identity-layout-map' \
         --cse \
         step2_tiled.mlir -o step3_bufferized.mlir

# 阶段 4：Buffer Placement（Layer 4 实现）
afir-opt --ascendc-buffer-placement \
         step3_bufferized.mlir -o step4_buffer_placement.mlir

# 阶段 5：Compute Lowering（Layer 5 实现）
afir-opt --linalg-to-ascendc \
         --canonicalize --cse \
         step4_buffer_placement.mlir -o step5_ascendc.mlir

# 阶段 6：多核调度（Layer 5 实现）
afir-opt --ascendc-parallelize \
         --canonicalize --cse \
         step5_ascendc.mlir -o step6_parallelize.mlir

# 阶段 7：Emit 前处理（Layer 5 实现）
afir-opt --ascendc-prepare-for-emit \
         --canonicalize --cse \
         step6_parallelize.mlir -o step7_kernel.mlir

# 阶段 7b：规范化 CANN Signature（Layer 5 实现）
afir-opt --canonicalize-cann-signature \
         step7_kernel.mlir -o step7_cann.mlir

# 阶段 8：Codegen（Layer 5 实现）
afir-translate -mlir-to-cann \
               step7_cann.mlir -o step8_kernel.cpp
```

#### 9.1.2 混合 Cube+Vector 流水线扩展

适用于：matmul-add-leakyrelu、gemm 系列 kernel。在阶段 3 增加以下两个标注 Pass：

```
afir-opt '--one-shot-bufferize=...' \
         --annotate-ascendc-kernel-kind \
         --annotate-mix-matmul-semantics \
         --cse \
         step2_tiled.mlir -o step3_bufferized.mlir
```

阶段 10 改用 `mix-compiler` 驱动 bisheng 编译，而非 `runtime-session --kernel`：

```
mix-compiler \
  --kernel step8_kernel.cpp \
  --cann-mlir step7_cann.mlir \
  --npy-dir DATA_DIR/npy \
  --output ARTIFACT_DIR \
  --soc Ascend910B1
```

#### 9.1.3 Gather 融合流水线扩展

适用于：gather-elementwise-fusion 示例。在阶段 1 前增加结构化标记：

```
afir-opt --mark-structured-ops \
         --fuse-gather-elementwise \
         INPUT.mlir -o step1_gather_fused.mlir
```

其余阶段与标准向量流水线相同。

#### 9.1.4 原型路径的手工介入点

| 手工介入点 | 说明 | V2 目标替换 |
|---|---|---|
| 手写 `step2_transform.mlir` | 手动指定 tiling 尺寸和循环结构 | Layer 3 `ScheduleSearch` 自动搜索 |
| 手写 `tiling_space.json` | 手动维护 tiling 参数空间描述 | 编译器自动生成（见 6.6.7 节） |
| 手选融合策略 | 手动组合 `--linalg-fuse-elementwise-ops` 和 `--mark-structured-ops` | Layer 2 `FusionCandidateAnalyzer` 自动分析 |
| 手动标注 kernel kind | `--annotate-ascendc-kernel-kind` 需人工判断 | Layer 2 `OpRoleClassifier` 自动分类 |

---

### 9.2 目标 V2 自动化流水线

V2 完成后，Layers 1–3 替换为自动化 Pass，手工介入点全部消除。目标 Pass 序列如下。

> **命名约定说明**：本节列出的 Pass 名称（如 `--ascend-normalize`、`--ascend-kernelize`）是**目标命名约定**，作为 V2 实现阶段的命名指导。每个 Pass 内部由 V2-1.2 节最小接口表中的核心类组合实现（例如 `--ascend-kernelize` 内部展开为 `DependencyAnalyzer` → `StructuralMarker` → `OpRoleClassifier` → `FusionCandidateAnalyzer` → `CandidateMergeAnalyzer` → `KernelPatternBuilder` → `KernelPartitioner`）。实现时**以 V2-1.2 的类名为权威**；如需调整 Pass 边界（如把 `--ascend-kernelize` 拆为多个细粒度 Pass），需同步更新本节命名表，但内部类边界不变。

#### 9.2.1 统一入口命令

```
ascend-mlir-opt \
  # Layer 1：Normalize
  --ascend-normalize \
  # Layer 2：Kernelize
  --ascend-kernelize \
  # Layer 3：Schedule（含 structured lowering，输出已含 scf loop 骨架）
  --ascend-schedule \
  # Layer 4：Realize
  --one-shot-bufferize='bufferize-function-boundaries=true \
    allow-return-allocs-from-loops=true \
    function-boundary-type-conversion=identity-layout-map' \
  --ascend-buffer-placement \
  --ascend-movement-planning \
  # Layer 5：Translate
  --ascend-compute-lower \
  --ascend-parallelize \
  --ascend-prepare-for-emit \
  --ascend-canonicalize-cann-signature \
  INPUT.mlir -o KERNEL_CANN.mlir

ascend-mlir-translate -mlir-to-cann \
  KERNEL_CANN.mlir -o KERNEL.cpp
```

#### 9.2.2 与原型路径的对应关系

| V2 Pass | 等价原型操作 | 所属层 |
|---|---|---|
| `--ascend-normalize` | 方言白名单验证、shape 符号化、gather 规范化 | Layer 1 |
| `--ascend-kernelize` | 手写融合脚本 + 手动标注 kernel kind | Layer 2 |
| `--ascend-schedule` | 手写 `step2_transform.mlir` | Layer 3 |
| `--one-shot-bufferize` | 不变，社区 Pass | Layer 4 前置 |
| `--ascend-buffer-placement` | `--ascendc-buffer-placement` | Layer 4 |
| `--ascend-movement-planning` | 隐含在 bufferize 之后，当前无独立 Pass | Layer 4 |
| `--ascend-compute-lower` | `--linalg-to-ascendc` | Layer 5 |
| `--ascend-parallelize` | `--ascendc-parallelize` | Layer 5 |
| `--ascend-prepare-for-emit` | `--ascendc-prepare-for-emit` | Layer 5 |
| `--ascend-canonicalize-cann-signature` | `--canonicalize-cann-signature` | Layer 5 |

#### 9.2.3 自动生成产物清单

V2 完成后，编译器在一次调用中自动输出以下产物，无需人工维护：

| 产物文件 | 内容 | 当前状态 |
|---|---|---|
| `<kernel>.cpp` | AscendC C++ kernel 源码 | 已实现 |
| `<kernel>_cann.mlir` | CANN 标准签名 kernel MLIR | 已实现 |
| `tiling_space.json` | tiling 参数空间描述（v2.0 规范格式，见 6.6.7 节） | 待实现 |
| `runtime_manifest.json` | Runtime 调度清单，含 kernelGraph DAG（见 6.6.6 节） | 待实现 |
| `<KernelName>_get_tiling.so` | C ABI 动态库（见 9.4.3 节） | 待实现 |

---

### 9.3 编译产物与消费方

#### 9.3.1 产物总览

```
编译器输出目录/
├── <kernel>.cpp               ← bisheng 编译输入
├── <kernel>_cann.mlir         ← 元数据来源（ABI、buffer 顺序等）
├── tiling_space.json          ← Autotuner / Runtime 参数空间
├── runtime_manifest.json      ← Runtime 调度总入口
└── <KernelName>_get_tiling.so ← C ABI 动态库（非 C++ Runtime 对接）

bisheng 编译后追加：
├── <kernel>.o                 ← device 侧目标文件
└── <kernel>.bin               ← device 侧 ELF，可直接加载
```

#### 9.3.2 各产物消费方

| 产物 | 主要消费方 | 消费场景 |
|---|---|---|
| `<kernel>.cpp` / `<kernel>.bin` | CANN Runtime / 自定义 device 侧执行引擎 | device 侧 kernel 执行 |
| `<kernel>_cann.mlir` | RuntimeMix、测试框架 | ABI 解析、tiling 参数填充 |
| `tiling_space.json` | Level-2 Autotuner、Runtime 调度框架 | tiling 参数搜索、运行时参数查询 |
| `runtime_manifest.json` | C++ Runtime、外部 Runtime 框架 | 全局调度，多 kernel DAG 执行 |
| `<KernelName>_get_tiling.so` | Python / Go / Rust 推理框架 | 非 C++ 语言跨语言调用 tiling 查询 |

---

### 9.4 外部 Runtime 框架对接规范

本节定义编译器产物与外部 Runtime 框架的集成最小协议。"外部 Runtime 框架"包括：PyTorch、TensorFlow、MindSpore、自研推理引擎等，以及非 C++ 语言实现的调度框架。

#### 9.4.1 接入最低要求

外部 Runtime 框架接入本编译器产物，必须满足以下最低要求：

| 要求 | 说明 |
|---|---|
| 能加载 `.bin` ELF | 调用 CANN `AscendCL` 或等效接口执行 device 侧 kernel |
| 能读取并解析 `runtime_manifest.json` | 获取 kernel 名称、ABI、tiling 参数入口、workspace 大小、DAG 边 |
| 能分配 workspace buffer | 按 `GetWorkspaceSize` 或 manifest 中 `workspaceSizeExpr` 计算所需字节，在 device 侧分配 |
| 能按顺序（或 DAG 拓扑序）触发 kernel 执行 | 单 kernel 按顺序，多 kernel 按 `kernelGraph` 的拓扑序调度 |
| 能填充 tiling 参数结构体并传入 kernel | 通过 C ABI（见 9.4.3 节）或直接解析 `tiling_space.json` 填充 |

外部 Runtime **不需要**：

- 理解 MLIR IR 格式
- 依赖 CANN 编译器内部实现
- 重新实现 tiling 算法（tiling 计算由编译器生成的 C ABI 函数完成）

#### 9.4.2 静态 Shape 对接流程

静态 shape（运行前形状固定）时，对接步骤最简：

```
1. 读取 runtime_manifest.json
   → 获取 kernelName、ABI 字段（inputs/outputs 顺序与类型）

2. 调用 GetTilingSize() → 获取 tiling 结构体字节数

3. 分配 tiling buffer（host 侧）

4. 调用 GetTiling(shape_args, shape_count, tiling_out)
   → 填充 tiling 结构体

5. 调用 GetWorkspaceSize(shape_args, shape_count)
   → 在 device 侧分配 workspace memref<ui8>

6. 调用 GetBlockDim(shape_args, shape_count)
   → 设置 block_dim（AICore 并行数）

7. 调用 AscendCL 执行 kernel
   参数顺序：inputs... outputs... workspace tiling_ptr
```

#### 9.4.3 C ABI 接口规范

编译器为每个 kernel 生成以下四个 C ABI 函数，以动态库（`.so`）形式导出，供任意支持 FFI 的语言调用：

```c
extern "C" {
  /**
   * 返回 tiling 结构体的字节大小。
   * 结果为编译期常量，与 shape 无关。
   */
  int32_t <KernelName>_GetTilingSize(void);

  /**
   * 根据运行时 shape 参数计算 tiling，填充到 tiling_out 指向的缓冲区。
   * shape_args: 按 runtime_manifest.json 中 shapeArgOrder 列出的维度值数组
   * shape_count: shape_args 的元素个数
   * tiling_out: 调用方分配、大小 >= GetTilingSize() 字节的缓冲区
   * 返回 0 表示成功，负数表示错误码
   */
  int32_t <KernelName>_GetTiling(
      const int64_t* shape_args, int32_t shape_count, void* tiling_out);

  /**
   * 根据运行时 shape 参数返回所需的 AICore 并行数（block_dim）。
   */
  int64_t <KernelName>_GetBlockDim(
      const int64_t* shape_args, int32_t shape_count);

  /**
   * 根据运行时 shape 参数返回所需的 workspace 字节数。
   * 返回 0 表示该 kernel 不需要 workspace。
   */
  int64_t <KernelName>_GetWorkspaceSize(
      const int64_t* shape_args, int32_t shape_count);
}
```

**Python FFI 使用示例（ctypes）：**

```python
import ctypes, numpy as np

lib = ctypes.CDLL("./matmul_add_leakyrelu_get_tiling.so")

lib.matmul_add_leakyrelu_GetTilingSize.restype  = ctypes.c_int32
lib.matmul_add_leakyrelu_GetTiling.restype      = ctypes.c_int32
lib.matmul_add_leakyrelu_GetTiling.argtypes     = [
    ctypes.POINTER(ctypes.c_int64), ctypes.c_int32, ctypes.c_void_p
]
lib.matmul_add_leakyrelu_GetBlockDim.restype    = ctypes.c_int64
lib.matmul_add_leakyrelu_GetWorkspaceSize.restype = ctypes.c_int64

# shape_args 顺序见 runtime_manifest.json 的 shapeArgOrder 字段
shape_args = np.array([128, 256, 128], dtype=np.int64)
n_shapes   = len(shape_args)
shape_ptr  = shape_args.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

tiling_size = lib.matmul_add_leakyrelu_GetTilingSize()
tiling_buf  = (ctypes.c_uint8 * tiling_size)()

ret = lib.matmul_add_leakyrelu_GetTiling(shape_ptr, n_shapes, tiling_buf)
assert ret == 0, f"GetTiling failed: {ret}"

block_dim      = lib.matmul_add_leakyrelu_GetBlockDim(shape_ptr, n_shapes)
workspace_size = lib.matmul_add_leakyrelu_GetWorkspaceSize(shape_ptr, n_shapes)
```

#### 9.4.4 动态 Shape 对接

动态 shape 下，每次推理调用前 shape 才确定。对接步骤与静态形相同，差异在于：

- `GetTiling`、`GetBlockDim`、`GetWorkspaceSize` 在每次推理时以当前 shape 为参数调用
- 编译器在 `runtime_manifest.json` 的 `decision_guards` 字段记录 shape 约束（guard 条件），Runtime 无需自行实现分支选择，由 `GetTiling` 内部完成
- 对于多路 guard（shape bucket 分发），`GetTiling` 内部根据 shape 参数选择对应的 `ScheduleDecision`，外部 Runtime 只需调用一次 `GetTiling`，无需感知内部分支

```
动态推理调用流程（每次 forward）：

shape_args ← 本次输入的实际维度
GetTiling(shape_args, ..., tiling_out)  ← 内部自动选 decision
GetBlockDim(shape_args, ...)            ← 对应 block 数
GetWorkspaceSize(shape_args, ...)       ← 对应 workspace 大小
分配/复用 device workspace
执行 kernel
```

**shape_args 顺序约定**：`runtime_manifest.json` 中的 `shapeArgOrder` 字段（定义见 V2-6.6.2 节）显式列出每个位置对应哪个符号维度，Runtime 框架必须按此顺序传入，不得自行推断顺序。

#### 9.4.5 多 Kernel DAG 调度

多 kernel 场景（融合失败、多段 kernel）下，`runtime_manifest.json` 的 `kernelGraph` 字段描述内核间的数据依赖 DAG：

```json
"kernelGraph": {
  "nodes": [
    { "kernelName": "k1_matmul",  "blockDimExpr": "ceil(M/TB_M)" },
    { "kernelName": "k2_softmax", "blockDimExpr": "B" }
  ],
  "edges": [
    {
      "from": "k1_matmul",
      "to":   "k2_softmax",
      "carriedBuffers": ["attn_score"]
    }
  ]
}
```

外部 Runtime 框架的调度要求：

| 要求 | 说明 |
|---|---|
| 拓扑序执行 | 按 `edges` 的依赖关系确定执行顺序；无依赖关系的节点可并发执行 |
| carriedBuffers 生命周期管理 | 边上的 `carriedBuffers` 表示跨 kernel 传递的中间 buffer，Runtime 负责分配其生命周期，保证生产者执行完毕前消费者不启动 |
| 无需理解 buffer 语义 | `carriedBuffers` 仅作生命周期依赖标记，Runtime 不需要解析其数据格式 |
| 并发调度可选 | Runtime 可退化为串行拓扑序执行，不强制并发；并发执行可提升吞吐，但需正确处理 AscendCL stream 同步 |

当前 `lib/Runtime` 已支持多 kernel 并发调度（基于 AscendCL stream event），外部 Runtime 可直接复用，也可实现自己的 DAG 调度器。

**同步原语约定（重要）**：

编译器输出的 `kernelGraph.edges` 只表达**逻辑依赖关系**（"消费者必须在生产者完成后才能启动"），**不指定具体同步原语**。外部 Runtime 框架可自由选择以下任一实现方式：

| 同步方式 | 适用场景 | 说明 |
|---|---|---|
| AscendCL stream event（推荐） | 多 stream 并发执行 | 通过 `aclrtCreateEvent` / `aclrtStreamWaitEvent` 实现跨 stream 依赖；当前 `lib/Runtime` 默认采用此方式 |
| 单 stream 串行 | 单 stream / 退化场景 | 同一 stream 内 kernel 按提交顺序天然串行，无需显式同步原语；适合简单场景或调试 |
| AscendCL notify | 设备间同步（多卡） | 跨 device 的依赖通过 notify 机制；本编译器单 device 场景下不强制使用 |
| 自定义 semaphore / barrier | 框架定制场景 | PyTorch / 自研推理引擎的内部同步机制 |

**编译器对同步原语的零依赖**：

- `kernelGraph` 不携带 stream id、event id、notify id 等具体同步对象引用
- 编译器输出的 `.bin` 不包含任何同步原语调用代码（同步由 host 侧 Runtime 注入）
- 对于退化为单 stream 串行执行的 Runtime，`kernelGraph.edges` 仅作为执行顺序的合法性参考，不强制对应实际同步操作

#### 9.4.6 Workspace 管理约定

- `GetWorkspaceSize` 返回该 kernel 单次执行所需的 workspace 字节数
- Workspace buffer 类型为 `memref<ui8>`，对应 kernel ABI 签名中的倒数第二个参数（tiling 参数之前）
- 多次推理可复用同一 workspace buffer（只要大小满足当前 shape 的需求）
- 多 kernel 并发执行时，每个 kernel 的 workspace 必须**独立分配**，不得共享同一 workspace buffer 的不同偏移（除非编译器在 `workspaceSizeExpr` 中已做打包规划）

---

### 9.5 对接检查清单

以下清单供外部 Runtime 框架接入时自查：

#### 9.5.1 静态 Shape 场景

- [ ] 读取 `runtime_manifest.json`，验证 `schema_version`
- [ ] 按 `abi.inputs` / `abi.outputs` 字段顺序绑定 tensor buffer
- [ ] 调用 `GetTilingSize()` 确认 tiling 结构体大小
- [ ] 调用 `GetTiling(shape_args, ...)` 填充 tiling
- [ ] 按 `GetWorkspaceSize()` 分配 device workspace
- [ ] 按 `GetBlockDim()` 设置 block_dim
- [ ] 按 CANN ABI 顺序：inputs… → outputs… → workspace → tiling_ptr 传入 kernel
- [ ] 验证输出结果与 golden 吻合

#### 9.5.2 动态 Shape 场景（在静态清单基础上）

- [ ] 确认 `shapeArgOrder` 字段存在且与实际 shape 维度一一对应
- [ ] 每次推理前调用 `GetTiling` / `GetBlockDim` / `GetWorkspaceSize`（不缓存上次结果）
- [ ] 若 workspace 大小随 shape 变化，在 size 增大时重新分配 device buffer

#### 9.5.3 多 Kernel DAG 场景（在静态清单基础上）

- [ ] 读取 `kernelGraph.nodes` 和 `kernelGraph.edges`
- [ ] 按拓扑序（或并发）调度各 kernel
- [ ] 为 `carriedBuffers` 中的每个 buffer 分配独立 device 内存，并在依赖边两端正确插入同步点
- [ ] 验证所有 kernel 执行完毕后输出结果正确

#### 9.5.4 非 C++ 框架（Python / Go / Rust）

- [ ] 通过 FFI 加载 `<KernelName>_get_tiling.so`
- [ ] 绑定 `GetTilingSize`、`GetTiling`、`GetBlockDim`、`GetWorkspaceSize` 四个符号
- [ ] 按 `shapeArgOrder` 构造 `int64[]` shape 参数数组
- [ ] 分配 host 侧 tiling buffer（大小 = `GetTilingSize()` 字节），调用 `GetTiling` 填充
- [ ] 将 tiling buffer 按字节传入 device（通过 AscendCL `aclrtMemcpy` 或等效接口）
