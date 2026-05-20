## 8. Target Hardware Modeling

`Target Hardware Modeling` 不属于五层流程中的任何一层，而是为第三层（Schedule）、第四层（Realize）和第五层（Translate）提供统一硬件查询模型。所有对目标硬件能力的查询均通过此模型完成，不允许各层自行推断或临时扩充 target 能力。

### 8.1 职责边界

**模型负责（目标）：**

- 声明 target 支持的 memory place 集合，及各 place 的容量、对齐和执行单元可见性
- 声明 place 之间合法的搬运路径、路径类型、约束和代价
- 声明 target 支持的 compute / movement / fixpipe intrinsic 及 dtype 约束
- 为 schedule 约束生成、placement 决策、movement 规划、backend lowering 提供统一查询接口

**模型不负责（非目标）：**

- 为具体 value 分配 memory place（第四层 Placement 负责）
- 判断某个 tile 是否应 promote（第四层 PlacementPlanner 负责）
- 生成 copy / DMA / queue / fixpipe op（第四层 Movement 负责）
- 生成最终 backend 指令编码（第五层 Translate 负责）
- 根据单个 kernel 临时修改 target 能力

### 8.2 核心对象与构造流程

```text
CANN platform_config/<socVersion>.ini
              │
              ▼
  CannTargetProfileLoader
              │
              ▼
        TargetProfile
    ├── TargetIdentity          ← SoC 版本、NpuArch、backend 版本
    ├── TargetHardwareInfo      ← core 数量、原生 tile shape、能力标志
    ├── TargetMemoryModel       ← memory place、容量、路径图
    ├── TargetIntrinsicModel    ← compute / movement intrinsic 表
    └── TargetCostModel         ← 带宽率、路径代价
```

| 子对象                 | 内容                                                         | 主要消费层                                             |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| `TargetIdentity`       | `SoC_version`、`Short_SoC_version`、`NpuArch`、`AIC_version`、Cube / Vector backend 版本 | 全流程 target 选择与 diagnostics                       |
| `TargetHardwareInfo`   | AI Core / Cube / Vector core 数、core 组合方式、原生 tile shape、BF16 / fixpipe 等能力标志 | 第三层调度搜索、第五层 backend lowering                |
| `TargetMemoryModel`    | memory place、容量、对齐、可见性、路径图、路径类型与约束     | 第三层 memory constraints、第四层 placement / movement |
| `TargetIntrinsicModel` | data movement / transpose / fixpipe / vector / cube intrinsic 及 dtype 支持表 | 第四层 movement 规划、第五层 intrinsic lowering        |
| `TargetCostModel`      | memory 带宽率、路径启动代价与单位字节代价                    | 第三层候选排序、第四层 movement 排序                   |

**核心构造类：**

| 类                            | 职责                                                 | 输入                        | 输出                   |
| ----------------------------- | ---------------------------------------------------- | --------------------------- | ---------------------- |
| `CannTargetProfileLoader`     | 加载并构造完整 target 描述                           | `CANN_ROOT`、`socVersion`   | `TargetProfile`        |
| `TargetIntrinsicModelBuilder` | 从 intrinsic dtype map 构建 intrinsic 能力模型       | `IniFile`、`TargetIdentity` | `TargetIntrinsicModel` |
| `TargetMemoryModelBuilder`    | 从 target profile 派生内存层次与路径图               | `TargetProfile`             | `TargetMemoryModel`    |
| `TargetModelVerifier`         | 校验 profile、memory model、intrinsic model 三者闭合 | `TargetProfile`             | `LogicalResult`        |

**`TargetModelVerifier` 最小闭合检查规则：**

| 检查项 | 规则 | 失败行为 |
| ------ | ---- | -------- |
| place 引用完整性 | `pathGraph` 中每条边的 `srcPlace` / `dstPlace` 必须存在于 `memoryPlaces` | fail-fast，报 `TargetProfileMissingField` |
| intrinsic 覆盖完整性 | `pathGraph` 中每条边至少有一个对应 intrinsic（通过 `movementIntrinsicMap` 可达）；必需路径（见 §8.3）缺少 intrinsic 时 fail-fast | fail-fast，报 `TargetPathIntrinsicMissing` |
| 能力标志一致性 | `TargetHardwareInfo.support_fixpipe == true` 当且仅当 `TargetIntrinsicModel` 包含 `Intrinsic_fix_pipe_*` 条目 | fail-fast，报 `TargetProfileMissingField` |
| 容量非零 | 每个非可选 place 的 `CapacityRule` 中静态容量 > 0 | fail-fast，报 `TargetProfileMissingField` |
| 路径约束 dtype 非空 | 每条 `PathEdge` 的 `pathConstraints` 至少包含一个合法 dtype | fail-fast，报 `TargetPathIntrinsicMissing` |

### 8.3 TargetMemoryModel

`TargetMemoryModel` 是路径规划和 placement 决策的基础数据源，描述目标硬件的内存层次结构与合法搬运路径。

**最小字段：**

| 字段              | 类型                                              | 含义                                                 | 填充来源                                      |
| ----------------- | ------------------------------------------------- | ---------------------------------------------------- | --------------------------------------------- |
| `memoryPlaces`    | `SmallVector<MemoryPlace>`                        | target 支持的 memory place 集合                      | `TargetHardwareInfo` + physical memory spec   |
| `capacity`        | `DenseMap<MemoryPlace, CapacityRule>`             | 每个 place 的容量规则                                | physical memory spec                          |
| `alignment`       | `DenseMap<MemoryPlace, AlignmentRule>`            | 地址、stride、tile 对齐约束                          | `TargetHardwareInfo` / `TargetIntrinsicModel` |
| `visibilityRules` | `DenseMap<MemoryPlace, VisibilityRule>`           | place 对 Cube / Vector / DMA 等执行单元的可见性      | `TargetHardwareInfo`                          |
| `pathGraph`       | `DenseMap<MemoryPlace, SmallVector<PathEdge>>`    | place 间有向可达图                                   | `TargetIntrinsicModel` + memory place 映射    |
| `pathKind`        | `DenseMap<PathEdge, PathKind>`                    | 每条边对应的搬运类型                                 | `TargetIntrinsicModel`                        |
| `pathConstraints` | `DenseMap<PathEdge, SmallVector<PathConstraint>>` | dtype、rank、layout、transpose、burst / 2D load 限制 | `TargetIntrinsicModel`                        |
| `pathCostModel`   | `DenseMap<PathEdge, PathCost>`                    | 路径启动代价、单位字节代价、是否可与计算重叠         | `TargetCostModel`                             |

**辅助类型：**

| 类型             | 定义                                                         |
| ---------------- | ------------------------------------------------------------ |
| `PathEdge`       | `{srcPlace, dstPlace, pathVariant}`；同一 `src → dst` 支持多种搬运方式时以不同 `pathVariant` 区分 |
| `CapacityRule`   | 静态容量、可用容量表达式、是否按 execution unit 分区         |
| `AlignmentRule`  | 最小地址对齐、stride 对齐、tile shape 对齐、是否要求 power-of-two |
| `VisibilityRule` | 可访问的执行单元集合、是否可跨 pipeline stage 重用、是否 ABI-visible |
| `PathConstraint` | 允许的 dtype、rank、layout、transpose、burst / 2D load 条件  |
| `PathCost`       | 固定启动代价、单位字节代价、是否可与计算重叠                 |

**`MemoryPlace` 枚举（最小集合）：**

| `MemoryPlace` | 对应硬件          | `memory_space` 编码 | 说明                    |
| ------------- | ----------------- | ------------------- | ----------------------- |
| `GM`          | DDR / HBM         | `0`                 | 全局内存（默认 memory_space，即不带 attribute 时） |
| `A1`          | L1（A 路径）      | `1`                 | Cube A 路径一级片上缓冲 |
| `A2`          | L0A               | `2`                 | Cube A 路径二级片上缓冲 |
| `B1`          | L1（B 路径）      | `3`                 | Cube B 路径一级片上缓冲 |
| `B2`          | L0B               | `4`                 | Cube B 路径二级片上缓冲 |
| `CO1`         | L0C               | `7`                 | Cube 输出中间缓冲       |
| `VECIN`       | UB（Vector 输入） | `9`                 | Vector 输入 place       |
| `VECOUT`      | UB（Vector 输出） | `10`                | Vector 输出 place       |
| `VECCALC`     | UB（Vector 临时） | `11`                | Vector 计算临时 place   |
| `GM_FLAT`     | DDR / HBM         | `22`                | PrepareForEmit 阶段引入的 flat GM 指针编码；仅出现在第五层 ABI 准备阶段，不参与第三、四层 placement 决策（见 V2-6.4 示例） |

**编码约束**：

- 上表是 `memory_space` 整数编码的**唯一权威定义**，V2-5、V2-6、V2-9 中所有 `memref<..., N>` 形式的示例必须按此表取值；后续如需新增 place，必须先在此表追加编码后再在其他文档中引用
- 编码 `0` 是 MLIR 默认 memory_space 含义，等价于"不写 attribute"；`memref<128xf16>` 与 `memref<128xf16, 0>` 在第四、五层视为同一类型
- `GM_FLAT` (22) 不参与 V2-5 `PlacementPlanner` 的候选 place 集合；它由第五层 `KernelSignatureCanonicalizationPass` 在生成 CANN 标准签名时引入，用于 ABI 边界的指针类型表达
- `getMemorySpace(Type)` 接口（见 `LinalgToAscendCUtils.h`）默认返回 `0` 表示 GM，与上表一致

**`PathKind` 枚举（最小集合）：**

| `PathKind`        | 含义                            | 典型 CANN intrinsic                                         |
| ----------------- | ------------------------------- | ----------------------------------------------------------- |
| `DirectCopy`      | 常规 copy / DMA                 | `Intrinsic_data_move_l12l0a`、`Intrinsic_data_move_ub2out`  |
| `Load2D`          | 二维搬运（外部内存 → 片上）     | `Intrinsic_data_move_out2l1`、`Intrinsic_data_move_out2l0a` |
| `Load2DTranspose` | 二维搬运并完成 layout transpose | `Intrinsic_data_move_transpose_l12l0a/b`                    |
| `FixPipe`         | Cube 输出专用 pipe 路径         | `Intrinsic_fix_pipe_l0c2out`、`Intrinsic_fix_pipe_l0c2l1`   |
| `QueueTransfer`   | Cube → Vector 执行单元切换      | 由 `CO1 → VECIN` 路径建模                                   |

**最小路径集合：**

"必需"路径（Required）缺少对应 intrinsic 时 `TargetModelVerifier` fail-fast；"可选"路径（Optional）仅在对应 intrinsic 存在时加入 `pathGraph`。

| 路径                | `PathKind`        | 必需/可选 | 说明                             |
| ------------------- | ----------------- | --------- | -------------------------------- |
| `GM → A1`           | `Load2D`          | 必需      | Cube A 路径一级提升              |
| `GM → B1`           | `Load2D`          | 必需      | Cube B 路径一级提升              |
| `GM → B1.transpose` | `Load2DTranspose` | 可选      | Cube B 路径一级提升并 transpose  |
| `A1 → A2`           | `DirectCopy`      | 必需      | Cube A 路径继续下沉              |
| `B1 → B2`           | `DirectCopy`      | 必需      | Cube B 路径继续下沉              |
| `CO1 → VECIN`       | `QueueTransfer`   | 必需      | Cube 结果交给 Vector             |
| `CO1 → GM.fixpipe`  | `FixPipe`         | 可选      | Cube 输出经 fixpipe 写回全局内存（需 `support_fixpipe=1`） |
| `GM → VECIN`        | `DirectCopy`      | 必需      | Vector 输入直接提升              |
| `VECOUT → GM`       | `DirectCopy`      | 必需      | Vector 结果写回全局内存          |

**查询接口：**

| 接口                            | 语义                                                         |
| ------------------------------- | ------------------------------------------------------------ |
| `isPlaceVisibleTo(place, unit)` | 判断 memory place 是否可被指定执行单元访问                   |
| `getCapacity(place)`            | 返回 place 的容量规则（不扣除当前 kernel 已用容量）          |
| `getAlignment(place)`           | 返回地址、stride、tile 对齐规则                              |
| `findPaths(src, dst)`           | 返回 `src → dst` 的合法路径候选列表                          |
| `getPathConstraints(edge)`      | 返回某条 path edge 的 layout / dtype / rank / transpose 限制 |
| `getPathCost(edge)`             | 返回路径代价摘要，供排序和启发式选择                         |

### 8.4 TargetIntrinsicModel

`TargetIntrinsicModel` 描述 target 支持的 intrinsic 能力，是 backend lowering 选择具体 API 的依据。

**最小字段：**

| 字段                   | 类型                                                | 含义                                    | 填充来源                                                     |
| ---------------------- | --------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------ |
| `intrinsicTable`       | `DenseMap<IntrinsicId, IntrinsicCapability>`        | 全量 intrinsic 及其能力摘要             | CANN `*intrinsicDtypeMap`                                    |
| `unitIntrinsicMap`     | `DenseMap<ExecutionUnit, SmallVector<IntrinsicId>>` | 每类执行单元可用的 intrinsic            | `AICoreintrinsicDtypeMap`、`CUBECoreintrinsicDtypeMap`、`VectorCoreintrinsicDtypeMap` |
| `dtypeSupport`         | `DenseMap<IntrinsicId, SmallVector<DTypePattern>>`  | 每个 intrinsic 支持的数据类型或类型组合 | `Intrinsic_xxx|dtype-list`                                   |
| `movementIntrinsicMap` | `DenseMap<PathKind, SmallVector<IntrinsicId>>`      | 每类搬运路径可用的 intrinsic            | `Intrinsic_data_move_*`、`Intrinsic_fix_pipe_*`              |
| `computeIntrinsicMap`  | `DenseMap<ComputeKind, SmallVector<IntrinsicId>>`   | 每类计算可用的 intrinsic                | `Intrinsic_mmad`、`Intrinsic_vadd` 等                        |

**`intrinsicDtypeMap` 解析规则：**

| CANN 条目                                                    | 建模结果                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `Intrinsic_mmad|...`                                         | 形成 Cube matmul / mma capability，供 matmul lowering 查询   |
| `Intrinsic_vadd`、`Intrinsic_vexp`、`Intrinsic_vtranspose`、`Intrinsic_vgather` | 形成 Vector compute / transform capability                   |
| `Intrinsic_data_move_out2l1`、`Intrinsic_data_move_l12l0a`   | 形成 memory path 候选搬运 intrinsic                          |
| `Intrinsic_data_move_transpose_l12l0a`                       | 形成带 transpose variant 的搬运 intrinsic，对应 `PathKind::Load2DTranspose` |
| `Intrinsic_fix_pipe_l0c2out`、`Intrinsic_fix_pipe_l0c2l1`    | 形成 Cube 输出后处理能力，对应 `PathKind::FixPipe`           |

**建模边界：**

- `TargetIntrinsicModel` 只描述 target 是否支持某类 intrinsic 及其 dtype / variant 约束
- 不描述 memory place 容量、value 生命周期或当前 kernel 是否应使用某条路径
- `TargetMemoryModel.pathGraph` 可引用 `TargetIntrinsicModel` 中的 intrinsic id，但不复制完整 intrinsic 表
- 第五层 backend lowering 再把 `IntrinsicId` 落成 AscendC / CCE / backend-native op，不在 target model 中生成最终指令编码

### 8.5 CANN 配置来源与映射

**已确认的配置文件位置（以 `Ascend910B2` 为例）：**

| 项           | 路径                                                         |
| ------------ | ------------------------------------------------------------ |
| CANN 根路径  | `/home/niu/Ascend/20260323_newest/cann-9.0.0`                |
| SoC 配置目录 | `aarch64-linux/data/platform_config/`                        |
| 代表配置文件 | `Ascend910B2.ini`、`Ascend950PR_9599.ini`、`Ascend310B*.ini` |
| 平台头文件   | `aarch64-linux/include/platform/soc_spec.h`、`platform_info.h` |

**`platform_config/*.ini` → `TargetProfile` 字段映射：**

| CANN section                                      | 关键字段示例                                                 | 目标对象                                                    |
| ------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| `[version]`                                       | `SoC_version`、`Short_SoC_version`、`AIC_version`、`CCEC_CUBE_version`、`CCEC_VECTOR_version`、`NpuArch` | `TargetIdentity`                                            |
| `[SoCInfo]`                                       | `ai_core_cnt`、`cube_core_cnt`、`vector_core_cnt`、`memory_size`、`l2_size`、`core_type_list`、`cube_vector_combine`、`support_bf16` | `TargetHardwareInfo`、`TargetMemoryModel`                   |
| `[AICoreSpec]`                                    | `cube_m_size`、`cube_n_size`、`cube_k_size`、`l0_a_size`、`l0_b_size`、`l0_c_size`、`l1_size`、`ub_size`、`ubblock_size`、`ubbank_size`、`ubbank_num`、`support_fixpipe` | `TargetHardwareInfo`、`TargetMemoryModel`                   |
| `[VectorCoreSpec]`                                | `vec_calc_size`、`ub_size`、`ubblock_size`、`ubbank_size`    | `TargetHardwareInfo`、`TargetMemoryModel`                   |
| `[AICoreMemoryRates]` / `[VectorCoreMemoryRates]` | `ddr_rate`、`l2_rate`、`l1_to_l0_a_rate`、`l1_to_l0_b_rate`、`l1_to_ub_rate`、`l0_c_to_ub_rate`、`ub_to_l2_rate`、`ub_to_ddr_rate`、`ub_to_l1_rate` | `TargetCostModel.pathCostModel`                             |
| `[AICoreintrinsicDtypeMap]`                       | `Intrinsic_mmad`、`Intrinsic_data_move_out2l1`、`Intrinsic_data_move_l12l0a`、`Intrinsic_fix_pipe_l0c2out` | `TargetIntrinsicModel`、`TargetMemoryModel.pathConstraints` |
| `[CUBECoreintrinsicDtypeMap]`                     | `Intrinsic_mmad`                                             | `TargetIntrinsicModel.computeIntrinsicMap`                  |
| `[VectorCoreintrinsicDtypeMap]`                   | `Intrinsic_vadd`、`Intrinsic_vexp`、`Intrinsic_vtranspose`、`Intrinsic_vgather`、`Intrinsic_vreduce` | `TargetIntrinsicModel.computeIntrinsicMap`                  |

**CANN 物理内存 → 编译器逻辑 place 映射：**

| CANN / 硬件字段     | 编译器逻辑 place             | 说明                                                         |
| ------------------- | ---------------------------- | ------------------------------------------------------------ |
| `memory_size` / DDR | `GM`                         | 全局内存                                                     |
| `l2_size`           | `L2`（可选）                 | 跨 core / 全局 cache 能力，不一定作为每个 kernel 的显式 place |
| `l1_size`           | `A1`、`B1`                   | Cube A/B 路径一级片上缓冲的逻辑视图                          |
| `l0_a_size`         | `A2`                         | Cube A 路径二级片上缓冲                                      |
| `l0_b_size`         | `B2`                         | Cube B 路径二级片上缓冲                                      |
| `l0_c_size`         | `CO1`                        | Cube 输出中间 place                                          |
| `ub_size`           | `VECIN`、`VECOUT`、`VECCALC` | Vector 侧输入、输出、计算临时 place                          |

**Ascend910B2 典型配置数值（参考）：**

| section             | 字段                                                | 典型值                       |
| ------------------- | --------------------------------------------------- | ---------------------------- |
| `SoCInfo`           | `ai_core_cnt` / `cube_core_cnt` / `vector_core_cnt` | `24` / `24` / `48`           |
| `SoCInfo`           | `memory_size` / `l2_size`                           | `68 GB` / `192 MB`           |
| `AICoreSpec`        | `l0_a_size` / `l0_b_size` / `l0_c_size`             | `65536` / `65536` / `131072` |
| `AICoreSpec`        | `l1_size` / `ub_size`                               | `524288` / `196608`          |
| `AICoreSpec`        | `support_fixpipe`                                   | `1`                          |
| `AICoreMemoryRates` | `l1_to_l0_a_rate` / `l1_to_l0_b_rate`               | `512` / `256`                |
| `AICoreMemoryRates` | `l0_c_to_ub_rate` / `ub_to_ddr_rate`                | `256` / `64`                 |

### 8.6 构造步骤

`TargetProfile` 的构造顺序如下，步骤间严格顺序依赖：

1. `CannTargetProfileLoader` 根据 `CANN_ROOT` 和 `socVersion` 定位 `platform_config/<socVersion>.ini`
2. 解析 `[version]`，构造 `TargetIdentity`
3. 解析 `[SoCInfo]`、`[AICoreSpec]`、`[VectorCoreSpec]`，构造 `TargetHardwareInfo` 和 physical memory spec
4. 解析 `[AICoreMemoryRates]` / `[VectorCoreMemoryRates]`，构造 `TargetCostModel`
5. 解析 `[AICoreintrinsicDtypeMap]`、`[CUBECoreintrinsicDtypeMap]`、`[VectorCoreintrinsicDtypeMap]`，由 `TargetIntrinsicModelBuilder` 构造 `TargetIntrinsicModel`
6. 组装 `TargetProfile`（含上述四个子对象）
7. 由 `TargetMemoryModelBuilder` 从 `TargetProfile` 派生 `TargetMemoryModel`：
   - 构造 `memoryPlaces`、`capacity`、`alignment`、`visibilityRules`
   - 根据 data movement / fixpipe intrinsic 构造有向 `pathGraph`
   - 为每条 `PathEdge` 填充 `pathKind`、`pathConstraints`、`pathCostModel`
8. 运行 `TargetModelVerifier`，校验三者闭合（profile、memory model、intrinsic model）
9. 输出最终 `TargetProfile`，供全流程只读查询

**构造约束：**

| 场景                                    | 规则                                         |
| --------------------------------------- | -------------------------------------------- |
| 未声明的 memory place                   | 不进入模型，后续阶段不得临时补充             |
| 未声明的搬运路径                        | 不允许后续阶段临时补造 path                  |
| path 缺少对应 intrinsic                 | 不进入 `pathGraph`；若为必需路径则 fail-fast |
| path variant 不满足 dtype / layout 限制 | 不能被 data movement 选择                    |
| 目标 SoC 的 config 缺少必要字段         | fail-fast，不允许后续阶段以默认值推断        |
| 不同 SoC 目标                           | 只替换 `TargetProfile`，pass pipeline 不变   |

### 8.7 使用示例

**`matmul + vector epilogue` 所需的 target model 片段：**

```text
memoryPlaces = [GM, A1, A2, B1, B2, CO1, VECIN, VECOUT, VECCALC]

pathGraph = {
  GM:     [GM → A1, GM → B1, GM → B1.transpose, GM → VECIN],
  A1:     [A1 → A2],
  B1:     [B1 → B2],
  CO1:    [CO1 → VECIN, CO1 → GM.fixpipe],
  VECOUT: [VECOUT → GM]
}

pathKind = {
  GM → A1:             Load2D,
  GM → B1:             Load2D,
  GM → B1.transpose:   Load2DTranspose,
  A1 → A2:             DirectCopy,
  B1 → B2:             DirectCopy,
  CO1 → VECIN:         QueueTransfer,
  CO1 → GM.fixpipe:    FixPipe,
  VECOUT → GM:         DirectCopy
}

intrinsicTable = {
  Intrinsic_data_move_out2l1:           dtypes = [u8, s8, f16, u16, s16, f32, s32, u32],
  Intrinsic_data_move_l12l0a:           dtypes = [u8, s8, f16, u16, s16, f32, s32, u32],
  Intrinsic_data_move_transpose_l12l0b: dtypes = [u8, s8, f16, u16, s16, f32, s32, u32],
  Intrinsic_mmad:                       dtypes = [f16f16f16, f32f16f16, s32s8s8, ...],
  Intrinsic_fix_pipe_l0c2out:           dtypes = [f32, s32, f16]
}
```

**模型能回答的查询：**

| 查询                                     | 结果                                   |
| ---------------------------------------- | -------------------------------------- |
| `findPaths(GM, A2)`                      | `[GM → A1, A1 → A2]`                   |
| `findPaths(CO1, VECIN)`                  | `[CO1 → VECIN]`                        |
| `isPlaceVisibleTo(A2, Vector)`           | `false`                                |
| `isPlaceVisibleTo(VECIN, Vector)`        | `true`                                 |
| `getPathKind(GM → B1.transpose)`         | `Load2DTranspose`                      |
| `queryIntrinsic(GM → B1.transpose, f16)` | `Intrinsic_data_move_transpose_l12l0b` |

**模型不回答（由后续层负责）的问题：**

| 问题                                        | 负责层                           |
| ------------------------------------------- | -------------------------------- |
| `lhs tile` 是否真的放到 `A2`                | 第四层 Placement                 |
| `rhs tile` 是否选择 transpose load          | 第四层 Placement + Data Movement |
| `CO1 → VECIN` 的 queue op 插在哪里          | 第四层 Data Movement             |
| `Intrinsic_mmad` 如何落成 backend-native op | 第五层 backend lowering          |

### 8.8 代码接口参考

```cpp
// 顶层加载入口
class CannTargetProfileLoader {
public:
  FailureOr<TargetProfile> load(StringRef cannRoot, StringRef socVersion,
                                DiagnosticEmitter &diag) const;

private:
  FailureOr<IniFile> loadPlatformConfig(StringRef cannRoot,
                                        StringRef socVersion,
                                        DiagnosticEmitter &diag) const;
  FailureOr<TargetProfile> buildTargetProfile(const IniFile &ini,
                                              DiagnosticEmitter &diag) const;
};

// Intrinsic 能力表构造
class TargetIntrinsicModelBuilder {
public:
  FailureOr<TargetIntrinsicModel> build(const IniFile &ini,
                                        const TargetIdentity &identity,
                                        DiagnosticEmitter &diag) const;

private:
  FailureOr<SmallVector<IntrinsicCapability>>
  parseIntrinsicDtypeMap(const IniFile &ini, StringRef section,
                         ExecutionUnit unit,
                         DiagnosticEmitter &diag) const;

  DenseMap<PathKind, SmallVector<IntrinsicId>>
  buildMovementIntrinsicMap(ArrayRef<IntrinsicCapability> caps) const;

  DenseMap<ComputeKind, SmallVector<IntrinsicId>>
  buildComputeIntrinsicMap(ArrayRef<IntrinsicCapability> caps) const;
};

// 内存模型构造
class TargetMemoryModelBuilder {
public:
  FailureOr<TargetMemoryModel> build(const TargetProfile &profile,
                                     DiagnosticEmitter &diag) const;

private:
  SmallVector<MemoryPlace>
  buildPlaces(const TargetProfile &profile) const;

  DenseMap<MemoryPlace, CapacityRule>
  buildCapacityRules(const TargetProfile &profile) const;

  DenseMap<MemoryPlace, AlignmentRule>
  buildAlignmentRules(const TargetProfile &profile) const;

  DenseMap<MemoryPlace, VisibilityRule>
  buildVisibilityRules(const TargetProfile &profile) const;

  DenseMap<MemoryPlace, SmallVector<PathEdge>>
  buildPathGraph(const TargetProfile &profile) const;

  DenseMap<PathEdge, PathKind>
  buildPathKinds(const TargetProfile &profile) const;

  DenseMap<PathEdge, SmallVector<PathConstraint>>
  buildPathConstraints(const TargetProfile &profile) const;

  DenseMap<PathEdge, PathCost>
  buildPathCostModel(const TargetProfile &profile) const;
};

// 模型验证
class TargetModelVerifier {
public:
  LogicalResult verify(const TargetProfile &profile,
                       DiagnosticEmitter &diag) const;
};
```

**各消费层的典型调用模式：**

```cpp
// 第三层：查询 memory 约束
auto paths = targetProfile.memoryModel.findPaths(MemoryPlace::GM, MemoryPlace::A1);
bool visible = targetProfile.memoryModel.isPlaceVisibleTo(MemoryPlace::VECIN,
                                                           ExecutionUnit::Vector);
CapacityRule cap = targetProfile.memoryModel.getCapacity(MemoryPlace::A1);

// 第四层：查询路径代价，辅助 movement 选择
PathCost cost = targetProfile.memoryModel.getPathCost({GM, A1, 0});
auto constraints = targetProfile.memoryModel.getPathConstraints({GM, B1, 1}); // transpose variant

// 第五层：查询 intrinsic 支持，辅助 lowering
auto matmulIntrinsics = targetProfile.intrinsicModel.computeIntrinsicMap[ComputeKind::Matmul];
auto moveIntrinsics   = targetProfile.intrinsicModel.movementIntrinsicMap[PathKind::Load2D];
```
