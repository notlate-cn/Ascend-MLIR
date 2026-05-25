# Real NPU 调试经验记录

本文记录 Ascend-MLIR mainline examples 在共享 910C 真机上的调试经验。
目标是沉淀定位方法和失败模式，不替代 `examples/real-npu.md` 的操作指南。

## 基本原则

1. xvm simulator 是候选修复进入真机前的门禁，但不是完成标准。
2. 真机失败先按 `session.error_stage` 分类，不要直接猜 kernel 源码。
3. 每个候选只改变一个变量；若同一方向连续不能改变失败签名，应回到证据采集。
4. checkpoint kernel 可以用于定位，但不能作为修复通过的证明。
5. 原始 demo 的 xvm pass 不能授权未验证候选上真机；候选自身也必须有对应 xvm gate。
6. 真机 job 结论以 `session.backend=npu`、`session.result`、`session.validation` 和 plog `errorStr` 为准。

## 失败分类

### validation mismatch

特征：

- `session.backend=npu`
- `session.result=error`
- `session.validation=fail`
- `session.error_stage=validate`
- plog 通常无 `errorStr`

处理方法：

- 先保存 novalidate 输出，再和 expected/input 逐 bucket 比较。
- 按 row tile、column tile、tail tile、split half 分桶统计误差。
- 反推 actual 更接近哪个中间表达式，而不是只看 max diff。
- 若 xvm pass 而真机 mismatch，优先怀疑 tail、alignment、queue/tensor lifetime、primitive count 语义。

### kernel launch/runtime error

特征：

- `session.error_stage=kernel_launch`
- 可能出现 `aclrtSynchronizeStream failed`
- plog 可能包含硬件错误，例如资源冲突

处理方法：

- 先读 plog `errorStr`，不要用数值 diff 思路处理。
- 若有 L0/L1/UB 资源冲突，优先查 planner、tiling、split-K、buffer 复用。
- 这类错误通常需要改变 kernel 规划，而不是只改 validation tolerance。

### hang / SIGTERM

特征：

- NPU log 停在 launch 后，没有 `session.result`
- 外部 kill 后 rc 通常是 143
- plog 不一定有有效错误

处理方法：

- 做 shape sweep，把触发条件压到最小 tile 数。
- 优先检查循环内重复 `InitBuffer`/`InitQueue`、queue depth、EnQue/DeQue/FreeTensor 配对。
- 若修复后 hang 变成 mismatch，说明至少一个真机进度问题被解决，但不代表用例通过。

## 已处理案例

### relu-broadcast-transpose

历史失败：`rtStreamSynchronize failed: rc=507035`，plog 报 VEC UB out-of-bounds
和 scalar GM address over 48 bits。

有效经验：

- launch ABI、H2D/D2H、GM pointer 512B alignment、workspace、argument count
  和 tiling words 都正常时，不要停在 runtime launch 层；继续查 generated
  kernel lowering、tile size 和 UB lifetime。
- VECOUT output 必须先从 VECOUT queue `AllocTensor`，再 enqueue。把 VECCALC
  tensor enqueue 到 VECOUT queue 可能过 xvm，但真机会触发 UB/MTE fault。
- 临时 GM-to-VECIN tensor 通过 `AllocTensor -> DataCopy -> EnQue -> DeQue`
  创建后，必须在最后一次使用后 `FreeTensor`。
- all-parallel tail-tiled kernel 的 buffer allocation 应使用 enclosing
  loop-step upper bound；DataCopy/compute 仍使用 actual tail element count。
  这样一个 max-sized queue/tbuf 可以跨 tail iteration 复用。
- 计算 all-parallel VECOUT/VECIN/VECCALC buffer byte size 时，要把
  `affine.min(remaining, step)` 和 `memref.dim(subview)` 解析到 loop-step
  upper bound。
- loop-invariant `InitBuffer` / `InitQueue` 要 hoist 到 inner tile loop 外。
  每轮循环重复 init 在 simulator 与真机上的 UB 消耗表现不同，真机可能触发
  `507035`。
- 不要把同样的 max-size substitution 盲目套到 reduction output。rank-1
  VECOUT reduction output 可能需要 exact tail size，因为 `ReduceSum2DL2`
  codegen 会从 source 和 destination tensor size 推导 rows/cols。
- `TB_N` 从 64 调到 16 可以降低该 demo 的 UB live set，但 tile 修改本身不是
  根因修复；必须同时保证 queue 和 buffer lifetime 正确。

### add-broadcast-concat

历史失败：`rtStreamSynchronize failed: rc=507035`。

有效经验：

- launch trace 若显示 argument count、H2D/D2H、512B-aligned GM pointers、
  workspace、tiling words 都正常，应把问题转向 tiling configuration 和
  generated kernel live set。
- 在该 generated kernel 中，`TB_N` 实际表现为 inner M tile size；`N=500`
  仍 full-width 进入每个 tile。
- `TB_N=192` 会让 UB live set 超界，并在 910C 上触发 `507035`。当前接受配置
  是 `TB_M=64, TB_N=16`。
- tile 配置调整也要按候选流程验证：先 xvm Ascend910B1 simulation，再真实 NPU。

### gather-elementwise-fusion

历史失败经历了两个阶段：

1. 真机 validation mismatch，xvm pass，plog 无新 `errorStr`。
2. 扩大索引范围后，真机在 kernel launch 阶段失败，plog 报
   `The address for the VEC instruction to read/write UB is out of bounds.`

有效经验：

- gather 类问题先检查索引域、tail 访问和真实输入数据范围。
- 真机 mismatch 可能来自输入索引触及硬件路径的边界行为。
- CANN `Gather` 的 index 语义是 byte offset 时，source LocalTensor
  `SetSize` 也必须覆盖 byte span；只设置元素个数会让 xvm pass，但真机
  在较高索引上报 UB 越界。
- 生成代码里要同时检查 index 转换和 source `SetSize`，不能只看 IR shape。
- 修复后需要同一 case 的 xvm 和真机都重新跑，不能用旧 job 覆盖。
- standalone 通过但 all suite 中稳定失败时，要比较失败输出更接近哪个中间值；
  当前 suite 失败样本中首 16 lanes 多数等于 bias 原值，说明 gather 结果未写入
  或未被后续 ReLU/Max 使用，而不是输入 H2D 或 expected 文件损坏。
- `PipeBarrier<PIPE_ALL>` 只能验证流水依赖方向；若 failure signature 不变，应转向
  vector mask/count 状态、kernel 间 device 状态或 runtime cleanup 证据。
- 不能把 `ResetMask()` 当成“恢复默认状态”的同义词；CANN 9.1 中它只重置
  vector mask 值，不切换 mask mode。对 Gather 这类 count/normal 语义敏感的
  API，直接 reset 到 full mask 可能把 standalone 变成 VEC UB 越界。
- 当前 all suite 中的 gather-only 失败由 `AscendC::SetMaskNorm()` 消除：
  在 Gather 前只切回 normal mask mode，不重置 full mask。对应候选在 xvm
  gather 变体、真机 standalone、真机 all suite 中均通过。

### matmul-add-leakyrelu

历史失败：`aclrtSynchronizeStream failed: rc=507015`，plog 指向 L0B read/write conflict。

有效经验：

- 带 plog 硬件错误时，先按资源/tiling 问题处理。
- biased native matmul 的 split-K 规划会引入真实硬件资源冲突。
- 禁用该路径的 split-K 后，同一 case 在 device 5 通过。
- 若 all suite 中偶发 L0B conflict，但 standalone 随后通过，要把它作为
  sequence/device-state 或资源复用问题继续隔离，不能直接归入当前 kernel
  必现失败。

### split-relu-brc-add-mul

历史失败经历了多个阶段：

1. 默认完整形状在真机 hang。
2. shape sweep 显示 `N=64` 可过，`N>64` 进入第三列 tile 后触发问题。
3. GM broadcast source 的 VECIN queue/buffer 在内层列 tile 循环中重复初始化，会导致真机 hang。
4. 将 GM broadcast source 改成固定容量并用 scalar fill 后，`M=64,N=96` 在 xvm 和真机通过。
5. `M=64,N=80` 仍真机 validation fail，错误只集中在 tail column tile `cols 64..79`。
6. post-add checkpoint 显示 `relu+bias` 在真机 tail tile 上正确，说明 bias broadcast 和 Add 本身不是根因。
7. generated kernel 在 Add 和紧随其后的 in-place Mul 之间插入 `PipeBarrier` 后，完整 NPU validation 通过。
8. 后续 no-validate 对比显示新的错误集中在 `cols 0..31`，且多数错误位置的
   ReLU 输入应为 0。
9. 在 scalar zero `Duplicate` 和后续 `Max` 之间插入 `PipeBarrier` 后，默认
   split standalone 真机通过。

关键证据：

- `M=64,N=80,HM=32,block_dim=1` 是有效的最小真机复现形状。
- novalidate 分桶比只看 `max_abs_diff` 更有用：tail mismatch 指向 Add/Mul
  依赖，首列 tile 且 ReLU 为 0 的 mismatch 指向 zero constant dependency。
- post-add checkpoint pass 能排除 bias broadcast 和 Add 本身，把问题缩到
  后续 in-place read-after-write。
- no-validate 输出中错误位置是否落在 ReLU zero bucket，可以直接指向
  `Duplicate(0)` 到 `Max` 的 producer-consumer 边。

最终问题范围：

- elementwise body lowering 复用同一个 `accumLt` 时，上一条 vector op 写入后，下一条 op 又把同一个 LocalTensor 作为输入读取。
- xvm simulator 可以顺序观察到该写入，但 910C 真机在 tail tile 上需要显式 pipe barrier，否则后续 Mul 可能读到 Add 之前的值。
- scalar constant `Duplicate` 后立刻被 vector op 消费时，也需要显式 barrier；
  否则真机可能读到未稳定的 constant buffer。
- 修复应落在 lowering 的 LocalTensor 依赖边界，而不是继续改 broadcast 或 GM copy。

早期 507035/UB live set 经验：

- 让 run manifest tiling fields 对齐 generated CANN `TilingData` signature
  是必要 hygiene，但不是该 demo 的根因。8-field tiling probe 在 UB cleanup 前
  仍以同样 `507035` 失败。
- 根因之一是 queue-backed memref alloc 还保留了 standalone TBuf initializer。
  这些 TBuf 没有实际用户，只有 `TPipe.InitBuffer`，但 hoist 后仍会消耗真机 UB。
- data-move/compute conversion 后，应删除仅被 `TPipe.InitBuffer` 使用的 TBuf。
  该 demo 中 generated `InitBuffer` 从 20 个降到 14 个后，xvm simulation
  保持通过，真实 NPU 也通过。
- `examples/split-relu-brc-add-mul` run manifest 要和当前 CANN signature 对齐：
  `TB_M`、`TB_N`、`dim_arg0_1`、`dim_arg1_0`、`dim_arg0_0`、`dim_arg3_0`、
  `dim_arg2_0`、`dim_arg4_0`。

### microcase synchronization

有效经验：

- `copy640`、`copy_tbuf640` 和 `copy_params640` 最初在真机上 validation fail，
  但 H2D roundtrip、launch 参数和 GM pointer alignment 均正常；根因是手写
  microcase kernel 缺少 MTE2 -> MTE3 同步，真机不会像仿真路径那样隐式掩盖。
- `relu_only` 在补 MTE2 -> V / V -> MTE3 后仍失败，说明问题不在 runtime ABI。
  将 `Duplicate(zero) + Max(x, zero)` 收敛为 `Maxs(x, 0)` 后真机通过。
- 后续需要单独验证 `Duplicate + binary Max` 时，不要把它混入基础 relu
  microcase。
- `scripts/real-npu-ci/run-real-npu-job.sh` 必须使用显式 microcase 顺序；glob
  字母序会让 `broadcast_add` 先运行，掩盖更基础的 `DataCopy` / vector 边界。

## 后续定位策略

1. standalone 已通过的 case 进入 all suite 后若失败，先判断是否是同一失败签名。
2. 同一签名复现时，继续按 kernel/ABI/tiling 定位；签名变化时，不复用旧结论。
3. all suite 失败但 standalone 随后通过时，优先采集序列前后 case、device id、
   plog 时间窗、runtime cleanup 和 stream/device reset 证据。
4. 对 validation mismatch，保留 novalidate 输出并按 tile/row/column/value bucket
   分析 actual 更接近哪个中间表达式。
5. 对 kernel launch error，先看 plog `errorStr`，再决定是资源冲突、UB 越界、
   还是 runtime/device-state 类问题。
6. 每个修复候选重新跑：
   - matching xvm sim gate；
   - 真机 standalone；
   - 必要时真机 all suite。
