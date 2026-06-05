# 设计:尾块处理从 overlap-tail 改为单一 ragged 路径

状态:设计已对齐,待评审。日期 2026-06-05。
关联:取代 `docs/superpowers/notes/2026-06-05-overlap-tail-reduce-f16-bug.md` 的
overlap-tail 机制;纠正其 §2.1 的误诊。

## 1. 背景与问题

`auto-fuse` 把一根轴按两级 tile 切:外层 `XBLOCK`(多核),内层 `XBLOCK_SUB`
(= T)。当 `extent` 不能被 T 整除时,末尾剩一个真实大小 `rem = extent - covered`
(`covered = floor(extent/T)*T`,`0 < rem < T`)的零头。

**当前机制(overlap-tail)**:不处理零头,而是从末尾**回退一整块 T**,重算
`[extent-T, extent)`,与主循环重叠。目的是让每块恒为静态 T(片上 buffer 固定、
DataCopy 长度固定且 32B 对齐、无 mask)。

### 1.1 当前机制的两个失败前提(经代码核实)

| 模式 | 真病因 | 现状 | 出处 |
|------|--------|------|------|
| ① f16 不对齐 | parallel overlap-tail 每行幂等、值对,但尾块 GM 偏移 `(extent-T)*elemBytes` 对 f16(2B)常非 32B 对齐 → `DataCopy` 非法 | **已触发**(`reduce-sum-3d-f16-tail-e2e` XFAIL) | DataCopyPad 仅在累加器 store 触发(`CannTranslation.cpp:2627`),未推广到尾块 load/store |
| ② reduction 轴尾巴 | reduction split 的 `for(0, ext, RBLOCK)` 不整除时尾巴漏算/越读,无约束兜底 | **潜在**(今天靠整行单 tile 规避) | `GroupEmitter.cpp:185`;`TilePlanBuild.cpp:341-351` 只保证 RBLOCK\|XBLOCK |

### 1.2 纠正一处误诊

原 note §2.1 称 "reduce 非幂等 → overlap 双算"。**代码核实:不成立。**
`emitGroupWithReductionSplit`(`GroupEmitter.cpp:52`)与 overlap-tail 的 scf.if
(`GroupEmitter.cpp:715-780`)是**互斥两条路** —— 有 inner reduction 轴时
`:702` 提前 return,overlap-tail **只在 parallel 轴 fire**,每行独立 reduce →
幂等、值对。所以"重叠双算"在当前代码里**从不发生**;reduction 轴的真问题是
§1.1 模式②的漏算,不是双算。

> 推论:修复**不需要** reduce identity(-inf/+inf/1)。代码现有 identity 基建只
> 有 0(`ComputeReduce.cpp:231`),本设计也不新增。

## 2. 目标与非目标

**目标**:用单一 ragged 尾块机制取代 overlap-tail,使 reduce / f16 / 不整除尾巴在
AscendC 原生 codegen 下正确,去掉对 aclnn 路由和 `Divides{32}` 拒绝约束的依赖。

**非目标**:不新增 reduce identity 基建;不移植 AF 完整 `AlignmentStrategy`
(~1200 LOC);不改主循环的 `DataCopy` 快路径(只改尾块)。

## 3. 核心机制:固定 buffer、可变有效长度

关键洞察:**"不补满 T" ≠ "buffer 要变长"**。两者解耦:

- **buffer 仍按静态 T 分配**(沿用主循环那块 UB)→ bufferize 不因动态尺寸包
  shadow-alloc(保留 overlap-tail 当初图的这个好处)。
- **搬运/store 长度用真实 `rem`**,经 `DataCopyPad` 表达(吃下不对齐偏移与不整长)。
- **compute 照算满 T**:尾 lane 是垃圾,但**永不 store**(store 长度 = rem),故无害。
- **唯一需要 count 的是 reduce**:reduction 轴尾巴在累加层穿 `count=rem`,只折真实元素。

```
extent=50, T=16:  covered=48, rem=2
  现在(错):  回退 [34,50) 整块16, GM偏移 34*2=68B → f16 非 32B 对齐 → 崩
  新机制(对): buffer 仍 16; DataCopyPad 只搬 [48,50) 这 2 个, GM 端从 base+48 寻址(GlobalTensor 基址)
              不重算、不补假元素、不变长 buffer
```

三个失败前提同时拆掉:f16 → DataCopyPad 吃不对齐;reduce → count 折真实元素;
不整除 → 直接处理 rem。

### 3.1 循环结构

```
for blk in [0, covered) step T:        # 主循环,不变
    DataCopy(GM[blk..blk+T) → UB)       # 快路径,32B 对齐恒成立
    body(count = T)
    DataCopy(UB → GM[blk..blk+T))
if covered < extent:                   # 零头,唯一 ragged 块
    rem = extent - covered
    DataCopyPad(GM[covered..extent) → UB, len=rem, off=covered)
    body(count = T)                    # elementwise 算满 T(尾 lane 垃圾不 store)
    DataCopyPad(UB → GM[covered..extent), len=rem, off=covered)
```

`rem` 静态(static shape)或运行时计算(dynamic shape)均可;`rem==0`(整除)时
跳过整个 if。`extent < T` 时 `covered==0`,主循环空转、只跑 ragged 块。

## 4. 组件改动

### 4.1 把 DataCopyPad 推广到 ragged 尾块的 load + store(核心 enabler)

**先纠正一处机制理解**:`CannTranslation.cpp:2636` 的
`DataCopyExtParams{1, $2*sizeof, 0, 0, 0}` 那几个 "0" **不是 GM 偏移**,而是
`{blockCount, blockLen(bytes), srcStride, dstStride, rsvd}`(块间 stride)。
**GM 偏移不在 ExtParams 里**,而是在 GlobalTensor 基址层,且**已支持**:

- `:2612` `SetGlobalBuffer(reinterpret_cast<__gm__ T*>($1) + $2)` —— 基址 + 元素偏移;
- `:2647` `GlobalTensorBracketOp` → `GetPhyAddr(offset)` 重建偏移后的 GlobalTensor。

所以真实改动**比"加 offset 字段"更小**,只有两条:

1. **偏移**:ragged 尾块的 GM 端用 `base + covered` 寻址 —— 直接复用上面已有的
   GlobalTensor 基址机制,无需新字段。DataCopyPad 的本职就是 GM 端可不对齐、UB 端
   32B 对齐(buffer 静态 T 已满足)。
2. **触发**:DataCopyPad emission 现在(`:2627` 的 `walk`)**只在 TBuf 累加器
   store 这个特例触发**。需推广:让 ragged 尾块的 GM↔UB **load 与 store** 都走
   DataCopyPad(按 marker / 尾块标记区分,而非仅累加器特例),长度填 `rem*elemBytes`。

注:`DataCopyExtParams` 的 `blockCount/srcStride/dstStride` 在**扁平 1D 尾块**保持
`1/0/0`;若尾块是 **rank-N strided**(如 `15e357b0` 修的内轴切分列条带),则按现有
strided-copy 逻辑填 `blockCount + stride`,不是恒 0。

### 4.2 尾块 emission 改为 ragged

`LoopNestBuilder` / `GroupEmitter.cpp:715-780`:

- **删** overlap-tail 的 `tailComposed = innerTileExtent - innerTileStep` 重叠逻辑;
- 改发 §3.1 的 ragged 块:`rem` 计算 + DataCopyPad(load/store,len=rem,off=covered);
- 尾块 UB tensor **显式按静态 T 分配**(不按 rem),确保不触发 shadow-alloc;
- elementwise body 不变(算满 T)。

### 4.3 reduction 轴尾巴(模式②)

`emitGroupWithReductionSplit`(`GroupEmitter.cpp:52`)的 reduction `for` 改为覆盖
完整 extent:末尾零头单独处理,累加步 `AddL2/MaxL2/MinL2`(`ComputeReduce.cpp:497,
595-614`,已带 `cnt` 操作数)传 `count=rem`,只折真实元素。块内 final
`ReduceSum`(`CannTranslation.cpp:3184`,shape 写死、穿不了 count)仅在需要折列且
列是尾轴时受影响;当前 reduce 把整行放进单 RBLOCK,列非尾轴,不触发。若未来列成
尾轴,退路是对 final-sum 的 `UB[rem..T)` 填 0(sum identity,基建已有,不碰 max/min)。

### 4.4 删除/放宽约束

- **删** overlap-tail 专属代码路径(§4.2)。
- **删/放宽** `TilePlanBuild.cpp:379-414` 的 `Divides{32,(extent-INNER)*elemBytes}`
  硬拒绝约束 —— DataCopyPad-with-offset 已能正确处理不对齐。保留为**软偏好**
  (cost tiebreak,对齐 tiling 仍略快、优先)而非硬约束。
- reduction 轴**不再**需要 `Divides(RBLOCK, extent)` 兜底(§4.3 已正确处理尾巴)。

## 5. 边界情形

| 情形 | 处理 |
|------|------|
| `extent % T == 0`(无尾) | `rem==0`,跳过 ragged if |
| `extent < T` | `covered==0`,主循环空,只跑 ragged 块 |
| dynamic shape | `rem = extent - covered` 运行时算;DataCopyPad len/off 用 SSA 值 |
| 尾 lane 含 NaN/inf | elementwise 算满 T 产生 NaN 在尾 lane,但不 store → 无害;不主动清零 |
| 多输出 kernel | 每个输出各发一条 ragged store(len=rem,各自 off) |

## 6. 验证

- **主 gate**:`examples/reduce-sum-3d-f16-tail-e2e/run.sh` 从 XFAIL 转 **PASS**
  (sim,后续真机)。
- **回归**:GPT-2 tiny 那 49 个 AscendC elementwise kernel(尾块从 DataCopy 换
  DataCopyPad)、`two-elewise(-dyn)`、`gelu-dyn`、`reduce-big-r` 全 PASS,max_diff
  不退化;lit 全绿。
- **新单测**:构造 reduction 轴不整除尾巴用例(模式②),验证 count=rem 正确。

## 7. 一句话

废 overlap-tail,改唯一 ragged 尾块:buffer 仍静态 T(无 shadow-alloc),搬运/store
长度 = rem(DataCopyPad 带 offset 吃不对齐),elementwise 算满 T 不 store 垃圾,
reduce 靠累加层已有的 count 只折真实元素。不需要 identity 基建,只改尾块、不动主循环。

## 参考

- `lib/Target/CannKernel/CannTranslation.cpp:2627`(DataCopyPad 仅累加器 store,待推广到尾块)、
  `:2612`/`:2647`(GM 偏移走 GlobalTensor 基址,已支持)、`:3184`(final ReduceSum 穿不了 count)
- `lib/Conversion/AutoFuse/TileFuse/GroupEmitter.cpp:52`(reduction split)、
  `:715-780`(待删的 overlap-tail)
- `lib/Conversion/LinalgToAscendC/ComputeReduce.cpp:497,595-614`(累加 count 已支持)、
  `:231`(identity 仅 0)
- `lib/Conversion/AutoFuse/TileFuse/TilePlanBuild.cpp:379-414`(待放宽的 Divides{32})
- 相关记忆:`project_af_alignment_strategy_port`、`project_reduce_path_r3`
