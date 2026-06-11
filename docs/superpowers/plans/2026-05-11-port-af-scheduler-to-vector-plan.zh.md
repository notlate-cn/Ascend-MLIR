# 把 AutoFuse 的 schedule 逻辑移植到 vector-plan TilePlanGen（v2）

> **给执行者：** 分阶段实现。每个阶段范围明确、以"测试全绿 + 一个 commit"收口。
> **本版相对 v1 的改动**（根据 review）：
> - P1 验收门松绑：不要求 codegen 字节级 diff=0，只要 **lit + e2e 精度通过**。
> - 轴选取**进一步对齐 AF**：引入 `vectorized_axis`（每个 tensor 的内层"向量化轴"，不参与切分），ub-tiling 轴由**枚举 tiling case**（y_group × x_group × r_group 笛卡尔积 + RCore 变体）+ 打分选出，不用我之前那条"Y 排在 R 后面就不能当 ub 轴"的临时启发式 —— 那条规则其实是 `vectorized_axis` 机制的一个推论。
> - transpose：**两套模板都生成**（保留模板 = 策略 b，作为兜底；消除模板 = 策略 a，把置换吸进上游节点的轴序），打分选 —— 这正是 AF `transpose_schedule_case_generator` 的做法。
> - reduce 模板：**对齐 AF 三模板** `kCommon / kAllLoad(=FullLoad) / kRCore`，RCore 不再"先 TODO"，按 AF `reduce_schedule_case_generator` 做（如确实太大再在 P3 里临时降级，但默认目标是做掉）。

**目标：** 在我们的 MLIR / `linalg.generic` 流水线上重新实现 AutoFuse `optimize/autoschedule` + `att` 那套轴处理的*模型*：轴语义分类（X/Y/R/N）+ `vectorized_axis` + block 轴 fuse + tiling-case 枚举 + 打分 + reduce 三模板 + transpose 双模板 + 最简 UB 峰值约束。

**非目标：** concat / split / gather 的 schedule case（留 hook + TODO）；*调好的* score function（先放占位规则，但接口/结构按 AF 来）。

**AF 参考（只读）：**
- 文档：`/home/gser/code/ge-eco/.../autofuse/doc/{schedule_ATT轴处理全流程说明,reduce全流程说明,transpose全流程说明,dynamic_shape峰值内存约束量化说明}.md`
- 源码：`autofuse/optimize/autoschedule/{autoschedule.cpp,schedule.cpp,tiling_group.cpp}`、`autofuse/optimize/task_generator/{reduce,transpose,concat,split}_schedule_case_generator.cpp`、`autofuse/optimize/optimize.{h,cpp}`、`autofuse/att/...`

**编译：** `cd build && ninja afir-opt afir-translate 2>&1 | tail -20`
**lit：** `PATH="$PWD/externals/llvm-project/build/bin:$PWD/build/bin:$PATH" externals/llvm-project/build/bin/llvm-lit build/test/Conversion/{Collapse,VectorPlanCodegen,LinalgToAscendC} -q`
**E2E 门禁：** `examples/{relu-e2e,reduce-sum-3d-e2e,combo-elewise-reduce-e2e,bcast-multi-axis-e2e,reduce-axis1-e2e}/run.sh` 全 `validation=pass`

---

## 0. 为什么不是"照搬代码"

AF 跑在 GE `ComputeGraph` 上，C++，直接改节点 attr（`sched.axis`、`tensor.attr.axis`、`tensor.attr.vectorized_axis`、`repeats`、`strides`），`ApplySplit/Merge/Reorder` 落图。我们这边是 MLIR：`linalg.generic` + `affine_map`（loop↔operand 关系在 map 里），`vector-plan-tile-fuse` → `CollapsedGroupInfo` → **TilePlanGen** 出 `TilePlan` → **LoopNestBuilder** 物化 `scf.for` → **GroupEmitter** clone linalg op 进最内 body。所以"落图"结构不同，但 **`TilePlan` ≈ AF `TilingCase`**，TilePlanGen ≈ `AutoSchedule::DoAutoSchedule` + `Scheduler::DoScheduler`。我们重新实现的是*模型*。

### AF 的实际流程（从源码读出来的，作为我们对标的"黄金路径"）

`autoschedule.cpp`：
1. `TilingGroup::GenTilingGroup(graph, axes_group, is_reduce_full_load)` —— 把所有 loop 轴分进 `x_group / y_group / r_group / n_group`（+ `axes_order`）。
2. `TilingGroup::NormGroup(axes_group)` —— 归一化（去空、补默认）。
3. `GenTilingCase(tiling_cases)` —— **枚举**：
   - cube 路径：每个 y_id 一个 case（只设 `ub_tiling_id_y`）。
   - gather 路径（非平凡 gather）：从第 2 个 y 起每个一个 case，`block_tiling_id=0`。
   - 通用路径：`for x_id in x_group: for y_id in y_group: for r_id in r_group:` → 一个 `TilingCase{ub_tiling_id_x=x_id, ub_tiling_id_y=y_id, ub_tiling_id_r=r_id, block_tiling_id=0}`；若 `is_reduce_first_stage && r_id 有效`，再追加一个 `reduce_is_block=true, block_tiling_id=1` 的变体（= **RCore**）。
4. `PruneTilingCase` —— 单切场景下，若 `ub_tiling_id_y` 那根轴 size==1 且 case 数>1，删掉它（不切 size-1 轴）。
5. 对每个 `TilingCase`：`ProcessOneTilingCase` → `Scheduler::DoScheduler()`（TileSplit/BlockSplit/ApplyMerge/ApplyReorder）→ 若 `reduce_is_block` 则 `BindBlock` → `SelectLoopAxis`（见下）。
6. `TemplateGeneratorHandler::GenerateTemplates` —— 在上面之外再叠 reduce/transpose/concat/split 的"多模板"（各 `*_schedule_case_generator` 各自生成几套图 + score func）。
7. 最后由打分机制（每个 schedule 产物带一个 `score_func`，是一段会被编进 host tiling 的 C++）选最优。

`SelectLoopAxis`（`autoschedule.cpp`）：对每个 compute 节点，
- `not_loop_axis_set` = 该节点所有 output 的 `vectorized_axis` ∪（若有 reduce 且非 full-load：该节点 input 里 `repeats==1 && strides==0` 的轴 —— 即 reduce 轴 —— 但排除已进 block 轴的）。
- 节点的 `sched.axis` 里凡落在 `not_loop_axis_set`（含其 `from` 链全是 not-loop）的 → 置 `kIdNone`（不生成 loop）。
- 节点的 `loop_axis` = 剩下的 `sched.axis` 里**最内**那个非 `kIdNone` 的。
- → **`vectorized_axis` 的轴不会有循环；它们留在 op 的 tile 里被一条向量指令处理。**

→ Case B `[P,R,P]`（`out[d0,d2]=sum_{d1}x[d0,d1,d2]`）在 AF 里：reduce 节点 output `out` 的 `vectorized_axis ⊇ {d2}`，input 的 reduce 轴 = `{d1}`，所以 `not_loop_axis_set ⊇ {d1, d2}` → 只剩 `d0` 能当 loop/block 轴 → block=d0、没有可切的 ub 轴 → d0 退化成 step-1 行循环。**这正是我们 splitParallel hack 出来的结果**，但在 AF 里是 `vectorized_axis` 机制的自然产物，不是特例。

---

## 1. 术语映射表（AF ↔ 这边）

| AF | 这边 |
|---|---|
| `sched.axis`（节点循环轴） | post-collapse `linalg.generic` 的迭代轴，按 `CollapsedGroupInfo::collapsedAxes` 编号 |
| `tensor.attr.axis`（张量逻辑轴序） | 某 operand 的维序 = 该 operand `affine_map` 的 result 顺序 |
| **`tensor.attr.vectorized_axis`**（一次向量化局部参与的轴子集） | **新增**：每个 operand 上算出来的 `vectorizedDims`（block/ub 切完后仍整维进 tile 的那些迭代维） |
| `AxisGroup`（`x_group/y_group/r_group/n_group` + `axes_order`） | **新增** `AxisGrouping`（`AxisClass[]` over collapsed 轴） |
| `TilingCase`（`ub_tiling_id_x/y/r`、`block_tiling_id`、`reduce_is_block`、`reduce_outer_id`…） | **扩展后的** `TilePlan`（哪些轴 Outer/Inner/Full、blockFusedAxes、ubTilingAxis(x/y/r)、reduceTemplate、reduceIsBlock…） |
| `Scheduler::TileSplit / BlockSplit / SelectLoopAxis / ApplySchedAxisMerge / ApplySchedAxisReorder` | `TilePlanGen`（决策 + 算 `vectorizedDims` + `loop_axis`）+ `LoopNestBuilder`（物化） |
| `SubAxis / AttAxis`（`is_bind_multi_core / enable_tail / enable_pad / is_last / is_node_innerest_dim / is_reduce_split_axis / is_broadcast_split_axis`…） | `TileParam` **+ 新增 flag 字段** |
| ATT 符号化 `tail_size / loop_num / block_dim` | tunable func args + `TilePlan::blockDimExprs`（**+ 新增约束列表**） |
| `score_func`（编进 host tiling 的一段 C++） | **新增** `costEstimate(TilePlan)` —— 占位规则，但按"每个候选带一个分数、取最优"的结构来 |
| reduce 模板 `kCommon / kAllLoad / kRCore` | `TilePlan::ReduceTemplate { None, Common, FullLoad, RCore }` |

---

## 2. 新数据结构（`include/Conversion/VectorPlan/TilePlan.h`）

```cpp
namespace mlir::vector_plan {

enum class AxisKind { Y, R, X, N };
// Y/R/X/N 对应 AF 的 y/r/x/n_group（见 §3.1）

struct AxisClass {
  AxisKind kind = AxisKind::Y;
  bool bindMultiCore = false;   // 可做块派发（≈ SubAxis::is_bind_multi_core）
  bool enableTail    = true;
  bool enablePad     = false;   // 不齐 DataCopy → DataCopyPad（future）
  bool isReduceSplit    = false;
  bool isBroadcastSplit = false;
  int  origPos = -1;
};

struct AxisGrouping {
  SmallVector<AxisClass> axes;          // 每个 collapsed 轴一个
  SmallVector<int> yAxes, rAxes, xAxes, nAxes;  // index 列表，按 axesOrder
  SmallVector<int> axesOrder;
};

struct TileConstraint { enum Kind { Divides, LeBytes } kind; Value lhs, rhs; };  // §6

// TileParam 加上 AxisClass 的 flag（默认初始化，老调用点照编）。
// TilePlan 加：
//   enum class ReduceTemplate { None, Common, FullLoad, RCore };
//   ReduceTemplate reduceTemplate = ReduceTemplate::None;
//   bool reduceIsBlock = false;
//   SmallVector<int> blockFusedAxes;   // fuse 进 block 轴的 collapsed-axis id（按序）
//   int ubTilingAxisY = -1, ubTilingAxisX = -1, ubTilingAxisR = -1;  // ≈ ub_tiling_id_*
//   // 每个 operand 的 vectorizedDims（迭代维 id 集合）；GroupEmitter/ComputeConversion 用
//   DenseMap<Value, SmallVector<int>> vectorizedDims;
//   SmallVector<TileConstraint> constraints;
} // namespace
```

---

## 3. 重构后的 TilePlanGen

`genVectorTilePlan` 变薄编排器（对标 `AutoSchedule::DoAutoSchedule`）：

```
genVectorTilePlan(func, info, ...):
  1. AxisGrouping g = classifyAxes(info);                         // §3.1  ≈ GenTilingGroup
  2. normGroup(g);                                                //        ≈ NormGroup
  3. SmallVector<TilePlanDraft> cases = enumerateTilingCases(g);  // §3.2  ≈ GenTilingCase + PruneTilingCase
  4. SmallVector<TilePlan> built;
     for (draft : cases) {
       TilePlan p; blockSplit(g, draft, p);   // §3.3  ≈ Scheduler::BlockSplit (+RCore: ReduceBlockTiling)
       ubSplit(g, draft, p);                  // §3.4  ≈ Scheduler::TileSplit
       computeVectorizedDims(g, p);           // §3.5  ≈ FindNotLoopAxis/SelectLoopAxis 的前半
       selectLoopAxes(g, p);                  // §3.5  ≈ SelectLoopAxis（决定哪些轴有 scf.for）
       addConstraints(p); estimateMemPeak(p); // §6
       if (legal(p)) built.push_back(p);
     }
  5. // transpose 多模板（§5）：若 group 里有 transpose，再生成"消除"变体的草案，重跑 1–4
  6. return pickBest(built, costEstimate);    // §3.6  ≈ score 选最优
```

**P1（重构）的验收门**（已松绑）：现有 lit 全过 + 所有 e2e `validation=pass`（精度通过即可，不要求 codegen 字节不变）。其余阶段往里加 group 类别 / 候选 / 模板。

### 3.1 `classifyAxes(info) -> AxisGrouping`（≈ `TilingGroup::GenTilingGroup` / `tiling_group.cpp`）

- **elewise/broadcast/load/store/cube**（`GenElewiseTilingGroup`）：所有 loop 轴 → `y_group`。
- **reduce**（`GenReduceTilingGroup` / `CalcReduceAxes`）：reduce 轴判定 = `input_stride != output_stride && output_stride == 0`（这边迭代器类型已经说明，但**保留 stride 检查作交叉校验 assert**）；命中 → `r_group`、`isReduceSplit=true`；其余 → `y_group`。**full-load 变体**（`GenReduceTilingGroupFullLoad`）：reduce 轴改进 `n_group`（不进 r_group）。
- **broadcast 轴**（`info.broadcastAxes`）：保持 `y_group` 但 `isBroadcastSplit=true`、`bindMultiCore=false`。
- **transpose**（`GenTransposeTilingGroup`）：从尾向前，input-pos==output-pos 的维 → `n_group`；从第一个不同位起：**输入侧轴 → `x_group`，输出侧轴 → `y_group`**；剩余交叠 → `y_group`。
- **concat**（`GenConcatTilingGroup`）：`concat_dim` 之前的轴 → `y_group`；`concat_dim` 及之后 → `n_group`；输入/输出在 `concat_dim` 上的 axis_id 也进 `n_group`。**首轴 concat 或前积=1 → 退化成 y_group**（首轴 concat schedule 前会改写成 load/store）。—— **本期留 TODO 函数体**。
- **split**（`GenSplitTilingGroup`）：与 concat 对称（`split_dim` 前 → y、之后 → n）。首轴 split 退化。—— **本期留 TODO**。
- **gather**：默认按 `y_group`（合轴/block 选取上另有限制，见 §3.3）。—— **本期留 TODO**。
- `axes_order` = 这些轴在原始逻辑序里的顺序。`bindMultiCore` 初值 = `(kind==Y && !broadcast)`。
- `NormGroup`：去空组、保证 `axes_order` 覆盖全部、补默认值。

### 3.2 `enumerateTilingCases(g) -> [TilePlanDraft]`（≈ `GenTilingCase` + `PruneTilingCase` / `autoschedule.cpp`）

`TilePlanDraft { int ubX=-1, ubY=-1, ubR=-1; int blockTilingId=0; bool reduceIsBlock=false; }`

- **gather 路径**（非平凡 gather）：从第 2 个 `y` 起每个一个 draft，`blockTilingId=0`。（本期 gather TODO，先不触发。）
- **通用路径**：`for x_id in xAxes(默认含一个 -1 占位): for y_id in yAxes: for r_id in rAxes(默认含 -1):` → `TilePlanDraft{ubX=x_id, ubY=y_id, ubR=r_id, blockTilingId=0}`；若 `is_reduce_first_stage && r_id != -1` 再追加 `{...,reduceIsBlock=true, blockTilingId=1}`（RCore）。
- **prune**：单切场景（`ubX==-1 && ubR==-1`）下，若 `ubY` 那根轴 size==1 且 draft 数>1 → 删。
- 至少留一个 draft（否则报图不合法）。

> 第一版可以**先只枚举 `yAxes`（× `rAxes`，reduce 时）**，`xAxes` 暂用占位 -1（transpose 单独走 §5 的多模板），即先不上 x 维笛卡尔积 —— 但**结构按上面来**，方便以后补。

### 3.3 `blockSplit(g, draft, p)`（≈ `Scheduler::BlockSplit` + `FuseTileOutAxes`；reduce: `ReduceBlockTiling` / `schedule.cpp`）

- **普通**：沿 `axesOrder` 收集 `bindMultiCore==true`（Y、非 broadcast）的**前导连续段**，遇第一个 `N`/`X`/`R` 停。该段 = `blockFusedAxes`。空段 → 退化"第一个 Y"或"不分核"。block 轴 extent = 段内 extent 之积。emit tunable `XBLOCK`（default 128），`blockDimExprs += ceildiv(积, XBLOCK)`。
  - gather：通常**只保留第一根 outer 轴做 block，后续 outer 不全 merge**（`FuseTileOutAxes` 里 gather 分支）。
- **RCore**（`draft.reduceIsBlock`）：block 轴 = `(非 R 外轴 merge) ++ (reduce 轴的一个 tile-split)`；记 `reduce_block_tiling`、`rm_org_size`、`a_org_size`；落图时 `BindBlock(block_tiling_id, reduce_block_inner_id)`。
- **物化**（P2 的 LoopNestBuilder 改动）：对 fuse 出的乘积 emit *一个* 外层 `scf.for`（步长 XBLOCK），再 `div`/`mod` 恢复各原始轴 IV。段长 1 时退化成今天的单轴循环（canonicalize 折掉 div/mod）。

### 3.4 `ubSplit(g, draft, p)`（≈ `Scheduler::TileSplit` / `TileTiling` / `ApplyTiling`）

- `ubTilingAxisY = draft.ubY`（若 != -1 且不在 block 轴里）：给它 `Inner` 层 + tunable `XBLOCK_SUB`（default 16）+ tail-peel（今天的 overlap-tail 逻辑）。同理 `ubTilingAxisR`（reduce 切轴，开 reduction-split 时）、`ubTilingAxisX`（transpose 的 x 轴，少见）。
- 其余 `Y` 轴：`Full`（step=extent）→ SliceComputer 整维 slice。（**替掉今天每个 parallel 轴一个 `XBLOCK_SUB_n`** —— splitParallel 的"非 ub 轴全载"成为默认。）
- broadcast 轴：小 ⇒ `Full` step-1（`BCAST_n`）；大 ⇒ `Inner` tunable（`BCAST_TILE_n`）。
- `R` 轴：`Common` ⇒ `Full` step=extent（留 generic 里，降 `reduce_sum_2d_l2`/`ReduceSum`）；`RCore` ⇒ reduce 一部分在 block 轴（§3.3）+ 余 Full；`FullLoad` ⇒ R 轴已改判 N。
- `N` 轴：无 plan 条目、整维 slice、无循环。
- `X` 轴：§5 处理；TilePlanGen 视角当 N。

### 3.5 `computeVectorizedDims` + `selectLoopAxes`（≈ `FindNotLoopAxis` + `SelectLoopAxis` / `autoschedule.cpp`）

- 对每个 operand：`vectorizedDims[operand]` = 该 operand `affine_map` 里出现的、**没被 block 轴吃掉、也不是 ubTiling 轴**的迭代维（即"整维进 tile"的那些）。把它存进 `p.vectorizedDims`，GroupEmitter（决定 linalg op 的 tile rank/位置）和 ComputeConversion（AR/RA layout、totalElems）都读它。
- `notLoopAxisSet` = ∪ 所有 output operand 的 `vectorizedDims` ∪（有 reduce 且非 full-load：input 里的 reduce 轴，但排除已进 block 轴的）。
- 哪些迭代维生成 `scf.for`：`{ blockFusedAxes 的 Outer 循环 } ∪ { ubTiling 轴的 Inner 循环 } ∪ { 小 broadcast 轴的 step-1 循环 }`，且**减去** `notLoopAxisSet` 里的。
- **退化情形（= Case B）**：若 block 轴是某轴 `a` 且没有合格的 ubTiling 轴（所有 y 候选都在 `notLoopAxisSet` 里，比如 `[P,R,P]` 的 d2 因为是 reduce 输出的 vectorized 维而进了 notLoop），则给 block 轴 `a` 一个 step-1 的 Inner 循环（今天 splitParallel 的行为）。这条不是特例，是"ub 轴选不出来"的兜底。

### 3.6 `costEstimate` + `pickBest`（≈ score_func + 选最优）—— **最简**

- 每个 built 出来的 `TilePlan` 给一个分：`score = w1·(blockDim 离 #AICores 的距离) + w2·(有 step-1 小步长轴 ? 1 : 0) + w3·(UB 峰值溢出惩罚, > 容量则 ∞) + w4·(vectorized region 字节数, 越大越好 → 负权)`。
- transpose 的 preserve vs eliminate：照 AF `TransposeScoreFunctionGenerator` —— 尾轴转置 → preserve 优先；非尾轴转置 → 尾轴维 < `512B/dtype_size` 则 preserve 优先，否则 eliminate 优先。
- `pickBest` = argmin score；平局 → 枚举序靠前（`Common`、preserve 是安全默认）。

---

## 4. reduce 三模板（吸收 Case-B 的 splitParallel hack）

对标 AF `optimize.h::ReduceTemplateType { kCommon, kAllLoad, kRCore }` + `reduce_schedule_case_generator.cpp` + `reduce全流程说明.md`。

- **classifyAxes** 给 reduce 轴判 `R`（Common/RCore）或 `N`（FullLoad）。
- **enumerateTilingCases** 给 reduce 出 `ubR ∈ rAxes` × `ubY ∈ yAxes` 的 case；`is_reduce_first_stage` 时每个再追加 `reduceIsBlock=true` 变体（RCore）。
- **`Common`**：reduce 轴 Full，留 generic，`ComputeConversion` 按 §3.5 的 `vectorizedDims` 决定 AR vs RA（reduce 轴在 operand 末维 → AR；后面还有非 size-1 维 → RA），`CannTranslation` 的 RA 分支已 commit（`66a126d`）—— **这部分不动，保留**。
- **`FullLoad`（=kAllLoad）**：reduce 轴进 N、整维全载，整个 reduce 在一个 vector 区里做（适合 reduce 轴小）。仅当 `reduction_tile_bytes ≤ UB_budget`（§6）时作为候选。
- **`RCore`（=kRCore）**：R 轴切多核 —— block 轴 = `(非 R 外轴) ++ (R 块轴)`，每核出 partial reduce，再第二阶段 combine（两阶段：`groups_relations_in` 表达 group 间依赖；对标 `GenerateGeneralCase` 的 `ReducePartition*` + `IsReduceFirstStage` + `RefreshGroupRelation`）。适合 P 小 R 巨大。**按 AF 做掉**；若实现量确实超预算，P3 里临时把 RCore 候选标"永不选中"并留 TODO，但默认目标是做。
- **Case B 不再有专门分支**：`[P,R,P]` 走 `Common` —— blockSplit 前导 Y 段 = `{d0}`（遇 R 停），ubSplit 选不出合格 ubY（d2 在 notLoopAxisSet），§3.5 退化 → d0 step-1 内循环 + generic tile `[1,D1,D2]` + RA reduce。把 `TilePlanGen.cpp` 现有的 `splitParallel` if-分支删掉。

---

## 5. transpose 双模板（对标 `transpose_schedule_case_generator.cpp` + `transpose全流程说明.md`）

AF：永远生成两套图模板，打分选 —— **保留模板**（不改图，Transpose 节点保留，UB 里重排，处理"尾轴转置"和"非尾轴转置且尾轴<512B"）+ **消除模板**（改图：把 Transpose 输出的轴序沿路径向上 `ApplyTensorAxisReorder`/`ApplySchedAxisReorder` 摊到上游所有节点，再删 Transpose 节点；处理"非尾轴转置且尾轴≥512B"；只在 Transpose 之上无额外向下分支时可用）。

我们这边：
- **保留模板（= 策略 b，兜底）**：`classifyAxes` 标 X/N（§3.1 transpose 分支），ubSplit 不把 X / transpose-N 选成 ubTiling 轴、不内层切它们；`linalg.transpose` 保持独立 op，给它**单独的 lowering**（LinalgToAscendC 里加 transpose conversion）emit AscendC 转置原语（`DataCopy` 配转置 `repeat/stride`，或 `Transpose`/`TransDataTo5HD` 风格）在那个 tile 上做。
- **消除模板（= 策略 a，优先尝试）**：把 `linalg.transpose` 的置换吸进**下游消费者的 indexing map**（仿 `BroadcastAbsorb.cpp`：消费者 `linalg.generic` 的对应 operand 的 `affine_map` 复合上这个 permutation；transpose 之上若有别的 linalg op，也相应改它们的 map），然后删掉 transpose op。这等价于 AF 的"沿路径 ApplyTensorAxisReorder 后删节点"。前置条件：transpose 输出只有一个消费链（无额外分支）。做成 `enumerateTilingCases` 之外的一个"草案变体"：先对原图跑一遍 §3.1–3.6 得保留模板的 plan，再对"吸收后的图"跑一遍得消除模板的 plan，两个都进 `built`，§3.6 打分选。
- 打分按 §3.6（AF `TransposeScoreFunctionGenerator` 的尾轴/512B 规则）。

---

## 6. tiling-data 约束 + UB 峰值内存量化 —— **最简**

- `TileConstraint`：`{Divides, XBLOCK_SUB, XBLOCK}`（现有 tail-peel 前置）、`{LeBytes, Σ live tile bytes, UB_capacity}`。emit 进 `vector_plan.tiling_infos` 的新 `constraints` 数组（给 tiling-space 生成器 / autotuner 用）。目前只记录。
- **UB 峰值**：linalg op 在最内 body 时同时 live 的片上 buffer = 每个 input tile（VECIN）+ output tile（VECOUT）+ VECCALC 累加器/临时 + reduce workspace。峰值 ≈ Σ `tile_extent · elem_bytes`（用 tunable extent 符号化，如 `XBLOCK_SUB · D2 · 4 + …`）。和目标 SoC 的 UB 容量（从 soc 配置读，如 192/256 KB）比；某候选符号峰值（代入默认 param 值）超容量 → `costEstimate=∞`（剪掉）。更聪明的版本会自动缩 `XBLOCK_SUB`，第一版只剪。精确量化模型参考 `dynamic_shape峰值内存约束量化说明.md`（先做简单求和，cite 为 future）。

---

## 7. 阶段、文件表、验收门

| 阶段 | 范围 | 验收门 |
|---|---|---|
| **P1** ✅ | §2 数据结构 + §3 重构骨架（`classifyAxes`/`pickBlockAxis`/in-order ubSplit），**行为不变**。老 `splitParallel` hack 暂以 `degradeToRowLoop` 路径等价保留（P3a 收编）。`LoopNestBuilder` 不改。 | done — commit `98030e6`；lit 无新增 fail；e2e reduce-{sum-3d,axis1}/combo/bcast-multi-axis pass。 |
| **P3a** | §3.5 `computeVectorizedDims`（每个 operand 的向量化迭代维 = reduce 轴 ∪ operand layout 里排在 reduce 轴*之后*的迭代维）+ `selectLoopAxes`：把 `degradeToRowLoop = numParallel≥2 && numR≥1` 这条临时启发式换成"block 轴后面跟着一根落在 `vectorizedDims` 里的 parallel 轴 ⇒ 退化成 step-1 行循环"。**行为不变**（对现有 shape 等价），但原因正确、可推广。`vectorizedDims` 存进 `TilePlan`（GroupEmitter/后续用）。 | `reduce-axis1-e2e`（Case B）、`reduce-sum-3d-e2e`、`combo-elewise-reduce-e2e`、`bcast-multi-axis-e2e` 全 `validation=pass`；现有 lit 全过；新 lit `tile-fuse-vector-reduce-middle.mlir` 仍 pin 住退化形态（已有）。 |
| **P3b** | §4 reduce 三模板里的 `FullLoad`（只在 `enableReductionSplit=true` 时与 `Common` 有区别 —— reduce 轴进 N、永不 tile-split）+ `RCore`（R 轴切多核 + 两阶段 partial→combine + `groups_relations_in`，对标 AF `reduce_schedule_case_generator` —— 这块大，可能独立成阶段）。`classifyAxes` 加 reduce-full-load 变体。 | 新 lit 验 `FullLoad`/`RCore` 选中；RCore e2e（P 小 R 大，如 softmax 形）`validation=pass`。 |
| **P2+P4** | §3.3 block 轴 fuse（fuse 前导 Y 段，**需配套给 operand 发 `tensor.collapse_shape` 把对应维也合掉**，否则 flatten 后的 tile 不是 box）+ `LoopNestBuilder` emit 乘积外层循环 + div/mod 恢复 IV；**且** §5 transpose 双模板：`classifyAxes` X/N + 保留模板的 transpose op lowering + 消除模板（吸 permutation 进 indexing map）+ §3.6 transpose 打分。两者一起做的原因：transpose 是 block-fuse"多 parallel 轴"的主要客户来源（Collapse 已把可合并的合掉了）。给 `Collapse.cpp` 加 transpose 相关禁止合并也在这里。 | 新 example `transpose-elementwise-e2e/run.sh`（尾轴/非尾轴各一个）`validation=pass`；新 lit `tile-fuse-vector-{block-fuse,transpose}.mlir`；现有 example 全过。 |
| **P5** | §3.2 完整笛卡尔积枚举（含 xAxes）+ §3.6 `costEstimate` 占位 + `pickBest` 填实 + §3.2 prune。 | lit 断言几个 shape 下选中的 case；无 e2e 回归。 |
| **P6** | §6 约束 + UB 峰值（简单求和）+ 溢出剪枝 + emit 进 `vector_plan.tiling_infos`。 | lit 断言 `constraints` 出现；一个 example 里超大 `FullLoad` 被正确不选中。 |
| **P7（可选/后续）** | concat / split / gather 的 classifyAxes + schedule case generator（对标 AF 对应文件）。 | 各自 e2e。 |

每阶段一个 commit，`feat(vector-plan): <阶段>`（P1 = `refactor(vector-plan): axis-class scheduler skeleton`），带 Co-author footer。

### 文件表

| 动作 | 路径 | 阶段 |
|---|---|---|
| 改 | `include/Conversion/VectorPlan/TilePlan.h` —— `AxisKind/AxisClass/AxisGrouping`、`TileParam` flag、`TilePlan` 扩展（reduceTemplate/blockFusedAxes/ubTilingAxis*/vectorizedDims/constraints）、`TileConstraint`、`TilePlanDraft` | P1（constraints P6） |
| 改 | `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.{h,cpp}` —— 编排器 + `classifyAxes`+`normGroup`+`enumerateTilingCases`+`blockSplit`+`ubSplit`+`computeVectorizedDims`+`selectLoopAxes`+`pickBest`+`costEstimate`+`estimateMemPeak`（concat/split/gather 留 TODO 函数体） | P1 → P6 |
| 改 | `lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.cpp` —— fuse 乘积外层循环 + IV 恢复 | <br/> |
| 改 | `lib/Conversion/VectorPlan/TileFuse/SliceComputer.cpp` —— N/X 整维 slice、恢复 IV、读 `vectorizedDims` | P2/P3 |
| 改 | `lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp` —— 用 `vectorizedDims` 决定 generic tile rank/位置 | P3 |
| 改 | `lib/Conversion/VectorPlan/TileFuse/Collapse.cpp` —— transpose 禁止合并（TODO: concat/load/gather） | P3 |
| 改 | `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp` —— AR/RA 读 `vectorizedDims`（语义不变，更鲁棒）；transpose lowering hook | P3/P4 |
| 建 | `lib/Conversion/LinalgToAscendC/TransposeConversion.cpp`（或扩展 ComputeConversion）—— AscendC 转置原语（保留模板用） | P4 |
| 建/改 | `lib/Conversion/.../TransposeAbsorb*.cpp`（仿 `BroadcastAbsorb.cpp`）—— 把 transpose 置换吸进下游 indexing map（消除模板用） | P4 |
| 改 | `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` —— emit `constraints` 进 `vector_plan.tiling_infos` | P6 |
| 建 | `test/Conversion/Collapse/tile-fuse-vector-{block-fuse,reduce-fullload,reduce-rcore,transpose}.mlir` | P2–P4 |
| 建 | `examples/{transpose-elementwise-e2e,reduce-rcore-e2e}/{*.mlir,gen_inputs.py,run.sh}` | P3/P4 |
| 改 | 现有 `tile-fuse-vector-{pointwise,reduce,reduce-middle,bcast,...}.mlir` —— 仅当 P2–P4 合法地改了 IR | P2–P4 |

---

## 8. 风险与开放问题

- **score_func 调参** —— 占位规则会选错。缓解：`Common`/preserve 永远是候选且平局偏向；真正调参在 P6 之后（AF 那边也是经验权重 + 编进 host 的 `CalcScore`）。
- **`vectorized_axis` 在多 op 融合下** —— 我们的 group 经 elementwise-fusion 后通常一个 generic，`vectorizedDims` 好算；但若 group 里有 reduce + 后续 elementwise（combo），reduce 输出的 vectorized 维和下游的要一致 —— `computeVectorizedDims` 要对 group 里**所有** member 取并/交。需要在 P3 验 combo e2e。
- **RCore 两阶段 reduce** —— 单块最大；要在我们的 pipeline 里支持 group 间依赖（`groups_relations_in`）+ partial→combine 两个 kernel。可能 P3 工作量超预期 —— 备用方案：先只上 Common+FullLoad，RCore 留候选但不选中，单独排期。
- **transpose 消除模板的前置条件** —— transpose 之上不能有额外向下分支；MLIR 里要检查 `linalg.transpose` 结果的 use 链是否单一。不满足就只剩保留模板。
- **`linalg.transpose` vs 置换的 `linalg.generic`** —— 先只认 named `linalg.transpose`。
- **动态 shape + 约束** —— 完整符号约束求解（AF/ATT）超范围；只记录约束、代入默认 param 值剪枝。

---

## 9. AF 源码 cross-reference（给实现者）

| 本文 § | AF 入口（`autofuse/`下） |
|---|---|
| §0/§3 总流程 | `optimize/autoschedule/autoschedule.cpp` —— `AutoSchedule::{DoAutoSchedule,PrepareTilingCases,GenTilingCase,PruneTilingCase,ProcessOneTilingCase,SelectLoopAxis}`；`optimize/autoschedule/autoschedule.h`（`TilingCase`） |
| §3.1 classifyAxes | `optimize/autoschedule/tiling_group.cpp` —— `TilingGroup::{GenTilingGroup,NormGroup,GenElewiseTilingGroup,GenReduceTilingGroup(+FullLoad),GenTransposeTilingGroup,GenConcatTilingGroup,GenSplitTilingGroup}`、`CalcReduceAxes`；`inc/autoschedule/axis_group.h` |
| §3.3 blockSplit | `optimize/autoschedule/schedule.cpp` —— `Scheduler::{BlockSplit,FuseTileOutAxes,ApplyBlockSplitToNode,ApplyBlockSplit}`；reduce: `ReduceBlockTiling` |
| §3.4 ubSplit | `Scheduler::{TileSplit,TileTiling,ApplyTiling}` |
| §3.5 vectorized/loop axis | `optimize/autoschedule/autoschedule.cpp` —— `FindNotLoopAxis` / `IsNotLoopAxis` / `AutoSchedule::SelectLoopAxis` |
| §3.6 / §4 / §5 多模板 + score | `optimize/autoschedule/template_generator_handler.cpp`；`optimize/task_generator/{reduce,transpose,concat,split}_schedule_case_generator.cpp`、`*_score_function_generator.cpp` |
| §4 reduce 模型 | `optimize/optimize.{h,cpp}`（`ReduceTemplateType`、`IsReduceFirstStage`、`RefreshGroupRelation`、`groups_relations_in`）、`optimize/task_generator/reduce_schedule_case_generator.cpp`、`doc/reduce全流程说明.md` |
| §5 transpose | `optimize/task_generator/transpose_schedule_case_generator.cpp`（`TransposeFusionCaseGenerator::{Generate,TransposeConvertProcess,UpdateAxisByPath}`、`TransposeScoreFunctionGenerator`）、`doc/transpose全流程说明.md` |
| §6 内存 | `doc/dynamic_shape峰值内存约束量化说明.md`、`optimize/buffer_allocate/*` |
| §2 AttAxis flag | `att/gen_model_info/parser/tuning_space.h`（`SubAxis`）、`att/base/model_info.h`（`AttAxis`） |
| 合轴 | `optimize/optimize.cpp` —— `Optimizer::{MergeContinuousAxis,GetNonContinuousAxisPairBySpecialRule}` |
