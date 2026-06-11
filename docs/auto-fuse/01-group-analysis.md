# impl-01: Pass 1 — Group Analysis

**设计依据**: [00-architecture.md](./00-architecture.md) §3, [00-data-model.md](./00-data-model.md) §3  
**前置**: impl-00（数据结构 + pass 骨架已存在，`mlir-opt --auto-fuse-group-analysis /dev/null` 通过）

---

## 定位

`auto-fuse-group-analysis` 是 func-level pass，输入是 linalg-on-tensor func，
输出是每个 linalg op 上的两个 attribute：

```mlir
%0 = linalg.reduce { ... }
     {auto_fuse.group_id = 0 : i32,
      auto_fuse.topo_index = 2 : i32} ins(...) outs(...)
```

Pass 内部使用轻量的 `FusionGroup` 结构（不同于最终的 `GroupInfo`）进行迭代融合，
最终将 group 归属和拓扑位置写入 IR attribute，供 Outline Pass 消费。

**接口边界**：只写 attribute，不修改 IR 结构。非 linalg op 不进任何 group。

---

## 内部数据结构（不对外导出）

```cpp
// AxisLattice.cpp / CanFuse.cpp 共用

enum class FusionKind { None, Vertical, Horizontal };

struct FusionGroup {
  int32_t                            id;
  llvm::SmallVector<linalg::LinalgOp> members;     // 无序；outline 时再按 topo_index 排
  llvm::DenseSet<Value>              boundaryIn;   // group 外输入 tensor
  GroupInfo::Kind                    kind;         // Vector 或 Cube
  llvm::SmallVector<AxisInfo>        canonicalAxes; // 每次 merge 后调 recomputeAxes() 更新
};

struct FusionPair {
  int32_t    g1, g2;    // group id
  FusionKind kind;
  int64_t    score;     // 共享 tensor 字节数
  int        priority;  // 0=DEFAULT (VV), 1=LOW (CV)
};
```

---

## Phase 1: AxisLattice — 上确界推导

文件：`lib/Conversion/AutoFuse/GroupAnalysis/AxisLattice.cpp`

### 算法

对 group 内所有 linalg op 的 `iterator_types`，按位置编号做 join：

```
class_of(member m, axis a):
  若 m 的某个 indexing_map result 含 dim(a)：
    iterator_types[a] == "parallel"  → Parallel
    iterator_types[a] == "reduction" → Reduction
  否则 → Absent

join(Parallel,  Parallel)  = Parallel
join(Parallel,  Reduction) = Reduction
join(Parallel,  Absent)    = Parallel
join(Reduction, Absent)    = Reduction
join(Absent,    Absent)    = Absent  （全 Absent → 过滤掉）
```

### 实现

```cpp
SmallVector<AxisInfo> computeCanonicalAxes(ArrayRef<linalg::LinalgOp> members) {
  // 1. 收集所有 member 的 numLoops 上界
  unsigned maxLoops = 0;
  for (auto op : members)
    maxLoops = std::max(maxLoops, op.getNumLoops());

  // 2. 对每个轴做 join
  SmallVector<AxisRole> roles(maxLoops);
  SmallVector<bool>     initialized(maxLoops, false);

  for (auto op : members) {
    auto iterTypes = op.getIteratorTypesArray();
    for (auto [a, type] : enumerate(iterTypes)) {
      AxisRole cur = (type == utils::IteratorType::reduction)
                       ? AxisRole::Reduction : AxisRole::Parallel;
      if (!initialized[a]) {
        roles[a] = cur;
        initialized[a] = true;
      } else {
        // join：Reduction 吸收 Parallel
        if (cur == AxisRole::Reduction) roles[a] = AxisRole::Reduction;
      }
    }
  }

  // 3. 过滤全 Absent（未被任何 member 初始化的轴）
  SmallVector<AxisInfo> result;
  for (auto [a, init] : enumerate(initialized)) {
    if (!init) continue;
    AxisInfo ax;
    ax.name       = "";             // 可由上层填写
    ax.staticSize = ShapedType::kDynamic; // 可由 shape 查询
    ax.role       = roles[a];
    result.push_back(ax);
  }
  return result;
}
```

### 测试用例

```mlir
// test/Conversion/AutoFuse/group-analysis-axis-lattice.mlir
// RUN: mlir-opt --auto-fuse-group-analysis %s | FileCheck %s

func.func @reduce_then_pointwise(%in: tensor<16x32xf16>) -> tensor<16xf16> {
  // reduce: iterator_types = [parallel, reduction]
  // add:    iterator_types = [parallel]
  // 合并后：axis 0 = Parallel, axis 1 = Reduction
  %0 = linalg.reduce { arith.addf } ins(%in: tensor<16x32xf16>)
                       outs(%init: tensor<16xf16>) dimensions = [1]
  %1 = linalg.generic {
    indexing_maps = [...], iterator_types = ["parallel"]
  } ins(%0: tensor<16xf16>) outs(%out: tensor<16xf16>) { ... }
  return %1
}
// CHECK: auto_fuse.group_id = 0
// CHECK: auto_fuse.group_id = 0
// CHECK: auto_fuse.topo_index = 0
// CHECK: auto_fuse.topo_index = 1
```

---

## Phase 2: CanFuse — 融合判断

文件：`lib/Conversion/AutoFuse/GroupAnalysis/CanFuse.cpp`

### getFusionKind

```cpp
FusionKind getFusionKind(const FusionGroup &g1, const FusionGroup &g2) {
  // Vertical：g1 某 member 的 result 被 g2 某 member 使用（或反向）
  auto hasEdge = [](const FusionGroup &src, const FusionGroup &dst) {
    for (auto op : src.members)
      for (Value result : op->getResults())
        for (Operation *user : result.getUsers())
          if (dst.containsOp(user)) return true;
    return false;
  };
  if (hasEdge(g1, g2) || hasEdge(g2, g1)) return FusionKind::Vertical;

  // Horizontal：共享至少一个 boundary input tensor
  for (Value v : g1.boundaryIn)
    if (g2.boundaryIn.contains(v)) return FusionKind::Horizontal;

  return FusionKind::None;
}
```

### canFuseVector（7 条规则）

```cpp
struct CanFuseOptions {
  int32_t maxReduceEpilogueOps     = 3;
  int32_t maxHorizontalExtraInputs = 4;
};

/// 检查 g1 或 g2 中是否包含 linalg.transpose，或两者链接 tensor 的
/// indexing_map 会在合并后产生 B2 访问模式。
bool hasTransposePattern(const FusionGroup &g1, const FusionGroup &g2) {
  auto hasTranspose = [](const FusionGroup &g) {
    return llvm::any_of(g.members, [](linalg::LinalgOp op) {
      return isa<linalg::TransposeOp>(op);
    });
  };
  return hasTranspose(g1) || hasTranspose(g2);
}

bool canFuseVector(FusionGroup &g1, FusionGroup &g2,
                   FusionKind kind, ArrayRef<FusionGroup> allGroups,
                   const CanFuseOptions &opts) {
  // 规则 1：kind 已确定（None 在调用前已过滤）

  // 规则 2：cycle check
  // 建临时 DAG：把 g1+g2 合并为单节点，检查剩余 group 中是否有路径
  // merge_group → X → merge_group 的回路
  if (wouldCreateCycle(g1, g2, allGroups)) return false;

  // 规则 3：轴类相容
  auto merged = computeCanonicalAxes(concat(g1.members, g2.members));
  auto tileable  = getAxesByRole(merged, AxisRole::Parallel);
  auto reduction = getAxesByRole(merged, AxisRole::Reduction);
  if (tileable.empty()) return false;
  // Tileable ∩ Reduction = ∅ 由 join 规则定义保证

  // 规则 4：链接 tensor 的所有 Vector-track 消费者都在 g1 或 g2 内
  // 仅 Vertical pair 有实质约束
  if (kind == FusionKind::Vertical) {
    for (Value linkTensor : getLinkTensors(g1, g2)) {
      for (Operation *user : linkTensor.getUsers()) {
        if (!isa<linalg::LinalgOp>(user)) continue;
        if (!g1.containsOp(user) && !g2.containsOp(user)) return false;
      }
    }
  }

  // 规则 5：DPS init 透明（linalg.fill 作为 linalg.reduce 的 outs 产生者，
  //          按普通成员纳入 group；tensor.empty 不进 group）
  // → 无需额外检查，fill 是 linalg op，会自然进入 group

  // 规则 6：epilogue reduction 依赖约束（含 reduction 时才检查）
  if (!reduction.empty()) {
    auto reductionOps = getReductionOps(g1, g2);
    for (auto epi : getEpilogueOps(g1, g2, reductionOps)) {
      for (Value in : epi.getInputOperands()) {
        // epilogue 的输入不能是 reduction op 的 partial sum
        if (isPartialSum(in, reductionOps)) return false;
      }
    }
    // 规则 6 附加：epilogue 成员数量上限（仅 enableReductionSplit 时检查）
    if (countEpilogueOps(g1, g2, reductionOps) > opts.maxReduceEpilogueOps)
      return false;
  }

  // 规则 7：transpose 约束（为 Collapse 阶段 B2 可行性服务）
  // 含 linalg.transpose 或产生 B2 访问模式的 group，只允许与全 parallel
  // (pointwise) op 融合。原因：Collapse 的 B2 处理（Variant 1 插 transpose /
  // Variant 2 改 map）假设 G 内轴全为同类型；若 transpose 与 reduction op
  // 融合，collapse 候选组 G 可能跨 Parallel/Reduction 边界，导致 B2 fixup
  // 语义错误（transpose 重排的轴与 reduction 累加的轴交叉）。
  if (hasTransposePattern(g1, g2)) {
    auto merged = computeCanonicalAxes(concat(g1.members, g2.members));
    if (!getAxesByRole(merged, AxisRole::Reduction).empty())
      return false;
  }

  // Horizontal 额外规则
  if (kind == FusionKind::Horizontal) {
    // H1：merged_Tileable 非空（规则 3 已保证）

    // H2：新增 boundary input 数量
    int extra = countNewBoundaryInputs(g1, g2);
    if (extra > opts.maxHorizontalExtraInputs) return false;
  }

  return true;
}
```

### canFuseCube（Epilogue：CubeGroup ← VectorGroup）

```cpp
bool canFuseCubeEpilogue(const FusionGroup &cube, const FusionGroup &vec) {
  // E1：vec 内所有 op 的 iterator_types 全为 parallel
  for (auto op : vec.members)
    if (hasReductionIterator(op)) return false;

  // E2：无 in-place 写或 aliasing（DPS outs 不与 ins alias）
  for (auto op : vec.members)
    if (hasDPSAlias(op)) return false;

  // E3：每个 op 的 indexing_map 对 matmul 输出轴是 affine projective
  //   允许：identity / broadcast（rank-drop）
  //   禁止：permutation（M/N 轴混合）
  auto matmul = getMatmulOp(cube);
  for (auto op : vec.members)
    if (!isAffineProjective(op, matmul)) return false;

  // E4：matmul 输出到 vec anchor 路径上无 expand_shape / linalg.broadcast view node
  if (hasViewNodeOnPath(matmul, vec)) return false;

  // E5：matmul 输出只被一个 VectorGroup 消费（fan-out 检查）
  if (countVectorGroupConsumers(matmul, allGroups) > 1) return false;

  // vec 不能全为纯 shape/view op
  if (allShapeOps(vec)) return false;

  return true;
}
```

### canFuseCube（Prologue：VectorGroup → CubeGroup）

```cpp
bool canFuseCubePrologue(const FusionGroup &vec, const FusionGroup &cube) {
  // P1：vec 内所有 op 的 iterator_types 全为 parallel
  for (auto op : vec.members)
    if (hasReductionIterator(op)) return false;

  // P2：无 in-place 写或 aliasing
  for (auto op : vec.members)
    if (hasDPSAlias(op)) return false;

  // P3：单消费者约束：vec 的每个输出只被 matmul 或 group 内其他成员消费
  auto matmul = getMatmulOp(cube);
  for (auto op : vec.members)
    for (Value result : op->getResults())
      for (Operation *user : result.getUsers())
        if (user != matmul && !vec.containsOp(user)) return false;

  // P4：indexing_map 对 A/B 的输入轴是 affine projective
  for (auto op : vec.members)
    if (!isAffineProjectiveForMatmulInput(op, matmul)) return false;

  return true;
}
```

### score 计算

```cpp
int64_t computeScore(const FusionGroup &g1, const FusionGroup &g2) {
  int64_t bytes = 0;
  for (Value v : getLinkTensors(g1, g2)) {
    auto type = cast<RankedTensorType>(v.getType());
    int64_t elems = 1;
    for (int64_t d : type.getShape())
      elems *= (d == ShapedType::kDynamic ? 1 : d);
    bytes += elems * type.getElementTypeBitWidth() / 8;
  }
  return bytes;
}
```

### 测试用例

```mlir
// test/Conversion/AutoFuse/group-analysis-can-fuse.mlir
// RUN: mlir-opt --auto-fuse-group-analysis %s | FileCheck %s

// ---- 不应融合：规则 4，链接 tensor 有 group 外消费者 ----
func.func @no_fuse_fanout(%in: tensor<8xf16>, %extra_consumer_sink: tensor<8xf16>)
    -> (tensor<8xf16>, tensor<8xf16>) {
  %mid = linalg.generic { ... } ins(%in) outs(...) // op A
  %out1 = linalg.generic { ... } ins(%mid) outs(...) // op B，消费 mid
  // %mid 也被 group 外消费：返回给调用方
  return %out1, %mid
}
// CHECK: auto_fuse.group_id = [[A:[0-9]+]]
// CHECK: auto_fuse.group_id = [[B:[0-9]+]]
// CHECK-NOT: group_id = [[A]]
// （A 和 B group_id 不同，表示未融合）

// ---- 应融合：horizontal，共享 boundary input ----
func.func @horizontal_fuse(%x: tensor<8xf16>, %y1: tensor<8xf16>, %y2: tensor<8xf16>)
    -> (tensor<8xf16>, tensor<8xf16>) {
  %s1 = linalg.generic { ... } ins(%x, %y1) outs(...)  // sibling 1
  %s2 = linalg.generic { ... } ins(%x, %y2) outs(...)  // sibling 2
  return %s1, %s2
}
// CHECK: {auto_fuse.group_id = [[G:[0-9]+]]
// CHECK: {auto_fuse.group_id = [[G]]
```

---

## Phase 3: GroupAnalysisPass — 主循环

文件：`lib/Conversion/AutoFuse/GroupAnalysis/GroupAnalysisPass.cpp`

### 算法

```
1. 初始化：每个 linalg op 各为单成员 FusionGroup
2. 迭代融合（多轮，直到无进展）：
   a. getFusablePairs：对所有 group pair 调 getFusionKind + canFuse
   b. 按 priority 分桶（DEFAULT=VV 先，LOW=CV 后）
   c. 在同一 priority 内按 score 降序 merge
   d. merge 后更新 boundaryIn + 重调 recomputeAxes
3. 全局 topo_index 分配：按 func.walk 的 SSA 程序序
4. 写 attribute：group_id + topo_index
```

### 实现骨架

```cpp
void runOnOperation() override {
  func::FuncOp func = getOperation();
  OpBuilder builder(func.getContext());

  // 1. 初始化
  llvm::SmallVector<FusionGroup> groups;
  func.walk([&](linalg::LinalgOp op) {
    FusionGroup g;
    g.id = groups.size();
    g.members = {op};
    g.kind = isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)
              ? GroupInfo::Kind::Cube : GroupInfo::Kind::Vector;
    g.boundaryIn = collectBoundaryIn({op}, func);
    g.canonicalAxes = computeCanonicalAxes({op});
    groups.push_back(std::move(g));
  });

  // 2. 迭代融合
  CanFuseOptions opts{maxReduceEpilogueOps, maxHorizontalExtraInputs};
  bool changed = true;
  while (changed) {
    changed = false;
    auto pairs = getFusablePairs(groups, opts);
    // priority: DEFAULT(0) 先处理
    llvm::stable_sort(pairs, [](const FusionPair &a, const FusionPair &b) {
      return a.priority < b.priority || (a.priority == b.priority && a.score > b.score);
    });
    for (auto &pair : pairs) {
      auto *g1 = findGroup(groups, pair.g1);
      auto *g2 = findGroup(groups, pair.g2);
      if (!g1 || !g2) continue; // 已被之前的 merge 消灭
      mergeGroups(groups, *g1, *g2);
      changed = true;
    }
  }

  // 3. topo_index：按 func.walk SSA 程序序
  llvm::DenseMap<linalg::LinalgOp, int32_t> topoIndex;
  int32_t idx = 0;
  func.walk([&](linalg::LinalgOp op) { topoIndex[op] = idx++; });

  // 4. group_id 反查表
  llvm::DenseMap<linalg::LinalgOp, int32_t> opToGroup;
  for (auto &g : groups)
    for (auto op : g.members)
      opToGroup[op] = g.id;

  // 5. 写 attribute
  func.walk([&](linalg::LinalgOp op) {
    op->setAttr("auto_fuse.group_id",
                builder.getI32IntegerAttr(opToGroup[op]));
    op->setAttr("auto_fuse.topo_index",
                builder.getI32IntegerAttr(topoIndex[op]));
  });
}
```

---

## 完整测试场景

### 场景 1：单 op

```mlir
// test/Conversion/AutoFuse/group-analysis-single.mlir
// RUN: mlir-opt --auto-fuse-group-analysis %s | FileCheck %s
func.func @single(%x: tensor<8xf16>) -> tensor<8xf16> {
  %out = linalg.generic { ... } ins(%x) outs(...) { ... }
  return %out
}
// CHECK: auto_fuse.group_id = 0
// CHECK: auto_fuse.topo_index = 0
```

### 场景 2：reduce→pointwise（Vertical fusion）

```mlir
// test/Conversion/AutoFuse/group-analysis-vertical.mlir
// RUN: mlir-opt --auto-fuse-group-analysis %s | FileCheck %s
func.func @reduce_pointwise(%in: tensor<4x8xf16>) -> tensor<4xf16> {
  %r = linalg.reduce { arith.addf } ins(%in) outs(...) dimensions = [1]
  %out = linalg.generic { ... } ins(%r) outs(...)
  return %out
}
// CHECK: {auto_fuse.group_id = [[G:[0-9]+]], auto_fuse.topo_index = 0
// CHECK: {auto_fuse.group_id = [[G]], auto_fuse.topo_index = 1
```

### 场景 3：LayerNorm skip connection（多轮迭代验证）

```mlir
// test/Conversion/AutoFuse/group-analysis-layernorm.mlir
// RUN: mlir-opt --auto-fuse-group-analysis %s | FileCheck %s
//
// input → mean → sub → square → mean2 → sqrt → div → output
//                └──────────────────────────────────↗
//
// Round 1：{sqrt, mean2, square, div} → G
// Round 2：sub 的两个消费者（square 和 div）均在 G → sub 融入 G
func.func @layernorm(%input: tensor<8x16xf16>) -> tensor<8x16xf16> {
  // ... (6 个 linalg op，含 skip connection)
}
// CHECK-COUNT-6: auto_fuse.group_id = [[G:[0-9]+]]
// （所有 6 个 op 在同一 group）
```

### 场景 4：matmul + epilogue（CubeGroup epilogue fusion，LOW priority）

```mlir
// test/Conversion/AutoFuse/group-analysis-cube-epilogue.mlir
// RUN: mlir-opt --auto-fuse-group-analysis %s | FileCheck %s
func.func @matmul_bias(%A: tensor<32x64xf16>, %B: tensor<64x32xf16>,
                        %bias: tensor<32xf16>) -> tensor<32x32xf16> {
  %mm = linalg.matmul ins(%A, %B) outs(...)
  %add = linalg.generic { ... } ins(%mm, %bias) outs(...)  // epilogue
  return %add
}
// linalg.matmul 和 linalg.generic 应在同一 group，kind 为 Cube
// CHECK: {auto_fuse.group_id = [[G:[0-9]+]]
// CHECK: {auto_fuse.group_id = [[G]]
```

### 场景 5：不应融合的情况

```mlir
// test/Conversion/AutoFuse/group-analysis-no-fuse.mlir
// RUN: mlir-opt --auto-fuse-group-analysis %s | FileCheck %s

// 规则 6：CumSum 类（epilogue 依赖 reduction partial sum）→ 不融合
func.func @cumsum_no_fuse(%in: tensor<8xf16>) -> tensor<8xf16> {
  // partial sum 被 epilogue 直接依赖（不是 post-reduction 结果）
}
// CHECK: auto_fuse.group_id = [[A:[0-9]+]]
// CHECK: auto_fuse.group_id = [[B:[0-9]+]]
// CHECK-NOT: group_id = [[A]]  ← B 与 A 不同
```

### 场景 6：transpose + reduce 不应融合（规则 7）

```mlir
// test/Conversion/AutoFuse/group-analysis-transpose-no-fuse.mlir
// RUN: mlir-opt --auto-fuse-group-analysis %s | FileCheck %s

// transpose 的输出被 reduce 消费，但二者不应融合：
// 若融合，collapse 候选组 G 会同时包含被 transpose 重排的轴和 reduce 累加的轴，
// 导致 B2 fixup 语义错误。
func.func @transpose_reduce_no_fuse(%in: tensor<8x16xf16>) -> tensor<16xf16> {
  %t = linalg.transpose ins(%in: tensor<8x16xf16>)
       outs(%init_t: tensor<16x8xf16>) permutation = [1, 0]
  %r = linalg.reduce { arith.addf } ins(%t: tensor<16x8xf16>)
                       outs(%init_r: tensor<16xf16>) dimensions = [1]
  return %r
}
// transpose 和 reduce 应在不同 group
// CHECK: auto_fuse.group_id = [[T:[0-9]+]]
// CHECK: auto_fuse.group_id = [[R:[0-9]+]]
// CHECK-NOT: group_id = [[T]]  ← R 与 T 不同

// 对比：transpose + pointwise 应正常融合
func.func @transpose_pointwise_fuse(%in: tensor<8x16xf16>,
                                     %bias: tensor<16x8xf16>) -> tensor<16x8xf16> {
  %t = linalg.transpose ins(%in: tensor<8x16xf16>)
       outs(%init_t: tensor<16x8xf16>) permutation = [1, 0]
  %out = linalg.generic {
    indexing_maps = [affine_map<(d0,d1) -> (d0,d1)>,
                     affine_map<(d0,d1) -> (d0,d1)>,
                     affine_map<(d0,d1) -> (d0,d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%t, %bias) outs(%init_out: tensor<16x8xf16>) { ^bb0(%a,%b,%c): ... }
  return %out
}
// transpose 和 pointwise 应在同一 group
// CHECK: {auto_fuse.group_id = [[G2:[0-9]+]]
// CHECK: {auto_fuse.group_id = [[G2]]
```
