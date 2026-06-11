# impl-02: Outline Pass — Group → Kernel Func

**设计依据**: [00-architecture.md](./00-architecture.md) §2, [00-data-model.md](./00-data-model.md) §4  
**前置**: impl-01（每个 linalg op 已有 `auto_fuse.group_id` / `auto_fuse.topo_index`）

---

## 定位

`auto-fuse-group-outline` 是 module-level pass，消费 Pass 1 写的 attribute，
把每个 fusion group 的 op 提取为独立的 `@kernel_groupN` func，
并把原 func 改写为 coordinator（顺序 call 各 kernel func）。
最后 strip 所有 `auto_fuse.*` attribute，写出两类文件：

```
network.mlir           coordinator func + 所有 kernel func 的 private 声明
kernel_group0.mlir     每个 group 一个文件，含 kernel func 定义（干净的 linalg-on-tensor）
kernel_group1.mlir
...
```

Pass 2 的输入（`kernel_groupN.mlir`）不含任何 `auto_fuse.*` attribute。

**接口边界**：
- 输入：带 `auto_fuse.group_id` / `auto_fuse.topo_index` 的 ModuleOp
- 输出：ModuleOp（coordinator + kernel func 声明）；可选写文件
- attribute strip 在输出写入前完成

---

## Phase 1: 分桶 + 两级拓扑排序（Step 1–3）

文件：`lib/Conversion/AutoFuse/GroupOutline/GroupOutlinePass.cpp`

```cpp
// Step 1：分桶
llvm::DenseMap<int32_t, SmallVector<linalg::LinalgOp>> buckets;
module.walk([&](linalg::LinalgOp op) {
  auto gid = op->getAttrOfType<IntegerAttr>("auto_fuse.group_id");
  if (!gid) return; // 非 linalg op 或未标注（不应出现）
  buckets[gid.getInt()].push_back(op);
});

// Step 2：组内按 topo_index 排序
for (auto &[gid, ops] : buckets) {
  llvm::stable_sort(ops, [](linalg::LinalgOp a, linalg::LinalgOp b) {
    return getTopoIndex(a) < getTopoIndex(b);
  });
}

// Step 3：组间按 representative（min topo_index）排序
// 保证 producer group 总在 consumer group 之前
SmallVector<int32_t> sortedGroupIds;
for (auto &[gid, _] : buckets) sortedGroupIds.push_back(gid);
llvm::sort(sortedGroupIds, [&](int32_t a, int32_t b) {
  return getMinTopoIndex(buckets[a]) < getMinTopoIndex(buckets[b]);
});
```

**正确性保证**：若 Group A 的某 op 是 Group B 的 producer，
则 A op 的 topo_index < B op 的 topo_index（Pass 1 按 SSA def 顺序分配），
因此 representative(A) < representative(B)，组间排序正确。

### 测试用例（Phase 1）

```mlir
// test/Conversion/AutoFuse/group-outline-sort.mlir
// RUN: mlir-opt --auto-fuse-group-analysis --auto-fuse-group-outline %s \
// RUN:   | FileCheck %s --check-prefix=OUTLINE

func.func @two_groups(%x: tensor<8xf16>, %y: tensor<8xf16>) -> tensor<8xf16> {
  %a = linalg.generic { ... } ins(%x) outs(...)
  %b = linalg.generic { ... } ins(%a, %y) outs(...)
  return %b
}
// OUTLINE:      func.func private @kernel_group[[G0:[0-9]+]]
// OUTLINE-NEXT: func.func private @kernel_group[[G1:[0-9]+]]
// OUTLINE:      func.func @two_groups
// OUTLINE:        call @kernel_group[[G0]]
// OUTLINE-NEXT:   call @kernel_group[[G1]]
```

---

## Phase 2: 重建 GroupInfo（Step 4）

在 IR 中无 GroupInfo 存储，Outline Pass 从 attribute + def-use 现场重建：

```cpp
struct RebuiltGroupInfo {
  int32_t                            id;
  SmallVector<linalg::LinalgOp>      topoMembers; // 已按 topo_index 排序
  GroupInfo::Kind                    kind;
  SmallVector<AxisInfo>              canonicalAxes;
  SmallVector<Value>                 boundaryIn;
  SmallVector<Value>                 boundaryOut;
};

RebuiltGroupInfo rebuildGroupInfo(int32_t gid,
                                   ArrayRef<linalg::LinalgOp> topoMembers) {
  RebuiltGroupInfo info;
  info.id          = gid;
  info.topoMembers = topoMembers;

  // kind：含 matmul op → Cube
  info.kind = llvm::any_of(topoMembers, [](linalg::LinalgOp op) {
    return isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op);
  }) ? GroupInfo::Kind::Cube : GroupInfo::Kind::Vector;

  // canonicalAxes：对所有成员按上确界推导（复用 AxisLattice 逻辑）
  info.canonicalAxes = computeCanonicalAxes(topoMembers);

  // boundaryIn：被 group 内 op 使用、但定义在 group 外的 Value
  DenseSet<Value> memberResults;
  for (auto op : topoMembers)
    for (Value r : op->getResults())
      memberResults.insert(r);

  DenseSet<Value> seen;
  for (auto op : topoMembers) {
    for (Value operand : op->getOperands()) {
      if (!memberResults.count(operand) && !seen.count(operand)) {
        info.boundaryIn.push_back(operand);
        seen.insert(operand);
      }
    }
  }

  // boundaryOut：group 内定义、被 group 外 op 使用的 Value
  DenseSet<Operation *> memberOps(topoMembers.begin(), topoMembers.end());
  for (auto op : topoMembers) {
    for (Value result : op->getResults()) {
      bool usedOutside = llvm::any_of(result.getUsers(), [&](Operation *user) {
        return !memberOps.count(user);
      });
      if (usedOutside) info.boundaryOut.push_back(result);
    }
  }

  return info;
}
```

---

## Phase 3: outlineGroup — 创建 kernel func（Step 5–6）

```cpp
func::FuncOp outlineGroup(OpBuilder &builder, ModuleOp module,
                            const RebuiltGroupInfo &info,
                            StringRef funcName) {
  // 1. 构造 FunctionType
  SmallVector<Type> argTypes, resTypes;
  for (Value v : info.boundaryIn)  argTypes.push_back(v.getType());
  for (Value v : info.boundaryOut) resTypes.push_back(v.getType());
  auto funcType = builder.getFunctionType(argTypes, resTypes);

  // 2. 创建 kernel func（插在 module 开头）
  builder.setInsertionPointToStart(module.getBody());
  auto kernelFunc = builder.create<func::FuncOp>(
      module.getLoc(), funcName, funcType);
  Block *body = kernelFunc.addEntryBlock();

  // 3. value 映射：boundaryIn → block args
  IRMapping mapping;
  for (auto [orig, arg] : llvm::zip(info.boundaryIn, body->getArguments()))
    mapping.map(orig, arg);

  // 4. 按 topoMembers 顺序 clone op 进 body
  builder.setInsertionPointToEnd(body);
  for (linalg::LinalgOp op : info.topoMembers)
    builder.clone(*op, mapping);

  // 5. 插入 return
  SmallVector<Value> returnVals;
  for (Value v : info.boundaryOut)
    returnVals.push_back(mapping.lookupOrDefault(v));
  builder.create<func::ReturnOp>(module.getLoc(), returnVals);

  // 6. 在原 func 中：删除 group 内 op，替换为 func.call
  //    （replaceGroupWithCall 中处理）

  return kernelFunc;
}

void replaceGroupWithCall(OpBuilder &builder, func::FuncOp origFunc,
                           const RebuiltGroupInfo &info,
                           func::FuncOp kernelFunc) {
  // 在最后一个 topoMember 之前插入 call
  builder.setInsertionPoint(info.topoMembers.back());
  auto callOp = builder.create<func::CallOp>(
      origFunc.getLoc(), kernelFunc, info.boundaryIn);

  // 替换 boundaryOut 的使用
  for (auto [origOut, callResult] :
       llvm::zip(info.boundaryOut, callOp.getResults()))
    origOut.replaceAllUsesWith(callResult);

  // 删除 group 内 op（逆拓扑序）
  for (auto op : llvm::reverse(info.topoMembers))
    op->erase();
}
```

### 测试用例（Phase 3）

```mlir
// test/Conversion/AutoFuse/group-outline-basic.mlir
// RUN: mlir-opt --auto-fuse-group-analysis --auto-fuse-group-outline %s \
// RUN:   | FileCheck %s

func.func @single_group(%x: tensor<8xf16>, %bias: tensor<8xf16>) -> tensor<8xf16> {
  %r = linalg.reduce { arith.addf } ins(%x: tensor<8xf16>)
                       outs(%init: tensor<8xf16>) dimensions = [1]
  %out = linalg.generic { ... } ins(%r, %bias) outs(...)
  return %out
}

// 原 func 变为 coordinator
// CHECK:      func.func private @kernel_group0(%arg0: tensor<8xf16>, %arg1: tensor<8xf16>)
// CHECK-SAME:     -> tensor<8xf16>
// CHECK:      func.func @single_group
// CHECK:        %[[R:.*]] = func.call @kernel_group0
// CHECK:        return %[[R]]

// kernel func 含 linalg op，不含 auto_fuse.* attribute
// CHECK:      func.func private @kernel_group0
// CHECK:        linalg.reduce
// CHECK:        linalg.generic
// CHECK-NOT:    auto_fuse.group_id
```

---

## Phase 4: Attribute Strip + 文件分离（Step 7）

### Attribute Strip

```cpp
void stripAutoFuseAttrs(ModuleOp module) {
  module.walk([](linalg::LinalgOp op) {
    op->removeAttr("auto_fuse.group_id");
    op->removeAttr("auto_fuse.topo_index");
  });
}
```

### 文件分离（split pass / outputDir 非空时执行）

```cpp
void emitFiles(ModuleOp module, ArrayRef<int32_t> sortedGroupIds,
               StringRef outputDir) {
  // network.mlir：coordinator func + 所有 kernel func 的 private 声明
  // kernel_groupN.mlir：每个 group 对应的 kernel func 定义

  // 实现：
  // 1. 对每个 kernel func，建一个只含该 func 的临时 ModuleOp
  // 2. 在 network.mlir 中把 kernel func 改为 private 声明（只保留类型，无 body）
  // 3. 分别序列化写出

  for (int32_t gid : sortedGroupIds) {
    std::string filename = (outputDir + "/kernel_group" + Twine(gid) + ".mlir").str();
    auto kernelFunc = findKernelFunc(module, gid);

    // 创建独立 module，clone kernel func
    OpBuilder b(module.getContext());
    auto subModule = b.create<ModuleOp>(module.getLoc());
    b.setInsertionPointToStart(subModule.getBody());
    b.clone(*kernelFunc);

    // 写文件
    std::error_code ec;
    llvm::raw_fd_ostream os(filename, ec);
    subModule.print(os);

    // 把原 module 中的 kernel func 改为 private 声明（无 body）
    kernelFunc.eraseBody();
    kernelFunc.setPrivate();
  }

  // 写 network.mlir
  std::string netFile = (outputDir + "/network.mlir").str();
  std::error_code ec2;
  llvm::raw_fd_ostream osNet(netFile, ec2);
  module.print(osNet);
}
```

### 测试用例（Phase 4）

```mlir
// test/Conversion/AutoFuse/group-outline-strip.mlir
// RUN: mlir-opt --auto-fuse-group-analysis --auto-fuse-group-outline %s \
// RUN:   | FileCheck %s

func.func @two_ops(%x: tensor<8xf16>) -> tensor<8xf16> {
  %a = linalg.generic { ... } ins(%x) outs(...)
  %b = linalg.generic { ... } ins(%a) outs(...)
  return %b
}

// attribute strip 验证：输出中不含任何 auto_fuse.* attribute
// CHECK-NOT: auto_fuse.group_id
// CHECK-NOT: auto_fuse.topo_index

// coordinator 中只有 call，无 linalg op
// CHECK:      func.func @two_ops
// CHECK-NOT:    linalg.generic
// CHECK:        func.call
```

---

## 完整流水线验收测试

```mlir
// test/Conversion/AutoFuse/group-outline-pipeline.mlir
// RUN: mlir-opt --auto-fuse-group-analysis --auto-fuse-group-outline %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,NET

func.func @layernorm(%input: tensor<4x8xf16>, %scale: tensor<8xf16>,
                      %bias: tensor<8xf16>) -> tensor<4x8xf16> {
  // ... 6 个 linalg op，Pass 1 应合并为同一 group
  return %out
}

// NET: func.func private @kernel_group0
// NET-SAME:   (tensor<4x8xf16>, tensor<8xf16>, tensor<8xf16>) -> tensor<4x8xf16>

// coordinator 只含 call
// NET: func.func @layernorm
// NET:   call @kernel_group0

// kernel func 含完整 linalg op，无 attribute
// CHECK: func.func private @kernel_group0
// CHECK:   linalg.reduce
// CHECK:   linalg.generic
// CHECK-NOT: auto_fuse.
```
