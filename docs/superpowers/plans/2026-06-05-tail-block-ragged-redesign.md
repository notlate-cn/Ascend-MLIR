# 尾块 ragged 重设计 实现计划(mode ① f16 不对齐尾)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 AscendC tile-fuse 的 overlap-tail 换成单一 ragged 尾块,让 f16 / 不整除
尾巴在原生 codegen 下正确,并放宽 `Divides{32}` 拒绝约束;以 `reduce-sum-3d-f16-tail-e2e`
从 XFAIL 转 PASS 为验收。

**Architecture:** 采用 spec 的 "X 先行" 策略 —— 尾块 `scf.if` 改用真实 IV
(`outerOfTailIV + mainInnerUb`)+ `sizeOverride = remaining - mainInnerUb`,让 slice /
compute / buffer / copy 全走真实 `rem`;给尾块 `scf.if` 打 `afir.ragged_tail` 标记,
`CannTranslation` 据此把尾块 GM↔UB 的 load/store 路由到 `DataCopyPad`(吃不对齐)。
shadow-alloc 是否出现由生成 IR + e2e 裁判,出了再升级到 spec 的 Y(解耦静态-T)。

**Tech Stack:** MLIR(scf/linalg/tensor)、AscendC(DataCopy / DataCopyPad)、
afir-opt / afir-translate / runtime-session(sim)、lit + FileCheck。

**范围说明:** 本计划只覆盖 spec 的 **mode ①(parallel 轴 f16 不对齐尾,已触发)**
+ §4.4 约束放宽 + 回归。spec 的 **mode ②(reduction 轴尾巴,潜在未触发)** 改动在
`emitGroupWithReductionSplit`,与本计划独立、可单独交付,放到后续计划
`2026-06-XX-tail-block-reduction-axis.md`(见文末"后续")。

---

## 文件结构

| 文件 | 职责 | 改动 |
|------|------|------|
| `lib/Conversion/AutoFuse/TileFuse/GroupEmitter.cpp` | 尾块 `scf.if` emission | 改 overlap → ragged(IV+sizeOverride)、打 `afir.ragged_tail` 标记 |
| `lib/Target/CannKernel/CannTranslation.cpp` | AscendC 代码翻译 | 尾块 GM↔UB load/store 路由到 DataCopyPad |
| `lib/Conversion/AutoFuse/TileFuse/TilePlanBuild.cpp` | tile 约束生成 | 删 `Divides{32, (extent-INNER)*elemBytes}` 硬约束 |
| `test/Conversion/AutoFuseCodegen/tail-peel.mlir` | 尾块 IR 形状 lit | 改 CHECK 为 ragged 形状 |
| `examples/reduce-sum-3d-f16-tail-e2e/run.sh` | f16 e2e gate | 去掉早退,启用 sim 验证 |

---

## Task 1: 尾块改 ragged(honest IV + sizeOverride + 标记)

**Files:**
- Modify: `lib/Conversion/AutoFuse/TileFuse/GroupEmitter.cpp:728-770`
- Test: `test/Conversion/AutoFuseCodegen/tail-peel.mlir`

- [ ] **Step 1: 改 lit 的 CHECK 为 ragged 形状(先让它失败)**

把 `test/Conversion/AutoFuseCodegen/tail-peel.mlir` 第 53-72 行(`Tail body: overlap-tail ...`
到文件末)整段替换为下面的 ragged 期望。其余(第 1-52 行)不动。

```
// Tail body: ragged-tail uses the HONEST offset `outer_iv + mainInnerUb` and
// the HONEST size `remaining - mainInnerUb` (NOT `extent - T` / static `%arg3`).
// No overlap, no recompute.
//
// CHECK: %[[COND:.+]] = arith.cmpi slt, %[[MAIN_UB]], %[[REMAINING]]
// CHECK: %[[TAILRES:.+]] = scf.if %[[COND]]
//
// honest tail size = remaining - mainInnerUb
// CHECK: %[[TAILSZ:.+]] = arith.subi %[[REMAINING]], %[[MAIN_UB]]
// honest tail offset = outer_iv + mainInnerUb
// CHECK: %[[TAILOFF:.+]] = arith.addi %{{.*}}, %[[MAIN_UB]]
//
// extract_slice carries honest offset + honest (dynamic) size.
// CHECK: tensor.extract_slice %{{.*}}[%[[TAILOFF]]] [%[[TAILSZ]]]
// CHECK: linalg.generic
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[%[[TAILOFF]]] [%[[TAILSZ]]]
// CHECK: scf.yield
//
// else branch yields the inner-for results unchanged.
// CHECK: } else {
// CHECK: scf.yield
// CHECK: }
```

同时把第 7-8 行注释从 overlap 描述改成 ragged:

```
//   - scf.if then-block uses HONEST size `remaining - mainInnerUb` at offset
//     `outer_iv + mainInnerUb` (ragged-tail, NOT overlap `extent - T`)
```

- [ ] **Step 2: 跑 lit 确认失败**

Run: `build/bin/llvm-lit -v test/Conversion/AutoFuseCodegen/tail-peel.mlir`
Expected: FAIL —— 当前代码发的是 `arith.subi %.., %arg3` + `[%off][%arg3]`(静态 T),
匹配不到 `arith.subi %[[REMAINING]], %[[MAIN_UB]]` 与动态 size。

- [ ] **Step 3: 改 GroupEmitter 尾块 emission**

在 `GroupEmitter.cpp`,把第 728-729 创建 `tailIf` 之后**立即**加一行标记;并把第 753-769
的 overlap 逻辑替换为 ragged。

标记(紧跟 `tailIf = builder.create<scf::IfOp>(...)` 之后,`{` 块外):

```cpp
    tailIf = builder.create<scf::IfOp>(loc, resultTypes, cond,
                                        /*withElseRegion=*/true);
    // Mark this scf.if as the ragged tail so CannTranslation routes its
    // GM↔UB DataCopy to DataCopyPad (unaligned f16 offset/length legal).
    tailIf->setAttr("afir.ragged_tail", builder.getUnitAttr());
```

替换第 753-769(`Value tailComposed ...` 到 `emitGroupBodyOnce(... &tailSizeOverride? )`):

```cpp
      // Ragged-tail: process the true remainder [covered, extent) at honest
      // offset `outerOfTailIV + mainInnerUb` with honest size
      // `remaining - mainInnerUb`.  No overlap / recompute; the GM DataCopy of
      // this block is routed to DataCopyPad (see CannTranslation
      // `afir.ragged_tail` walk) so an unaligned f16 offset/length is legal.
      Value honestTailSize = builder.create<arith::SubIOp>(
          loc, loopNest.remaining, loopNest.mainInnerUb);
      Value honestIV = builder.create<arith::AddIOp>(
          loc, loopNest.outerOfTailIV, loopNest.mainInnerUb);

      DenseMap<int, Value> tailLoopIVs = loopNest.loopIVs;
      tailLoopIVs[loopNest.innerTileAxisIdx] = honestIV;

      DenseMap<int, Value> tailSizeOverride;
      tailSizeOverride[loopNest.innerTileAxisIdx] = honestTailSize;

      SmallVector<Value> tailIterArgs(innermostFor.getResults().begin(),
                                       innermostFor.getResults().end());

      SmallVector<Value> tailYieldVals =
          emitGroupBodyOnce(builder, loc, info, plan,
                             tailLoopIVs, loopNest.outerLoopIVs,
                             tailIterArgs, /*bcastForOps=*/{},
                             /*sizeOverride=*/&tailSizeOverride);
```

(`innerTileExtent` / `innerTileStep` 字段现在尾块不再用;**不要**删它们的定义,
LoopNestBuilder 仍在填,后续 mode ② 计划可能复用。)

- [ ] **Step 4: 编译 + 跑 lit 确认通过**

Run: `cmake --build build --target afir-opt -j && build/bin/llvm-lit -v test/Conversion/AutoFuseCodegen/tail-peel.mlir`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add lib/Conversion/AutoFuse/TileFuse/GroupEmitter.cpp test/Conversion/AutoFuseCodegen/tail-peel.mlir
git commit -m "feat(auto-fuse): emit ragged tail (honest IV+size) instead of overlap-tail

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: 尾块 GM↔UB copy 路由到 DataCopyPad

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp:2645`(在累加器 DataCopyPad walk 之后插新 walk)

- [ ] **Step 1: 加 ragged-tail DataCopyPad walk**

在 `CannTranslation.cpp` 累加器 walk(`:2627-2645`)**结束之后**(第 2645 行 `});` 之后)
插入下面的新 walk。它把祖先 `scf.if` 带 `afir.ragged_tail` 的 GM↔UB `DataCopyL2Op`
(load 与 store 两向)翻成 `DataCopyPad`;UB↔UB 与已被累加器 walk 擦掉的不受影响。

```cpp
  // Ragged-tail GM↔UB copies (the tile-fuse `afir.ragged_tail` scf.if) →
  // DataCopyPad, so an unaligned f16 offset/length (rem·elemBytes not 32B
  // aligned) is legal.  Plain DataCopy requires 32B-block alignment and
  // silently drops/over-reads otherwise.  GM offset is already baked into the
  // GlobalTensor (SetGlobalBuffer base+offset); DataCopyPad just takes a byte
  // length.  Load needs a (no-op) DataCopyPadExtParams; store does not.
  auto inRaggedTail = [](Operation *op) -> bool {
    for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
      if (auto ifOp = dyn_cast<scf::IfOp>(p))
        if (ifOp->hasAttr("afir.ragged_tail"))
          return true;
    return false;
  };
  moduleOp->walk([&](ascendc::DataCopyL2Op op) {
    if (!inRaggedTail(op))
      return;
    bool store = isa<ascendc::GlobalTensorType>(op.getDst().getType());
    bool load = isa<ascendc::GlobalTensorType>(op.getSrc().getType());
    if (store == load)
      return; // UB↔UB or GM↔GM — leave on the default path
    // Accumulator stores are already rewritten by the walk above; skip.
    if (store && op.getSrc().getDefiningOp<ascendc::TBufGetTensorOp>())
      return;
    Value localSide = store ? op.getSrc() : op.getDst();
    auto lt = cast<ascendc::LocalTensorType>(localSide.getType());
    std::string elemTypeStr = getAscendCScalarTypeName(lt.getElementType());
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    std::string tmpl;
    if (store) {
      tmpl = "{\n  AscendC::DataCopyExtParams _afir_dcp{(uint16_t)1, "
             "(uint32_t)($2 * sizeof(" + elemTypeStr + ")), (uint32_t)0, "
             "(uint32_t)0, (uint32_t)0};\n"
             "  AscendC::DataCopyPad($0, $1, _afir_dcp);\n}";
    } else {
      tmpl = "{\n  AscendC::DataCopyExtParams _afir_dcp{(uint16_t)1, "
             "(uint32_t)($2 * sizeof(" + elemTypeStr + ")), (uint32_t)0, "
             "(uint32_t)0, (uint32_t)0};\n"
             "  AscendC::DataCopyPadExtParams<" + elemTypeStr +
             "> _afir_pad{false, (uint8_t)0, (uint8_t)0, (" + elemTypeStr +
             ")0};\n"
             "  AscendC::DataCopyPad($0, $1, _afir_dcp, _afir_pad);\n}";
    }
    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl),
        ValueRange({op.getDst(), op.getSrc(), op.getCalCount()}));
    rewriter.eraseOp(op);
  });
```

- [ ] **Step 2: 编译**

Run: `cmake --build build --target afir-translate -j`
Expected: 编译通过。

- [ ] **Step 3: 验证标记存活到翻译期(关键风险点)**

用 f16 例子生成 kernel,确认 `afir.ragged_tail` 标记经 bufferize 存活、且尾块发出
`DataCopyPad`:

```bash
cd examples/reduce-sum-3d-f16-tail-e2e
afir-opt reduce_sum_3d_f16.mlir --auto-fuse-codegen -o /tmp/f16_kernel.mlir
grep -c 'afir.ragged_tail' /tmp/f16_kernel.mlir      # 期望 ≥1(标记存活)
afir-translate -mlir-to-cann /tmp/f16_kernel.mlir -o /tmp/f16_kernel.cpp
grep -c 'DataCopyPad' /tmp/f16_kernel.cpp             # 期望 ≥2(尾块 load+store)
```

Expected: 两个 grep 都 ≥ 期望值。
**若 `afir.ragged_tail` count = 0**(bufferize 丢了 scf.if 属性):走 fallback ——
改 `inRaggedTail` 为"GM↔UB 且 calCount 非编译期 32B 对齐常量 且 祖先是 scf.if then 区"
的检测,或在 bufferize 后加一个轻量标记传播 pass;记录到本 Task 注释并重跑 Step 3。

- [ ] **Step 4: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp
git commit -m "feat(cann): route ragged-tail GM<->UB copies to DataCopyPad

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: 启用 f16 e2e gate + 观察 shadow-alloc

**Files:**
- Modify: `examples/reduce-sum-3d-f16-tail-e2e/run.sh`

- [ ] **Step 1: 去掉早退,启用 sim 验证**

删除 `run.sh` 中从第一个 `echo "===="` 横幅块到 `exit 0` 的整段(即 XFAIL banner +
`exit 0`),让脚本落到下面 `set -e` 的真实 codegen→sim→verify 链。把顶部注释里
"XFAIL pending option (D)" 一句改为:

```
# 3D reduce-sum f16 — tail-peel via ragged-tail (DataCopyPad).  Re-enabled
# 2026-06-05 after the overlap→ragged redesign; verifies rows 48..49 written.
```

具体:删除从 `echo "======...` (第一个横幅,约第 19 行)到 `exit 0`(约第 27 行)之间
所有行,保留其后的 `# ====` 分隔注释与 `set -e ...` 全部。

- [ ] **Step 2: 跑 e2e,期望 PASS**

Run: `bash examples/reduce-sum-3d-f16-tail-e2e/run.sh`
Expected: 结尾打印 `✓ session.validation=pass`,退出码 0。
(此前 overlap-tail 会让 rows 48..49 未写、`session.validation=fail`。)

- [ ] **Step 3: 观察是否出 shadow-alloc(决定要不要升级到 Y)**

```bash
afir-opt examples/reduce-sum-3d-f16-tail-e2e/reduce_sum_3d_f16.mlir \
  --auto-fuse-codegen -o /tmp/f16_kernel.mlir
# 尾块 scf.if then 区内的 alloc 数 / 是否有 copy-into-alloc 的 shadow sandwich
grep -n 'memref.alloc\|bufferization.alloc_tensor' /tmp/f16_kernel.mlir
```

记录结论到 commit message:
- **若 e2e PASS 且无多余 shadow-alloc** → X 成立,mode ① 完成。
- **若 PASS 但有 shadow-alloc 且 sim 慢/真机有隐患** → 记 issue,后续按 spec Y
  解耦静态-T(不阻塞本计划交付,sim 已对)。

- [ ] **Step 4: Commit**

```bash
git add examples/reduce-sum-3d-f16-tail-e2e/run.sh
git commit -m "test(examples): re-enable reduce-sum-3d-f16-tail e2e (ragged tail)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: 放宽 `Divides{32}` 拒绝约束

**Files:**
- Modify: `lib/Conversion/AutoFuse/TileFuse/TilePlanBuild.cpp:379-416`

- [ ] **Step 1: 删除尾块 32B 对齐硬约束**

DataCopyPad-with-offset 已能正确处理不对齐尾巴,这条"拒绝不对齐 tiling"的硬约束不再
需要。删除第 379-416 的整段注释 + `const TileParam *innermostInner = nullptr; ...`
到 push `Divides{"32", rhs}` 的 `if (innermostInner) { ... }` 块。在原位留一行说明:

```cpp
  // (Removed: the Divides{32, (extent-INNER)*elemBytes} tail-alignment reject
  // constraint.  Ragged-tail DataCopyPad now handles unaligned f16 tails
  // directly, so misaligned tilings are correct rather than rejected.  Aligned
  // tilings remain naturally preferred by the footprint/cost model below.)
```

- [ ] **Step 2: 编译**

Run: `cmake --build build --target afir-opt -j`
Expected: 编译通过。

- [ ] **Step 3: 确认约束不再出现在 tiling JSON**

```bash
afir-translate -mlir-to-cann /tmp/f16_kernel.mlir -o /dev/null \
  --tiling-space-out /tmp/ts.json
grep -c '"32"' /tmp/ts.json     # 期望 0(不再发 32B 拒绝约束)
```

Expected: 0。

- [ ] **Step 4: Commit**

```bash
git add lib/Conversion/AutoFuse/TileFuse/TilePlanBuild.cpp
git commit -m "feat(auto-fuse): drop Divides{32} tail-align reject; DataCopyPad handles it

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: 回归门(lit 全量 + 关键 e2e)

**Files:** 无改动(仅运行验证)。

- [ ] **Step 1: lit 全量**

Run: `build/bin/llvm-lit -v test/Conversion/AutoFuseCodegen test/Conversion/LinalgToAscendC`
Expected: 全 PASS(尾块相关用例已在 Task 1 更新)。

- [ ] **Step 2: 幂等 elementwise 热路径回归(尾块从 DataCopy 换 DataCopyPad)**

Run:
```bash
bash examples/two-elewise-e2e/run.sh
bash examples/gelu-dyn-e2e/run.sh
```
Expected: 两者 `session.validation=pass` / max_diff 不退化(与改动前一致)。

- [ ] **Step 3: reduce 路径回归**

Run: `bash examples/reduce-big-r-e2e/run.sh`
Expected: PASS,max_diff 不退化。

- [ ] **Step 4: 记录回归结果(无代码改动则跳过 commit)**

若上述任一退化:停下,按 systematic-debugging 定位(优先看 Step 2 的 DataCopyPad 是否
误伤了对齐主循环 —— 本计划只应改尾块,主循环仍走 DataCopy)。全绿则本计划 mode ① 完成。

---

## Self-Review(已核对)

- **spec 覆盖**:§4.1 DataCopyPad 路由 → Task 2;§4.2 尾块 ragged → Task 1;§4.4 放宽
  Divides{32} → Task 4;§6 验证(f16 gate 转 PASS + 回归)→ Task 3/5。§4.3 mode ②
  (reduction 轴)**显式移出本计划**(见下"后续"),非遗漏。
- **占位**:无 TBD/TODO;每个 code step 给了完整代码或精确命令 + 期望。
- **类型一致**:`afir.ragged_tail`(UnitAttr)在 Task 1 写、Task 2 读;`honestTailSize` /
  `honestIV` 命名一致;`DataCopyL2Op` / `getCalCount` / `getAscendCScalarTypeName` /
  `GlobalTensorType` / `LocalTensorType` / `TBufGetTensorOp` 均沿用 CannTranslation 现有符号。
- **风险**:Task 2 Step 3 显式验证 `afir.ragged_tail` 经 bufferize 存活,并给了
  fallback;Task 3 Step 3 显式观察 shadow-alloc,不阻塞交付。

## 后续(不在本计划)

- **mode ② reduction 轴 ragged 尾**:`emitGroupWithReductionSplit` 的 reduction `for`
  覆盖完整 extent + 累加层穿 `count=rem`(`AddL2/MaxL2/MinL2` 已带 count)。潜在、当前
  未触发(靠整行单 RBLOCK 规避),独立可交付 → 单开计划
  `2026-06-XX-tail-block-reduction-axis.md`,需先精读 `GroupEmitter.cpp:52-260` 再写
  精确步骤。
- **spec Y(解耦静态-T)**:仅当 Task 3 Step 3 观察到 shadow-alloc 造成真机/性能问题时
  启动。
