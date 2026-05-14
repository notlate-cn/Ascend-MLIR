# Phase 5C+ Tail Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the generic axis tail-plan pipeline so non-16-aligned `gather-elementwise-fusion` shapes (`N=127`, `K=31`) run through the new Ascend mainline without gather-specific schedule shortcuts.

**Architecture:** Keep the decision boundary from the V2-3/4/5/6 specs: Schedule derives allowed tail policies and emits a concrete `ScheduledAxisTailPlan`; Structured Lowering serializes the selected plan as stable IR metadata; Realize binds the plan to buffers and movement facts; Phase 5 lowers the plan to legal AscendC mask/padding/scalar behavior. The first executable acceptance path targets gather N/K tail, but the data model and verifiers must remain op-generic.

**Tech Stack:** MLIR C++ passes, Ascend Schedule/Realize/Backend/CANN translation, Linalg-to-AscendC lowering, LIT/FileCheck, example runtime-session sim, xvm/docker verification via `ssh xvm@orb`.

---

## Scope

This plan implements the minimum generic tail-plan chain needed for gather N/K tail. It does not implement full matmul/cube tail, full chunked-reduction padding, or all future `PadAndMask` physical workspace modes. Those remain enabled by the data model and verifier, but only gather/vector data movement paths are required to pass in this phase.

The implementation must not modify the CANN installation directory. Host-side code edits happen in this repo; build and tests run in xvm at `/home/niu/code/Ascend-MLIR`.

## File Structure

- Modify `include/Conversion/Ascend/Schedule/ScheduleTypes.h`
  - Add `PadAndMask`, `PrimitiveAxisUseKind`, `TailBufferingMode`, `ScheduledAxisTailPlan`.
  - Change `AxisScheduleConstraint::tailPolicy` into `allowedTailPolicies`.
  - Add `primitiveUses` and `semanticAlignmentGranularity`.
  - Add `ScheduleDecision::tailPlans`.
- Modify `lib/Conversion/Ascend/Schedule/AxisCoalescer.cpp`
  - Derive allowed policy sets and primitive uses.
  - Keep legacy report output stable enough for existing tests while adding new fields.
- Modify `lib/Conversion/Ascend/Schedule/ScheduleSearch.cpp`
  - Generate divisibility guards only when `allowedTailPolicies` resolves to `MustDivide`.
  - Build selected `ScheduledAxisTailPlan` for each decision.
- Modify `lib/Conversion/Ascend/Schedule/ScheduleProblemBuilder.cpp`
  - Print the new tail-plan contract fields.
- Modify `lib/Conversion/Ascend/Schedule/StructuredLoweringDriver.cpp`
  - Serialize selected tail plans to IR attributes.
  - Preserve existing `ascend.schedule.tail_policies` artifact compatibility while adding richer metadata.
- Modify `include/Conversion/Ascend/Common/Attributes.h`
  - Add shared attr constants for selected tail plan metadata.
- Modify `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`
  - Read tail-plan metadata and keep it on realized operations/functions.
  - Add conservative binding hooks for `PadAndMask`/`MaskedTail` consumers.
- Modify `lib/Conversion/Ascend/Backend/ComputeLoweringPass.cpp`
  - Keep backend support matrix aligned with non-16 gather tail.
- Modify `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp`
  - Lower gather row count and K tail using selected tail plan metadata.
- Modify `lib/Conversion/LinalgToAscendC/DataMoveConversion.cpp`
  - Avoid assuming N/K are multiples of 16 in gather-related copy paths.
- Modify `lib/Target/CannKernel/CannTranslation.cpp`
  - Emit legal AscendC for gather count/tail sizes and aligned runtime buffers.
- Modify `lib/Target/CannKernel/CannRuntimeArtifacts.cpp`
  - Add tail-plan manifest entries while preserving existing `tail_policies`.
- Modify `examples/gather-elementwise-fusion/gen_data.py`
  - Remove `N % 16 == 0 && K % 16 == 0` assertions.
- Modify `examples/gather-elementwise-fusion/run-mainline.sh`
  - Remove the current script-level N/K multiple-of-16 rejection.
  - Add shape matrix entry points for `N=127`, `K=31`.
- Modify `test/Conversion/ascend-schedule-axis-coalescing.mlir`
- Modify `test/Conversion/ascend-schedule-guards.mlir`
- Modify `test/Conversion/ascend-schedule-search.mlir`
- Modify `test/Conversion/ascend-full-pipeline-gather-elementwise-fusion.mlir`
- Modify `test/Target/cann-translate-gather.mlir`
- Modify `test/tools/runtime/run_simbackend_examples.sh`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

Do not modify `docs/Ascend-MLIR-Detailed-Implementation-V1.zh.md`. Do not merge split V2 docs back into `docs/Ascend-MLIR-Detailed-Implementation-V2.zh.md` in this implementation plan.

## Task 1: Schedule Tail Data Model

**Files:**
- Modify `include/Conversion/Ascend/Schedule/ScheduleTypes.h`
- Modify `lib/Conversion/Ascend/Schedule/AxisCoalescer.cpp`
- Modify `lib/Conversion/Ascend/Schedule/ScheduleProblemBuilder.cpp`
- Modify `test/Conversion/ascend-schedule-axis-coalescing.mlir`

- [ ] **Step 1: Write failing LIT for contract fields**

Extend `test/Conversion/ascend-schedule-axis-coalescing.mlir` with FileCheck expectations for a vector/gather-shaped all-parallel op:

```mlir
// CHECK: axis_constraints = [
// CHECK-SAME: axis=0 kind=parallel roles=[bind_core,kernel_loop,vectorize]
// CHECK-SAME: allowed_tail=[masked_tail,scalar_epilogue]
// CHECK-SAME: primitive_uses=[data_copy,vector_compute,write_back]
// CHECK-SAME: semantic_align=0
// CHECK-SAME: axis=1 kind=parallel roles=[bind_core,kernel_loop,vectorize]
// CHECK-SAME: allowed_tail=[masked_tail,scalar_epilogue,pad_and_mask]
```

Run in xvm:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-schedule-axis-coalescing.mlir'
```

Expected before implementation: FAIL because `allowed_tail`, `primitive_uses`, and `semantic_align` are not printed.

- [ ] **Step 2: Update core enums and structs**

In `include/Conversion/Ascend/Schedule/ScheduleTypes.h`, update the model:

```cpp
enum class AxisTailPolicy {
  MustDivide,
  MaskedTail,
  ScalarEpilogue,
  PadAndMask,
  FullExtent,
};

enum class PrimitiveAxisUseKind {
  DataCopy,
  VectorCompute,
  Reduction,
  GatherIndex,
  CubeM,
  CubeN,
  CubeK,
  WriteBack,
};

enum class TailBufferingMode {
  SeparateTailBuffer,
  ReuseMainBufferAfterDrain,
};

struct AxisScheduleConstraint {
  unsigned logicalAxisId = 0;
  AxisKind kind = AxisKind::Unknown;
  SmallVector<AxisExecutionRole, 3> allowedRoles;
  SmallVector<AxisTailPolicy, 3> allowedTailPolicies;
  SmallVector<PrimitiveAxisUseKind, 4> primitiveUses;
  int64_t semanticAlignmentGranularity = 0;
  uint32_t coalescingGroupId = 0;
};

struct ScheduledAxisTailPlan {
  unsigned logicalAxisId = 0;
  AxisTailPolicy selectedPolicy = AxisTailPolicy::MaskedTail;
  SmallVector<PrimitiveAxisUseKind, 4> affectedPrimitiveUses;
  int64_t extent = ShapedType::kDynamic;
  int64_t tileSize = ShapedType::kDynamic;
  int64_t alignmentGranularity = 0;
  int64_t mainExtent = ShapedType::kDynamic;
  int64_t tailExtent = ShapedType::kDynamic;
  TailBufferingMode tailBufferingMode = TailBufferingMode::SeparateTailBuffer;
  bool emitsRuntimeGuard = false;
};
```

Add to `ScheduleDecision`:

```cpp
SmallVector<ScheduledAxisTailPlan, 4> tailPlans;
```

Add stringify helpers for the new enums using lower snake case:

```text
pad_and_mask
data_copy,vector_compute,reduction,gather_index,cube_m,cube_n,cube_k,write_back
separate_tail_buffer,reuse_main_buffer_after_drain
```

- [ ] **Step 3: Derive allowed policy sets**

In `AxisCoalescer.cpp`, replace `constraint.tailPolicy = ...` with set population:

```cpp
case AxisKind::Parallel:
  constraint.allowedTailPolicies.push_back(AxisTailPolicy::MaskedTail);
  constraint.allowedTailPolicies.push_back(AxisTailPolicy::ScalarEpilogue);
  constraint.primitiveUses.push_back(PrimitiveAxisUseKind::DataCopy);
  constraint.primitiveUses.push_back(PrimitiveAxisUseKind::VectorCompute);
  constraint.primitiveUses.push_back(PrimitiveAxisUseKind::WriteBack);
  break;
case AxisKind::Reduction:
  constraint.allowedTailPolicies.push_back(AxisTailPolicy::FullExtent);
  constraint.primitiveUses.push_back(PrimitiveAxisUseKind::Reduction);
  break;
case AxisKind::Unknown:
  constraint.allowedTailPolicies.push_back(AxisTailPolicy::MustDivide);
  break;
```

For gather/indexing axes detected by `gather_dim` or the existing gather role path, add:

```cpp
addUnique(constraint.allowedTailPolicies, AxisTailPolicy::PadAndMask);
addUnique(constraint.primitiveUses, PrimitiveAxisUseKind::GatherIndex);
constraint.semanticAlignmentGranularity = 16;
```

If the current pass cannot reliably identify gather axes at this layer, keep all parallel axes as `{MaskedTail, ScalarEpilogue}` and add `PadAndMask` in Task 2 when building the selected tail plan for gather-shaped operations. Do not add a gather-only side table.

- [ ] **Step 4: Update reports**

Update `printAxisScheduleConstraint`-style helpers in `AxisCoalescer.cpp` and `ScheduleProblemBuilder.cpp` to print:

```text
allowed_tail=[masked_tail,scalar_epilogue,pad_and_mask]
primitive_uses=[data_copy,vector_compute,write_back]
semantic_align=16
```

Keep existing `tail=...` output only if existing tests still require it, but make it an alias for the selected/default first policy in reports. New tests should use `allowed_tail`.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-schedule-axis-coalescing.mlir build/test/Conversion/ascend-schedule-problem.mlir'
```

Expected: both tests pass.

## Task 2: Schedule Decision Tail Plan Builder

**Files:**
- Modify `lib/Conversion/Ascend/Schedule/ScheduleSearch.cpp`
- Modify `lib/Conversion/Ascend/Schedule/ScheduleDecision.cpp` if decision reporting lives there
- Modify `test/Conversion/ascend-schedule-guards.mlir`
- Modify `test/Conversion/ascend-schedule-search.mlir`

- [ ] **Step 1: Write failing guard and selected-plan tests**

Add FileCheck to `test/Conversion/ascend-schedule-guards.mlir`:

```mlir
// MASKED-NOT: DivisibleBy
// MUSTDIVIDE: guard kind=divisible_by axis=0 value=16
```

Add FileCheck to `test/Conversion/ascend-schedule-search.mlir`:

```mlir
// CHECK: tail_plans = [
// CHECK-SAME: axis=0 selected=masked_tail
// CHECK-SAME: affected=[vector_compute,write_back]
// CHECK-SAME: buffering=separate_tail_buffer
```

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-schedule-guards.mlir build/test/Conversion/ascend-schedule-search.mlir'
```

Expected before implementation: FAIL because only `tailPolicy` exists and no selected `tail_plans` are reported.

- [ ] **Step 2: Replace `tailPolicy` guard lookup**

In `ScheduleSearch.cpp`, replace:

```cpp
if (!constraint || constraint->tailPolicy != AxisTailPolicy::MustDivide)
  continue;
```

with:

```cpp
if (!constraint || !requiresDivisibleGuard(*constraint))
  continue;
```

Define:

```cpp
static bool requiresDivisibleGuard(const AxisScheduleConstraint &constraint) {
  return constraint.allowedTailPolicies.size() == 1 &&
         constraint.allowedTailPolicies.front() == AxisTailPolicy::MustDivide;
}
```

- [ ] **Step 3: Select concrete tail policy**

Add a local helper in `ScheduleSearch.cpp`:

```cpp
static AxisTailPolicy selectTailPolicy(const AxisScheduleConstraint &constraint) {
  if (llvm::is_contained(constraint.allowedTailPolicies,
                         AxisTailPolicy::MaskedTail))
    return AxisTailPolicy::MaskedTail;
  if (llvm::is_contained(constraint.allowedTailPolicies,
                         AxisTailPolicy::ScalarEpilogue))
    return AxisTailPolicy::ScalarEpilogue;
  if (llvm::is_contained(constraint.allowedTailPolicies,
                         AxisTailPolicy::PadAndMask))
    return AxisTailPolicy::PadAndMask;
  if (llvm::is_contained(constraint.allowedTailPolicies,
                         AxisTailPolicy::FullExtent))
    return AxisTailPolicy::FullExtent;
  return AxisTailPolicy::MustDivide;
}
```

For gather N/K tail acceptance, if an axis has `GatherIndex` or a 16-element `semanticAlignmentGranularity`, select `PadAndMask` only when the target path cannot be safely masked. Otherwise keep `MaskedTail`. This keeps the first implementation conservative and generic.

- [ ] **Step 4: Populate `ScheduleDecision::tailPlans`**

In `makeInstance` or the nearest `ScheduleDecision` construction path, for each tile index:

```cpp
ScheduledAxisTailPlan plan;
plan.logicalAxisId = axis.logicalAxisId;
plan.selectedPolicy = selectTailPolicy(*constraint);
plan.affectedPrimitiveUses = getAffectedPrimitiveUses(*constraint, plan.selectedPolicy);
plan.extent = axis.staticExtent;
plan.tileSize = tileSize;
plan.alignmentGranularity = constraint->semanticAlignmentGranularity;
plan.mainExtent = ShapedType::kDynamic;
plan.tailExtent = ShapedType::kDynamic;
plan.tailBufferingMode = TailBufferingMode::SeparateTailBuffer;
plan.emitsRuntimeGuard = plan.selectedPolicy == AxisTailPolicy::MustDivide;
decision.tailPlans.push_back(std::move(plan));
```

Use static `mainExtent`/`tailExtent` only when both extent and tile size are static:

```cpp
plan.mainExtent = (axis.staticExtent / tileSize) * tileSize;
plan.tailExtent = axis.staticExtent - plan.mainExtent;
```

- [ ] **Step 5: Print and verify**

Add stable report output:

```text
tail_plans = [axis=0 selected=masked_tail affected=[vector_compute,write_back] align=0 buffering=separate_tail_buffer]
```

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-schedule-guards.mlir build/test/Conversion/ascend-schedule-search.mlir'
```

Expected: both tests pass.

## Task 3: Structured Lowering Tail Metadata

**Files:**
- Modify `include/Conversion/Ascend/Common/Attributes.h`
- Modify `lib/Conversion/Ascend/Schedule/StructuredLoweringDriver.cpp`
- Modify `lib/Target/CannKernel/CannRuntimeArtifacts.cpp`
- Modify `test/Conversion/ascend-full-pipeline-gather-elementwise-fusion.mlir`

- [ ] **Step 1: Write failing full-pipeline metadata test**

Extend `test/Conversion/ascend-full-pipeline-gather-elementwise-fusion.mlir`:

```mlir
// CHECK: ascend.schedule.selected_tile_shape
// CHECK: ascend.schedule.tail_policies
// CHECK: ascend.schedule.tail_plan
// CHECK-SAME: selected
// CHECK-SAME: buffering
```

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-full-pipeline-gather-elementwise-fusion.mlir'
```

Expected before implementation: FAIL because `ascend.schedule.tail_plan` does not exist.

- [ ] **Step 2: Add common attribute names**

In `include/Conversion/Ascend/Common/Attributes.h`, add:

```cpp
inline constexpr llvm::StringLiteral kScheduleTailPlanAttr =
    "ascend.schedule.tail_plan";
```

Keep existing `kScheduleTailPoliciesAttr` unchanged for compatibility.

- [ ] **Step 3: Serialize tail plans**

In `StructuredLoweringDriver.cpp`, add:

```cpp
static ArrayAttr buildTailPlanAttr(MLIRContext *context,
                                   ArrayRef<ScheduledAxisTailPlan> plans);
```

Use `DictionaryAttr` entries:

```text
axis = i64
selected = "masked_tail"
affected = ["vector_compute", "write_back"]
align = i64
buffering = "separate_tail_buffer"
```

Attach this attr to the same op/function locations that already receive `kScheduleTailPoliciesAttr`.

- [ ] **Step 4: Preserve runtime artifact compatibility**

In `CannRuntimeArtifacts.cpp`, keep existing `tail_policies` JSON and add:

```json
"tail_plan": [
  {
    "axis": 0,
    "selectedPolicy": "masked_tail",
    "alignmentGranularity": 0,
    "tailBufferingMode": "separate_tail_buffer",
    "affectedPrimitiveUses": ["vector_compute", "write_back"]
  }
]
```

- [ ] **Step 5: Verify metadata**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt afir-translate && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-full-pipeline-gather-elementwise-fusion.mlir build/test/Target'
```

Expected: gather full-pipeline LIT and target tests pass.

## Task 4: Realize Tail Binding Preservation

**Files:**
- Modify `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`
- Modify `test/Conversion/ascend-realize-gather-output-bridge.mlir`

- [ ] **Step 1: Write failing Realize test**

Extend `test/Conversion/ascend-realize-gather-output-bridge.mlir`:

```mlir
// CHECK: ascend.schedule.tail_plan
// CHECK: memory_space
// CHECK-NOT: unsupported shape
```

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-realize-gather-output-bridge.mlir'
```

Expected before implementation: FAIL if Realize drops the new attr or rejects non-16 tail assumptions.

- [ ] **Step 2: Preserve tail attrs during memory-space annotation**

In `MemoryRealizationDriver.cpp`, ensure any op/function attr copy path preserves:

```cpp
kScheduleTailPlanAttr
kScheduleTailPoliciesAttr
kScheduleSelectedTileShapeAttr
```

If no attr copy path exists, add a small helper:

```cpp
static void copyScheduleAttrs(Operation *from, Operation *to) {
  for (StringRef name : {kScheduleSelectedTileShapeAttr,
                         kScheduleTailPoliciesAttr,
                         kScheduleTailPlanAttr})
    if (Attribute attr = from->getAttr(name))
      to->setAttr(name, attr);
}
```

- [ ] **Step 3: Add conservative binding diagnostics**

When an op has `kScheduleTailPlanAttr`, do not reinterpret the plan in Realize. Only check that memory-space annotation does not remove the op/function metadata. Emit a diagnostic if `PadAndMask` appears and no output memory space can be inferred.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-realize-gather-output-bridge.mlir build/test/Conversion/ascend-full-pipeline-gather-elementwise-fusion.mlir'
```

Expected: both pass.

## Task 5: Gather Codegen Non-16 Tail

**Files:**
- Modify `lib/Conversion/Ascend/Backend/ComputeLoweringPass.cpp`
- Modify `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp`
- Modify `lib/Conversion/LinalgToAscendC/DataMoveConversion.cpp`
- Modify `lib/Target/CannKernel/CannTranslation.cpp`
- Modify `test/Target/cann-translate-gather.mlir`

- [ ] **Step 1: Write failing CANN translation test**

Extend `test/Target/cann-translate-gather.mlir` with checks that generated code uses dynamic gather count and does not require `K / 16` for gather count:

```mlir
// CHECK: _afir_idx32.SetSize((uint32_t)
// CHECK: AscendC::Gather(
// CHECK-SAME: _afir_idx32
// CHECK: SetSize((uint32_t)
// CHECK-NOT: unsupported shape
```

Add a focused input in `test/Target/cann-translate-gather-input.mlir` or extend the existing one so the gather count is a symbolic `K` that is not statically divisible by 16.

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Target/cann-translate-gather.mlir'
```

Expected before implementation: FAIL if any path still emits a static aligned-count assumption.

- [ ] **Step 2: Keep gather count logical**

In `ComputeConversion.cpp`, confirm the `GatherL2Op` count operand is the logical `dimK` count:

```cpp
b.create<GatherL2Op>(forLoc, gatheredRowLt, processedRowLt,
                     indicesLt, srcBaseAddr, dimK_i32);
```

Do not replace it with `ceil(K, 16)` or `K / 16`.

- [ ] **Step 3: Allocate physical buffers with aligned bytes**

Rely on the existing `TPipeInitBufferOp` / `TPipeInitQueueOp` translation path in `CannTranslation.cpp` that rounds physical byte capacities up to 32 bytes. If a gather-local TBuf bypasses those ops, route it through the same TPipe init path instead of open-coding a separate alignment expression.

- [ ] **Step 4: Bound logical operations by true K**

In `CannTranslation.cpp` gather lowering, keep:

```cpp
$0.SetSize((uint32_t)$4);
_afir_idx32.SetSize((uint32_t)$4);
AscendC::Gather($0, $1, _afir_idx32, $3, $4);
```

where `$4` is logical K. If destination physical LocalTensor capacity is aligned larger than K, the extra lanes must not be written back to GM.

- [ ] **Step 5: Guard source N bounds**

Ensure data generation and runtime validation keep indices in `[0, N)`. Do not add a compile-time `N % 16 == 0` guard. If source `DataCopy` uses N-sized row buffers, the physical VECCALC buffer may be aligned while its logical `SetSize` remains N.

- [ ] **Step 6: Verify focused target tests**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt afir-translate && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Target/cann-translate-gather.mlir'
```

Expected: PASS.

## Task 6: Example Acceptance For N/K Tail

**Files:**
- Modify `examples/gather-elementwise-fusion/gen_data.py`
- Modify `examples/gather-elementwise-fusion/run-mainline.sh`
- Modify `examples/gather-elementwise-fusion/run.sh` only if the wrapper needs argument passthrough correction
- Modify `test/tools/runtime/run_simbackend_examples.sh`

- [ ] **Step 1: Remove data generator alignment assertions**

In `gen_data.py`, replace:

```python
assert K >= 16, "K must be >= 16 for DataCopy alignment"
assert K <= N, "K must be <= N"
assert N % 16 == 0 and K % 16 == 0, "N and K must be multiples of 16"
```

with:

```python
assert K >= 1, "K must be positive"
assert N >= 1, "N must be positive"
assert K <= N, "K must be <= N"
```

Keep indices generated in `[0, N)`.

- [ ] **Step 2: Remove script-level N/K rejection**

In `run-mainline.sh`, delete:

```bash
if (( K < 16 )); then
  echo "unsupported shape: K must be >= 16 for current gather lowering" >&2
  exit 2
fi
...
if (( N % 16 != 0 || K % 16 != 0 )); then
  echo "unsupported shape: current gather mainline requires N and K to be multiples of 16" >&2
  exit 2
fi
```

Replace with:

```bash
if (( K < 1 )); then
  echo "unsupported shape: K must be positive" >&2
  exit 2
fi
if (( N < 1 )); then
  echo "unsupported shape: N must be positive" >&2
  exit 2
fi
```

- [ ] **Step 3: Add focused runtime matrix**

In `test/tools/runtime/run_simbackend_examples.sh`, add gather matrix entries:

```bash
run_gather_shape 65 127 31 1
run_gather_shape 96 128 31 1
run_gather_shape 96 127 32 1
```

Use the existing example `run.sh --log --m ... --n ... --k ... --block-dim ...` style and keep outputs under `build_mainline`.

- [ ] **Step 4: Verify examples on xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && bash examples/gather-elementwise-fusion/run.sh --log --m 65 --n 127 --k 31 --block-dim 1'
```

Expected:

```text
session.result=success
session.validation=pass
```

Then run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && bash examples/gather-elementwise-fusion/run.sh --log --m 96 --n 128 --k 31 --block-dim 1 && bash examples/gather-elementwise-fusion/run.sh --log --m 96 --n 127 --k 32 --block-dim 1'
```

Expected: both runs report `session.validation=pass`.

## Task 7: Regression And Tracking

**Files:**
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Run focused LIT regression**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt afir-translate && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion --filter="ascend-(schedule|full-pipeline|realize).*gather|ascend-schedule-(axis|guards|search)" && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Target --filter="cann-translate-gather"'
```

Expected: all selected tests pass.

- [ ] **Step 2: Run supported examples**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && bash test/tools/runtime/run_simbackend_examples.sh gather-elementwise-fusion'
```

Expected: gather example and shape matrix pass.

- [ ] **Step 3: Run ordinary examples that must not regress**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && bash examples/broadcast-add-reduce/run.sh --log && bash examples/relu-broadcast-transpose/run.sh --log && bash examples/add-broadcast-concat/run.sh --log'
```

Expected: each run reports `session.result=success` and `session.validation=pass`.

- [ ] **Step 4: Run static checks**

Run on host:

```bash
git diff --check -- . ':!AGENTS.md'
test/tools/check_ascend_no_v2_code_naming.sh
```

Expected: both pass.

- [ ] **Step 5: Update tracking**

Update `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`:

- Phase 5C+ note: `gather-elementwise-fusion N/K tail-policy lowering implemented`.
- Verification row: include `M=65,N=127,K=31`, `M=96,N=128,K=31`, `M=96,N=127,K=32`.
- Keep existing notes for broadcast/relu/add-concat unchanged.

- [ ] **Step 6: Final review prompt**

Prepare a short review summary containing:

```text
Implemented generic Schedule tail-plan data model and metadata propagation.
Removed gather N/K multiple-of-16 script rejection.
Validated gather N/K tail shapes and ordinary examples on xvm.
Key files: ScheduleTypes.h, AxisCoalescer.cpp, ScheduleSearch.cpp,
StructuredLoweringDriver.cpp, MemoryRealizationDriver.cpp,
ComputeConversion.cpp, CannTranslation.cpp, gather example scripts.
```

Do not commit until the user explicitly asks for commit/push.

## Self-Review

- Spec coverage: V2-3 contract fields are covered by Task 1; V2-4 selected `ScheduledAxisTailPlan` and lowering markers are covered by Tasks 2 and 3; V2-5 binding preservation is covered by Task 4; V2-6 codegen/runtime artifact consumption is covered by Tasks 5 and 7.
- Placeholder scan: no incomplete implementation slots are intentionally left in this plan.
- Type consistency: the plan consistently uses `allowedTailPolicies`, `PrimitiveAxisUseKind`, `ScheduledAxisTailPlan`, `TailBufferingMode`, `TailPlanMarker`, and `tailPlans`.
- Risk: current C++ already has uncommitted gather/backend/runtime changes. Workers must preserve unrelated user edits and must not revert existing dirty files.
