# Axis Schedule Contract Coalescing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the V2-3 axis-level schedule contract and coalescing hint model so the new Ascend mainline can schedule `broadcast + add + reduce` with non-trivial `M` shapes instead of relying on one full single-core tile.

**Architecture:** The current executable handoff between `--ascend-kernelize` and `--ascend-schedule` is IR attributes, while the C++ `kernelize::ScheduleContract` is not serialized. This plan therefore lands the axis contract in the Schedule data model first, deriving it deterministically from `KernelPatternView` + `CoalescedAxisInfo`, then uses the same model to drive guard generation and bounded reduction tiling. Kernelize-side contract serialization remains out of this plan until the pass-boundary attribute format is designed.

**Tech Stack:** MLIR C++ passes, Linalg iterator/indexing maps, Ascend Schedule/Realize/Phase 5 pipeline, LIT/FileCheck, xvm/docker verification via `ssh xvm@orb`.

---

## File Structure

- Modify `include/Conversion/Ascend/Schedule/ScheduleTypes.h`
  - Add `AxisExecutionRole`, `AxisTailPolicy`, `CoalescingHintKind`, `AxisScheduleConstraint`, and `AxisCoalescingHint`.
  - Add axis constraints and coalescing hints to `CoalescedAxisInfo`.
- Modify `lib/Conversion/Ascend/Schedule/AxisCoalescer.cpp`
  - Derive per-axis allowed roles, tail policy, and conservative coalescing hint groups.
  - Print the new contract fields in the schedule debug report.
- Modify `lib/Conversion/Ascend/Schedule/ScheduleProblemBuilder.cpp`
  - Preserve the new axis contract in `ScheduleProblem` and print it in `dump-report`.
- Modify `lib/Conversion/Ascend/Schedule/ScheduleSearch.cpp`
  - Generate tile shapes from axis roles and tail policy instead of only full/half-shape heuristics.
  - Generate `DivisibleBy` decision guards only for `MustDivide` axes.
- Modify `lib/Conversion/Ascend/Schedule/ScheduleDecision.cpp`
  - Report selected tile shape and guard counts for reviewable diagnostics.
- Modify `lib/Conversion/Ascend/Schedule/StructuredLoweringDriver.cpp`
  - Attach selected tile sizes and tail policy summary as stable schedule attrs for Phase 5 artifact consumption.
- Modify `include/Conversion/Ascend/Common/Attributes.h`
  - Add shared attr names for selected tile sizes and tail policy summaries.
- Modify `lib/Target/CannKernel/CannRuntimeArtifacts.cpp`
  - Include selected schedule tile metadata in `tiling_space.json` and runtime manifest schedule entries.
- Modify `examples/broadcast-add-reduce/run-mainline.sh`
  - Default to the original large acceptance shape after the scheduler can tile it.
- Modify `test/Conversion/ascend-schedule-axis-coalescing.mlir`
- Modify `test/Conversion/ascend-schedule-problem.mlir`
- Modify `test/Conversion/ascend-schedule-guards.mlir`
- Modify `test/Conversion/ascend-schedule-search.mlir`
- Modify `test/Conversion/ascend-full-pipeline-broadcast-add-reduce.mlir`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

Do not modify or stage `AGENTS.md`; it contains local user instructions and unrelated working-tree state.

## Task 1: Axis Contract Data Model And Debug Report

**Files:**
- Modify `include/Conversion/Ascend/Schedule/ScheduleTypes.h`
- Modify `lib/Conversion/Ascend/Schedule/AxisCoalescer.cpp`
- Modify `test/Conversion/ascend-schedule-axis-coalescing.mlir`

- [x] **Step 1: Write the failing LIT**

Extend `test/Conversion/ascend-schedule-axis-coalescing.mlir` with a `dump-report` case containing a rank-2 all-parallel linalg op and a rank-2 reduction op.

Expected report fragments:

```mlir
// CHECK: AxisCoalescing:
// CHECK: axis_constraints = [
// CHECK-SAME: axis=0 kind=parallel roles=[bind_core,kernel_loop,vectorize] tail=masked_tail group=1
// CHECK-SAME: axis=1 kind=parallel roles=[bind_core,kernel_loop,vectorize] tail=masked_tail group=1
// CHECK: coalescing_hints = [
// CHECK-SAME: group=1 kind=vectorizable members=[0,1]
// REDUCE: axis=0 kind=parallel roles=[bind_core,kernel_loop,vectorize] tail=masked_tail
// REDUCE: axis=1 kind=reduction roles=[full_reduction] tail=full_extent
```

Run in xvm:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-schedule-axis-coalescing.mlir'
```

Expected before implementation: FAIL because the report has only `parallel_axes`, `reduction_axes`, `broadcast_axes`, and `barriers`.

- [x] **Step 2: Add the schedule data model**

Add this model to `include/Conversion/Ascend/Schedule/ScheduleTypes.h`:

```cpp
enum class AxisExecutionRole {
  BindCoreCandidate,
  KernelLoopCandidate,
  VectorizeCandidate,
  FullReduction,
  ChunkedReduction,
  BroadcastProjection,
  LayoutCarry,
};

enum class AxisTailPolicy {
  MustDivide,
  MaskedTail,
  ScalarEpilogue,
  FullExtent,
};

enum class CoalescingHintKind {
  Vectorizable,
  LinearizeOnly,
};

struct AxisScheduleConstraint {
  unsigned logicalAxisId = 0;
  AxisKind kind = AxisKind::Unknown;
  SmallVector<AxisExecutionRole, 3> allowedRoles;
  AxisTailPolicy tailPolicy = AxisTailPolicy::MustDivide;
  uint32_t coalescingGroupId = 0;
};

struct AxisCoalescingHint {
  uint32_t groupId = 0;
  CoalescingHintKind kind = CoalescingHintKind::LinearizeOnly;
  SmallVector<unsigned, 2> memberAxisIds;
};
```

Add to `CoalescedAxisInfo`:

```cpp
SmallVector<AxisScheduleConstraint> axisScheduleConstraints;
SmallVector<AxisCoalescingHint> axisCoalescingHints;
```

- [x] **Step 3: Derive axis constraints in AxisCoalescer**

In `coalesceAxes`, after `logicalAxes`, `parallelAxes`, `reductionAxes`, and `broadcastAxes` are populated:

- parallel axes get `BindCoreCandidate`, `KernelLoopCandidate`, `VectorizeCandidate`, `MaskedTail`
- reduction axes get `FullReduction`, `FullExtent`
- broadcast axes add `BroadcastProjection` without removing consumer-side parallel roles
- unknown axes get no roles and keep `MustDivide`

Keep the derivation independent of target hardware capacity.

- [x] **Step 4: Derive conservative coalescing hints**

Create one `AxisCoalescingHint` for the maximal ordered run of at least two parallel axes when `info.barriers.empty()` and none of the axes is reduction.

Use:

```cpp
groupId = 1;
kind = any member has VectorizeCandidate ? Vectorizable : LinearizeOnly;
memberAxisIds = ordered parallel logical axis ids;
```

Write the same `groupId` into each member `AxisScheduleConstraint.coalescingGroupId`.

- [x] **Step 5: Print stable report strings**

Print roles as:

```text
bind_core,kernel_loop,vectorize,full_reduction,chunked_reduction,broadcast_projection,layout_carry
```

Print tail policies as:

```text
must_divide,masked_tail,scalar_epilogue,full_extent
```

Print hint kinds as:

```text
vectorizable,linearize_only
```

- [x] **Step 6: Verify GREEN**

Run the focused LIT from Step 1. Expected: PASS.

Actual:

- RED confirmed with direct xvm pipeline + `FileCheck`: rank-3 `[parallel, parallel, reduction]` expected `group=1`, old output had no coalescing hint.
- GREEN passed in xvm after `ninja -C build afir-opt`: `ascend-schedule-axis-coalescing.mlir` and dependent `ascend-schedule-problem.mlir` 2/2 passed.
- Spec compliance review passed after removing over-broad reduction suppression.
- Code quality re-review passed with notes; no required fixes remain.

## Task 2: ScheduleProblem Contract Consumption

**Files:**
- Modify `include/Conversion/Ascend/Schedule/ScheduleTypes.h`
- Modify `lib/Conversion/Ascend/Schedule/ScheduleProblemBuilder.cpp`
- Modify `test/Conversion/ascend-schedule-problem.mlir`

- [x] **Step 1: Write the failing LIT**

Extend `test/Conversion/ascend-schedule-problem.mlir` to require the problem report to include:

```mlir
// CHECK: ScheduleProblem:
// CHECK: axis_constraints = [
// CHECK-SAME: axis=0 roles=[bind_core,kernel_loop,vectorize] tail=masked_tail
// CHECK-SAME: axis=1 roles=[full_reduction] tail=full_extent
// CHECK: coalescing_hints =
```

Run in xvm:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-schedule-problem.mlir'
```

Expected before implementation: FAIL because `ScheduleProblem` does not print axis constraints.

- [x] **Step 2: Store the contract in ScheduleProblem**

The existing `ScheduleProblem` already owns `CoalescedAxisInfo axes`; keep the contract there. Do not duplicate the vectors at the `ScheduleProblem` top level.

- [x] **Step 3: Print contract fields from ScheduleProblem**

Update `printScheduleProblemReport` to print `axis_constraints` and `coalescing_hints` using the same formatting from Task 1.

- [x] **Step 4: Verify GREEN**

Run the focused LIT from Step 1. Expected: PASS.

Actual:

- RED confirmed in xvm: `ascend-schedule-problem.mlir` failed because `ScheduleProblem` did not print `axis_constraints`.
- GREEN passed in xvm after `ninja -C build afir-opt`: `ascend-schedule-problem.mlir` and `ascend-schedule-axis-coalescing.mlir` 2/2 passed.
- Spec compliance review passed with notes; no required fixes.
- Code quality review passed with notes; no required fixes.

## Task 3: Tail-Policy-Aware Guard Generation

**Files:**
- Modify `lib/Conversion/Ascend/Schedule/ScheduleSearch.cpp`
- Modify `lib/Conversion/Ascend/Schedule/ScheduleDecision.cpp`
- Modify `test/Conversion/ascend-schedule-guards.mlir`

- [x] **Step 1: Write the failing LIT**

Add a schedule guard case where a parallel axis has static extent `640`, selected tile size `64`, and `tailPolicy = MaskedTail`.

Expected:

```mlir
// CHECK: ScheduleDecisionSet:
// CHECK: decision_guards = 0
// CHECK-NOT: % 64 == 0
```

Add a second case where a synthetic `MustDivide` axis still emits:

```mlir
// MUST-DIVIDE: kind = divisible_by
// MUST-DIVIDE: a0 % 64 == 0
```

Run in xvm:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-schedule-guards.mlir'
```

Expected before implementation: FAIL because `appendDecisionGuards` only looks at tile size and emits divisibility guards for any static tile greater than 1.

- [x] **Step 2: Pass axis constraints into decision guard generation**

Change:

```cpp
void appendDecisionGuards(ArrayRef<int64_t> tileSizes,
                          SmallVectorImpl<ScheduleGuard> &guards)
```

to consume the problem:

```cpp
void appendDecisionGuards(const ScheduleProblem &problem,
                          ArrayRef<int64_t> tileSizes,
                          SmallVectorImpl<ScheduleGuard> &guards)
```

Only emit `GuardKind::DivisibleBy` when the matching `AxisScheduleConstraint.tailPolicy` is `AxisTailPolicy::MustDivide`.

- [x] **Step 3: Report guard counts**

Extend `printScheduleDecisionSetReport` to print:

```text
  candidate_guards = <count>
  decision_guards = <count>
```

Use the selected decision (`decisions.front()`) for the count.

- [x] **Step 4: Verify GREEN**

Run the focused LIT from Step 1. Expected: PASS.

Actual:

- RED confirmed in xvm with a direct source pipeline + `FileCheck`: old code emitted `decision_guards = 2` for `kernel_0` where `MaskedTail` axes now expect `decision_guards = 0`.
- GREEN passed in xvm after `ninja -C build afir-opt`: `ascend-schedule-guards.mlir`, `ascend-schedule-problem.mlir`, and `ascend-schedule-axis-coalescing.mlir` 3/3 passed.
- `ScheduleSearch` now emits `DivisibleBy` only for explicit `AxisTailPolicy::MustDivide`; current natural pipeline produces `MaskedTail` parallel axes and `FullExtent` reduction axes, so `MustDivide` positive coverage is deferred until a direct schedule-model test or contract-attribute path exists.
- Spec compliance review and code quality review both passed with notes; tracked follow-ups are positive `MustDivide` coverage, stronger missing-constraint invariant handling, and carrying explicit logical-axis identity if future tile vectors can be reordered.

## Task 4: Role-Driven Reduction Tile Search

**Files:**
- Modify `lib/Conversion/Ascend/Schedule/ScheduleSearch.cpp`
- Modify `test/Conversion/ascend-schedule-search.mlir`

- [x] **Step 1: Write the failing LIT**

Add a reduction schedule search case with logical axes:

```text
axis 0: parallel, extent 640, roles=[bind_core,kernel_loop,vectorize], tail=masked_tail
axis 1: reduction, extent 15000, roles=[full_reduction], tail=full_extent
```

Expected selected tile shape:

```mlir
// CHECK: ScheduleDecisionSet:
// CHECK: selected_tile_shape = [64, 15000]
// CHECK: decision_guards = 0
```

Run in xvm:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-schedule-search.mlir'
```

Expected before implementation: FAIL because reduction schedules currently prefer full logical axis tile `[640,15000]` or split reduction, not bounded M-axis tiling.

- [x] **Step 2: Add bounded parallel tile generation**

For `OpRole::Reduction`, generate a tile shape where:

- `FullReduction` axes use full static extent
- parallel axes with `BindCoreCandidate` or `KernelLoopCandidate` use `min(staticExtent, 64)` when static
- dynamic parallel axes use `ShapedType::kDynamic`
- reduction axes without `FullReduction` keep existing behavior

The constant `64` must be named:

```cpp
constexpr int64_t kDefaultParallelTile = 64;
```

- [x] **Step 3: Prefer bounded tiles for large reductions**

Update ranking so a reduction instance with a bounded parallel tile and full reduction axis ranks ahead of a full `[M,N]` tile when `M > kDefaultParallelTile`.

- [x] **Step 4: Print selected tile shape**

Extend `printScheduleDecisionSetReport` to print:

```text
  selected_tile_shape = [64,15000]
```

Use `?` for dynamic dimensions.

- [x] **Step 5: Verify GREEN**

Run the focused LIT from Step 1. Expected: PASS.

Actual:

- RED confirmed in xvm with direct `afir-opt | FileCheck`: old report had no `selected_tile_shape = [64,15000]` for the new large reduction case.
- GREEN passed in xvm after `ninja -C build afir-opt`: `ascend-schedule-search.mlir`, `ascend-schedule-guards.mlir`, `ascend-schedule-problem.mlir`, and `ascend-schedule-axis-coalescing.mlir` 4/4 passed.
- Reduction search now preserves full logical and split-reduction candidates while adding the role-driven bounded parallel reduction candidate; the large reduction case reports `generated = 3`, `kept = 3`, `decision_guards = 0`, and `selected_tile_shape = [64,15000]`.
- Spec compliance review and code quality review both passed with notes; the post-review test was tightened to assert the `kernel_4` candidate count. Follow-ups are dynamic reduction coverage and making `64` target/config-driven.

## Task 5: Persist Selected Tile Metadata For Phase 5 Artifacts

**Files:**
- Modify `include/Conversion/Ascend/Common/Attributes.h`
- Modify `lib/Conversion/Ascend/Schedule/StructuredLoweringDriver.cpp`
- Modify `lib/Target/CannKernel/CannRuntimeArtifacts.cpp`
- Modify `test/Conversion/ascend-full-pipeline-broadcast-add-reduce.mlir`

- [x] **Step 1: Write the failing full-pipeline LIT**

Extend `test/Conversion/ascend-full-pipeline-broadcast-add-reduce.mlir` to check for selected tile metadata after the full pipeline:

```mlir
// CHECK: ascend.schedule.selected_tile_shape
// CHECK-SAME: [64
// CHECK: ascend.schedule.tail_policies
// CHECK-SAME: masked_tail
// CHECK-SAME: full_extent
```

Run in xvm:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-full-pipeline-broadcast-add-reduce.mlir'
```

Expected before implementation: FAIL because structured lowering currently only writes `ascend.schedule.structured_lowering`.

- [x] **Step 2: Add shared attr names**

Add:

```cpp
inline constexpr llvm::StringLiteral kScheduleSelectedTileShapeAttr =
    "ascend.schedule.selected_tile_shape";
inline constexpr llvm::StringLiteral kScheduleTailPoliciesAttr =
    "ascend.schedule.tail_policies";
```

- [x] **Step 3: Attach attrs in StructuredLoweringDriver**

For each op in the scheduled pattern, attach:

- `ascend.schedule.selected_tile_shape = DenseI64ArrayAttr`
- `ascend.schedule.tail_policies = ArrayAttr<StringAttr>`

The selected tile shape comes from `decisionSet.decisions.front().instance.tileShape.tileSizes`.

- [x] **Step 4: Surface tile metadata in runtime artifacts**

In `CannRuntimeArtifacts.cpp`, when a primary kernel function contains the selected tile attrs, add these fields to each `scheduleEntries` entry:

```json
"tilingParams": {
  "selected_tile_shape": [64, 15000],
  "tail_policies": ["masked_tail", "full_extent"]
}
```

Keep `decisionId = "static_0"` until Phase 5 supports multiple runtime decisions.

- [x] **Step 5: Verify GREEN**

Run the focused full-pipeline LIT from Step 1. Expected: PASS.

Actual:

- RED confirmed in xvm: full-pipeline output lacked `ascend.schedule.selected_tile_shape`, and runtime manifest `tilingParams` was `{}` without selected tile metadata.
- GREEN passed in xvm after `ninja -C build afir-opt afir-translate AscendCommonAttributesTest`: `ascend-full-pipeline-broadcast-add-reduce.mlir`, schedule search/guards/problem/axis-coalescing, `cann-translate-runtime-artifacts.mlir`, and `cann-translate-runtime-artifacts-unsupported.mlir` 7/7 passed; `AscendCommonAttributesTest` and `ctest -R AscendCommonAttributesTest` passed.
- Structured lowering now attaches selected tile shape and tail policy attrs to scheduled ops and preserves a deterministic first scheduled kernel metadata pair on parent `func.func` for the current one-primary-global-kernel Phase 5 artifact bridge.
- Runtime manifest now emits `tilingParams.selected_tile_shape` and `tilingParams.tail_policies` when attrs are present, keeps `{}` when absent, and fails with a diagnostic for malformed non-string tail policy elements.
- Spec compliance review passed with notes. Code quality review found one required robustness issue in `tail_policies` parsing; it was fixed and re-review passed with notes. Follow-ups are stricter manually-authored top-level attr validation and replacing the func-level bridge with a per-kernel artifact contract.

## Task 6: Broadcast Add Reduce Large Shape Acceptance

**Files:**
- Modify `examples/broadcast-add-reduce/run-mainline.sh`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [x] **Step 1: Switch the mainline default shape**

Change the script default from:

```bash
M=64
N=15000
BLOCK_DIM=1
```

to:

```bash
M=640
N=15000
BLOCK_DIM=20
```

Keep `--m`, `--n`, and `--block-dim` overrides.

- [x] **Step 2: Run the mainline example**

Run in xvm:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && AFIR_OPT=$PWD/build/bin/afir-opt AFIR_TRANSLATE=$PWD/build/bin/afir-translate RUNTIME_SESSION=$PWD/build/bin/runtime-session bash examples/broadcast-add-reduce/run-mainline.sh --log'
```

Expected after implementation:

```text
session.backend=sim
session.result=success
session.validation=pass
```

- [x] **Step 3: Run small-shape regression**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && AFIR_OPT=$PWD/build/bin/afir-opt AFIR_TRANSLATE=$PWD/build/bin/afir-translate RUNTIME_SESSION=$PWD/build/bin/runtime-session bash examples/broadcast-add-reduce/run-mainline.sh --m 64 --n 15000 --block-dim 1 --log'
```

Expected:

```text
session.backend=sim
session.result=success
session.validation=pass
```

- [x] **Step 4: Update tracking**

Record:

- plan file path
- V2-3 axis contract spec closure
- focused LIT results
- large-shape runtime-session result
- remaining limitation: contract fields are currently Schedule-derived, not serialized from Kernelize `scheduleContract`

Actual:

- Default shape is now `M=640,N=15000,BLOCK_DIM=20`.
- RED/GREEN follow-up found two Phase 5 tail runtime issues before the default shape could be accepted:
  - dynamic selected reduction tiles needed materialization into per-block `scf.for` slices instead of full `M*N` buffers;
  - rows<16 broadcast tails cannot rely on the small VECIN copy path for scalar `A[M]`, so CANN emission now aligns runtime buffers to 32B and uses a GM-scalar `GetValue` + row-wise `Adds` fallback when it can trace the broadcast source back to the GM `DataCopy`.
- xvm runtime-session sim passed:
  - default `M=640,N=15000,BLOCK_DIM=20`
  - `M=65,N=128,BLOCK_DIM=2`
  - `M=70,N=128,BLOCK_DIM=2`
  - `M=72,N=128,BLOCK_DIM=2`
  - `M=70,N=123,BLOCK_DIM=2`
  - `M=128,N=123,BLOCK_DIM=2`
- Tracking updated in `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`.

## Task 7: Verification And Review

**Files:**
- Test-only unless review finds issues.

- [x] **Step 1: Run static checks**

```bash
git diff --check -- . ':!AGENTS.md'
test/tools/check_ascend_no_v2_code_naming.sh
```

Expected: no output from `git diff --check`; naming guard passes.

- [x] **Step 2: Run focused Schedule and full-pipeline LIT**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt afir-translate runtime-session && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-schedule-axis-coalescing.mlir build/test/Conversion/ascend-schedule-problem.mlir build/test/Conversion/ascend-schedule-guards.mlir build/test/Conversion/ascend-schedule-search.mlir build/test/Conversion/ascend-full-pipeline-broadcast-add-reduce.mlir'
```

Expected: all listed tests pass.

- [x] **Step 3: Run Ascend conversion regression**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion --filter="ascend-"'
```

Expected: all discovered Ascend conversion tests pass.

- [x] **Step 4: Run subagent reviews**

Use Subagent-Driven review checkpoints:

- spec compliance review against `docs/Ascend-MLIR-Detailed-Implementation-V2-3.zh.md`
- code quality review for axis contract model, guard behavior, schedule ranking, attr persistence, and runtime artifact changes

Fix review findings before marking the plan complete.

Actual so far:

- `git diff --check` passed.
- `test/tools/check_ascend_no_v2_code_naming.sh` passed.
- xvm `ninja -C build afir-opt afir-translate` passed.
- xvm source LIT passed: `test/Conversion/ascend-*.mlir` plus `test/Target/cann-translate*.mlir`, 57/57 passed.
- xvm Ascend runtime unit tests passed: 8/8.
- xvm runtime-session sim matrix passed again after review fixes:
  - `M=65,N=128,BLOCK_DIM=2`
  - `M=70,N=128,BLOCK_DIM=2`
  - `M=72,N=128,BLOCK_DIM=2`
  - `M=70,N=123,BLOCK_DIM=2`
  - `M=128,N=123,BLOCK_DIM=2`
  - default `M=640,N=15000,BLOCK_DIM=20`
- New target regression LIT:
  - `test/Target/cann-translate-align-runtime-buffers.mlir`
  - `test/Target/cann-translate-broadcast-tail-fallback.mlir`
- New/updated full-pipeline regression LIT:
  - `test/Conversion/ascend-compute-lower-selected-tile-materializes-loop.mlir`
  - `test/Conversion/ascend-compute-lower-selected-tile-rejects-partial-reduction.mlir`
  - `test/Conversion/ascend-compute-lower-selected-tile-rejects-unsupported-map.mlir`
  - `test/Conversion/ascend-schedule-dynamic-reduction-tile.mlir`
  - `test/Conversion/ascend-full-pipeline-broadcast-add-reduce.mlir`
- Spec compliance review passed.
- Code quality review found two Important materializer boundary issues; both fixed by fail-closed validation and covered by the three selected-tile compute-lower LITs above.
- Code quality re-review passed with no required fixes. Remaining risks are output-side unsupported-map negative coverage and future chunked dynamic reduction support, both deferred.
