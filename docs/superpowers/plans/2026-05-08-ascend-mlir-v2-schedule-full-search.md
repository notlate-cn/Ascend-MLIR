# Ascend MLIR V2 Schedule Full Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to execute this plan.

**Goal:** Upgrade the current Ascend V2 schedule MVP from a per-op fixed schedule annotator into a pattern-level, searchable, guard-aware schedule pipeline that can consume Phase 1 kernel pattern attributes and produce deterministic `ScheduleDecisionSet`-style results.

**Architecture:** Keep implementation local to Ascend V2 conversion code. Do not modify upstream MLIR. The schedule layer reconstructs a lightweight `KernelPatternView` from kernelize attributes, builds axis and schedule problem models, queries a local template registry, searches ranked schedule instances, emits schedule decision attributes, and reports/cache-models decisions for lit validation.

**Tech Stack:** C++17, MLIR pass infrastructure, LLVM ADT, lit/FileCheck, xvm docker validation through `/home/niu/code/Ascend-MLIR`.

## Current Baseline

The current schedule implementation is concentrated in:

- `lib/Conversion/AscendV2/Schedule/SchedulePass.cpp`

Current behavior:

- Requires per-op `ascend.v2.kernel`.
- Reads `ascend.v2.op_role`.
- Chooses one fixed template per linalg op:
  - `vector_static_1d`
  - `vector_static_2d`
  - `reduction_static`
  - `cube_static_matmul`
- Emits per-op attributes:
  - `ascend.v2.schedule.family`
  - `ascend.v2.schedule.template`
  - `ascend.v2.schedule.decision_id`

Existing tests that must keep passing:

- `test/Conversion/ascend-schedule-mvp.mlir`
- `test/Conversion/ascend-v2-pipeline-mvp.mlir`

## Non-Goals For This Phase

This phase does not lower to AscendC or runtime ABI. Those remain Layer 4 and Layer 5 responsibilities.

This phase does not persist a real on-disk tuning database. It introduces deterministic cache keys, cache reports, negative-cache semantics, and pass-local model behavior so later runtime/tuning work has stable contracts.

This phase does not require serialized `KernelPattern[]` function attributes from kernelize. Schedule reconstructs `KernelPatternView` from the existing per-op attrs produced by Phase 1.

## File Layout

Create these headers:

- `include/Conversion/AscendV2/Schedule/ScheduleTypes.h`
- `include/Conversion/AscendV2/Schedule/KernelPatternView.h`
- `include/Conversion/AscendV2/Schedule/AxisCoalescer.h`
- `include/Conversion/AscendV2/Schedule/ScheduleProblemBuilder.h`
- `include/Conversion/AscendV2/Schedule/TemplateRegistry.h`
- `include/Conversion/AscendV2/Schedule/ScheduleSearch.h`
- `include/Conversion/AscendV2/Schedule/ScheduleDecision.h`
- `include/Conversion/AscendV2/Schedule/ScheduleCache.h`
- `include/Conversion/AscendV2/Schedule/StructuredLoweringDriver.h`

Create these implementations:

- `lib/Conversion/AscendV2/Schedule/KernelPatternView.cpp`
- `lib/Conversion/AscendV2/Schedule/AxisCoalescer.cpp`
- `lib/Conversion/AscendV2/Schedule/ScheduleProblemBuilder.cpp`
- `lib/Conversion/AscendV2/Schedule/TemplateRegistry.cpp`
- `lib/Conversion/AscendV2/Schedule/ScheduleSearch.cpp`
- `lib/Conversion/AscendV2/Schedule/ScheduleDecision.cpp`
- `lib/Conversion/AscendV2/Schedule/ScheduleCache.cpp`
- `lib/Conversion/AscendV2/Schedule/StructuredLoweringDriver.cpp`

Modify:

- `lib/Conversion/AscendV2/Schedule/SchedulePass.cpp`
- `lib/Conversion/AscendV2/CMakeLists.txt`
- `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

Add focused tests:

- `test/Conversion/ascend-schedule-pattern-view.mlir`
- `test/Conversion/ascend-schedule-axis-coalescing.mlir`
- `test/Conversion/ascend-schedule-problem.mlir`
- `test/Conversion/ascend-schedule-template-registry.mlir`
- `test/Conversion/ascend-schedule-search.mlir`
- `test/Conversion/ascend-schedule-guards.mlir`
- `test/Conversion/ascend-schedule-decision-set.mlir`
- `test/Conversion/ascend-schedule-cache.mlir`
- `test/Conversion/ascend-schedule-structured-lowering.mlir`

## Shared Attribute Contract

Define all schedule attribute names once in `ScheduleTypes.h`:

```c++
inline constexpr llvm::StringLiteral kKernelAttr = "ascend.v2.kernel";
inline constexpr llvm::StringLiteral kPrimaryAttr = "ascend.v2.primary";
inline constexpr llvm::StringLiteral kOpRoleAttr = "ascend.v2.op_role";

inline constexpr llvm::StringLiteral kScheduleFamilyAttr =
    "ascend.v2.schedule.family";
inline constexpr llvm::StringLiteral kScheduleTemplateAttr =
    "ascend.v2.schedule.template";
inline constexpr llvm::StringLiteral kScheduleDecisionIdAttr =
    "ascend.v2.schedule.decision_id";
inline constexpr llvm::StringLiteral kScheduleRuntimeTopKAttr =
    "ascend.v2.schedule.runtime_top_k";
inline constexpr llvm::StringLiteral kStructuredLoweringAttr =
    "ascend.v2.schedule.structured_lowering";
```

Keep existing schedule family/template string values stable so existing tests and downstream assumptions continue to work.

## Task 1: Shared Schedule Types And KernelPatternView

**Owner:** one worker subagent.

**Purpose:** Replace per-op-only schedule traversal with pattern-level grouping while preserving current output attributes.

**Files:**

- Add `ScheduleTypes.h`
- Add `KernelPatternView.h`
- Add `KernelPatternView.cpp`
- Modify `SchedulePass.cpp`
- Modify `CMakeLists.txt`
- Add `ascend-schedule-pattern-view.mlir`

**Implementation details:**

`ScheduleTypes.h` starts with shared enums and model structs:

```c++
namespace mlir::ascend::v2::schedule {

enum class OpRole {
  Unknown,
  Vector,
  Reduction,
  Cube,
  Memory,
};

struct PatternOpView {
  Operation *op = nullptr;
  unsigned ordinal = 0;
  OpRole role = OpRole::Unknown;
  bool primary = false;
};

struct KernelPatternView {
  std::string kernelId;
  SmallVector<PatternOpView> ops;
  SmallVector<Operation *> primaryOps;
  OpRole dominantRole = OpRole::Unknown;
};

} // namespace mlir::ascend::v2::schedule
```

`KernelPatternView` builder behavior:

- Walk `func.func` in IR order.
- Only consider `linalg::LinalgOp`.
- Require every scheduled linalg op to have `ascend.v2.kernel`.
- Group ops by `ascend.v2.kernel`.
- Sort groups by the first op ordinal.
- Sort ops inside each group by ordinal.
- Read `ascend.v2.op_role`:
  - `"cube"` -> `OpRole::Cube`
  - `"reduction"` -> `OpRole::Reduction`
  - `"vector"` -> `OpRole::Vector`
  - `"memory"` -> `OpRole::Memory`
  - missing or unknown -> `OpRole::Unknown`
- Read `ascend.v2.primary` as boolean.
- If a group has no primary op, emit pass failure with a deterministic message.
- Dominant role priority:
  - cube
  - reduction
  - vector
  - memory
  - unknown

Add a report helper used by the pass debug path:

```c++
void printKernelPatternViews(ArrayRef<KernelPatternView> patterns,
                             llvm::raw_ostream &os);
```

Expected report shape:

```text
SchedulePatternView:
  kernel = kernel_0
  ops = 2
  primary_ops = 1
  dominant_role = vector
```

`SchedulePass.cpp` should initially use the first primary op to preserve the existing fixed-template behavior and then annotate every op in the pattern with the same schedule family/template/decision id.

**Test:**

Create `test/Conversion/ascend-schedule-pattern-view.mlir` with a two-op elementwise chain that kernelize fuses into one kernel. The test must run:

```text
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s
```

Checks:

```text
// CHECK: SchedulePatternView:
// CHECK: kernel = kernel_0
// CHECK: ops = 2
// CHECK: primary_ops = 1
// CHECK: dominant_role = vector
// CHECK: ascend.v2.schedule.family = "vector_static_1d"
```

**Verification in xvm/docker:**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
  /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v \
  test/Conversion/ascend-schedule-mvp.mlir \
  test/Conversion/ascend-v2-pipeline-mvp.mlir \
  test/Conversion/ascend-schedule-pattern-view.mlir'
```

**Review checkpoint:**

- No behavior regression in existing schedule tests.
- Pattern grouping is deterministic.
- `AGENTS.md` remains untouched.

## Task 2: AxisCoalescer MVP

**Owner:** fresh worker subagent after Task 1 is reviewed.

**Purpose:** Build conservative logical axis metadata from each `KernelPatternView`.

**Files:**

- Add `AxisCoalescer.h`
- Add `AxisCoalescer.cpp`
- Modify `SchedulePass.cpp`
- Add `ascend-schedule-axis-coalescing.mlir`

**Implementation details:**

Add types in `ScheduleTypes.h`:

```c++
enum class AxisKind {
  Parallel,
  Reduction,
  Broadcast,
  Unknown,
};

enum class AxisBarrierKind {
  None,
  UnsupportedIndexingMap,
  UnsupportedIteratorType,
  RankMismatch,
};

struct LogicalAxisInfo {
  unsigned logicalAxisId = 0;
  AxisKind kind = AxisKind::Unknown;
  int64_t staticExtent = ShapedType::kDynamic;
  SmallVector<std::pair<Operation *, unsigned>> rawAxes;
};

struct AxisCoalescingBarrier {
  Operation *op = nullptr;
  AxisBarrierKind kind = AxisBarrierKind::None;
  std::string reason;
};

struct CoalescedAxisInfo {
  SmallVector<LogicalAxisInfo> logicalAxes;
  SmallVector<unsigned> parallelAxes;
  SmallVector<unsigned> reductionAxes;
  SmallVector<unsigned> broadcastAxes;
  SmallVector<AxisCoalescingBarrier> barriers;
};
```

MVP algorithm:

1. Use primary op result rank as the pattern rank for vector/cube cases.
2. Use linalg iterator types to classify axes:
   - `parallel` -> `AxisKind::Parallel`
   - `reduction` -> `AxisKind::Reduction`
   - unsupported iterator -> barrier
3. For identity/projected permutation indexing maps, preserve axis order.
4. For non-projected-permutation maps, record an `UnsupportedIndexingMap` barrier and skip aggressive coalescing.
5. Coalesce adjacent parallel axes only when all participating shaped operands have static extents for those dimensions or all are dynamic in the same positions.
6. Never coalesce across reduction axes in this MVP.

Expected report:

```text
AxisCoalescing:
  kernel = kernel_0
  logical_axes = 2
  parallel_axes = [0, 1]
  reduction_axes = []
  barriers = 0
```

**Test:**

`ascend-schedule-axis-coalescing.mlir` must cover:

- rank-2 elementwise vector case
- reduction case with one parallel and one reduction axis

Checks:

```text
// CHECK: AxisCoalescing:
// CHECK: logical_axes = 2
// CHECK: parallel_axes = [0, 1]
// CHECK: reduction_axes = []
// CHECK: AxisCoalescing:
// CHECK: parallel_axes = [0]
// CHECK: reduction_axes = [1]
```

**Verification:**

Run focused lit for the new test and all existing schedule tests in xvm/docker.

## Task 3: ScheduleProblemBuilder MVP

**Owner:** fresh worker subagent after Task 2 is reviewed.

**Purpose:** Convert pattern and axis metadata into the Layer 3 boundary object `ScheduleProblem`.

**Files:**

- Add `ScheduleProblemBuilder.h`
- Add `ScheduleProblemBuilder.cpp`
- Modify `SchedulePass.cpp`
- Add `ascend-schedule-problem.mlir`

**Implementation details:**

Add types:

```c++
struct ScheduleProblem {
  std::string kernelId;
  OpRole dominantRole = OpRole::Unknown;
  unsigned resultRank = 0;
  SmallVector<int64_t> resultShape;
  CoalescedAxisInfo axes;
  unsigned guardBudget = 8;
  SmallVector<std::string> templateTags;
  SmallVector<std::string> structureConstraints;
  SmallVector<std::string> shapeConstraints;
};
```

Builder rules:

- `kernelId` comes from `KernelPatternView`.
- `dominantRole` comes from the pattern view.
- `resultRank` and `resultShape` come from `selectDominantPrimaryOp(pattern)` result shaped type.
- `templateTags`:
  - cube role -> `cube`
  - reduction role -> `reduction`
  - vector role -> `vector`
- `shapeConstraints`:
  - static dimensions are emitted as `dN == value`
  - dynamic dimensions are emitted as `dN dynamic`
- `structureConstraints`:
  - multi-op vector chain -> `elementwise_chain`
  - reduction -> `single_reduction_region`
  - cube/matmul -> `matmul_contract`

Expected report:

```text
ScheduleProblem:
  kernel = kernel_0
  role = vector
  result_rank = 2
  result_shape = [4, 8]
  guard_budget = 8
  template_tags = [vector]
```

**Test:**

`ascend-schedule-problem.mlir` must cover:

- static vector rank-2
- dynamic vector case
- matmul/cube case

**Verification:**

Focused lit for schedule tests in xvm/docker.

## Task 4: TemplateRegistry MVP

**Owner:** fresh worker subagent after Task 3 is reviewed.

**Purpose:** Replace ad hoc family selection with a registry matching `ScheduleProblem` to stable schedule families/templates.

**Files:**

- Add `TemplateRegistry.h`
- Add `TemplateRegistry.cpp`
- Modify `SchedulePass.cpp`
- Add `ascend-schedule-template-registry.mlir`

**Implementation details:**

Add types:

```c++
struct ScheduleTemplate {
  std::string family;
  std::string name;
  SmallVector<std::string> tags;
  unsigned minRank = 0;
  unsigned maxRank = 0;
};
```

Registry contents:

- `{family="vector_static_1d", name="single_tile_per_block", tags=["vector"], minRank=1, maxRank=1}`
- `{family="vector_static_2d", name="single_tile_per_block", tags=["vector"], minRank=2, maxRank=2}`
- `{family="reduction_static", name="single_tile_per_block", tags=["reduction"], minRank=1, maxRank=8}`
- `{family="cube_static_matmul", name="single_tile_per_block", tags=["cube"], minRank=2, maxRank=2}`

Matching rules:

- Template tag must intersect `ScheduleProblem::templateTags`.
- Rank must be within `[minRank, maxRank]`.
- Prefer exact role match.
- Sort matches by family string then template name for deterministic output.
- If no template matches, fail pass with `no schedule template for kernel <id>`.

Expected report:

```text
TemplateRegistry:
  kernel = kernel_0
  matches = 1
  template = vector_static_2d/single_tile_per_block
```

**Test:**

`ascend-schedule-template-registry.mlir` checks vector rank1, vector rank2, reduction, and matmul.

**Verification:**

Focused lit for schedule tests in xvm/docker.

## Task 5: ScheduleSearch And compileTimeTopK

**Owner:** fresh worker subagent after Task 4 is reviewed.

**Purpose:** Generate ranked symbolic `ScheduleInstance`s instead of immediately accepting the first template.

**Files:**

- Add `ScheduleSearch.h`
- Add `ScheduleSearch.cpp`
- Modify `SchedulePass.cpp`
- Add `ascend-schedule-search.mlir`

**Implementation details:**

Add types:

```c++
struct TileShape {
  SmallVector<int64_t> tileSizes;
};

struct ScheduleInstance {
  std::string instanceId;
  ScheduleTemplate tmpl;
  TileShape tileShape;
  int64_t estimatedCost = 0;
  SmallVector<std::string> reasonKinds;
};
```

Search config:

```c++
struct ScheduleSearchOptions {
  unsigned compileTimeTopK = 4;
};
```

Candidate generation:

- For static vector axes:
  - full tile: all static extents
  - half tile: each static extent halved when extent >= 2
- For dynamic vector axes:
  - symbolic tile `-1` for dynamic axis
- For reduction:
  - full reduction tile
  - split reduction tile when reduction extent is static and >= 2
- For cube/matmul:
  - use static matrix problem tile when all dims are static
  - use `-1` for dynamic dimensions

Cost model MVP:

- Prefer fewer dynamic tile sizes.
- Prefer larger static tile area.
- Prefer lower rank only as tie breaker.
- Stable sort key:
  1. `estimatedCost`
  2. `family`
  3. `template`
  4. `tileShape` lexicographically

Top-K:

- Keep only first `compileTimeTopK` instances.
- Report both generated and kept counts.

Expected report:

```text
ScheduleSearch:
  kernel = kernel_0
  generated = 2
  kept = 2
  compile_time_top_k = 4
  instance = kernel_0.vector_static_2d.0
```

**Test:**

`ascend-schedule-search.mlir` checks:

- static rank-2 vector produces at least 2 generated instances
- compile-time top K is reported
- decision remains deterministic across repeated functions

**Verification:**

Focused lit in xvm/docker.

## Task 6: Guard Generation And Guard Budget

**Owner:** fresh worker subagent after Task 5 is reviewed.

**Purpose:** Distinguish `candidateGuards` and `decisionGuards`, and prune decisions that exceed guard budget.

**Files:**

- Extend `ScheduleSearch.h/.cpp`
- Extend `ScheduleTypes.h`
- Modify `SchedulePass.cpp`
- Add `ascend-schedule-guards.mlir`

**Implementation details:**

Add:

```c++
enum class GuardKind {
  ShapeStaticEqual,
  ShapeDynamic,
  DivisibleBy,
  PositiveExtent,
};

struct ScheduleGuard {
  GuardKind kind = GuardKind::ShapeDynamic;
  unsigned dim = 0;
  int64_t value = ShapedType::kDynamic;
  std::string text;
};
```

Guard rules:

- Static dim `N` -> candidate guard `dN == value`.
- Dynamic dim `N` -> candidate guard `dN > 0`.
- Tile size `T > 1` on dim `N` -> decision guard `dN % T == 0`.
- Dynamic symbolic tile `-1` does not add divisibility guard.
- Drop any `ScheduleInstance` where `candidateGuards + decisionGuards > ScheduleProblem::guardBudget`.

Expected report:

```text
ScheduleGuards:
  kernel = kernel_0
  candidate_guards = 2
  decision_guards = 1
  guard_budget = 8
  pruned_by_guard_budget = 0
```

**Test:**

`ascend-schedule-guards.mlir` checks static and dynamic shapes. Include one case with enough dimensions to prove pruning is reported.

**Verification:**

Focused lit in xvm/docker.

## Task 7: ScheduleDecisionSet Builder

**Owner:** fresh worker subagent after Task 6 is reviewed.

**Purpose:** Materialize searched instances into deterministic schedule decisions and preserve existing per-op attrs.

**Files:**

- Add `ScheduleDecision.h`
- Add `ScheduleDecision.cpp`
- Modify `SchedulePass.cpp`
- Add `ascend-schedule-decision-set.mlir`

**Implementation details:**

Add:

```c++
struct ScheduleDecision {
  std::string decisionId;
  ScheduleInstance instance;
  SmallVector<ScheduleGuard> candidateGuards;
  SmallVector<ScheduleGuard> decisionGuards;
};

struct ScheduleDecisionSet {
  std::string kernelId;
  SmallVector<ScheduleDecision> decisions;
  unsigned runtimeTopK = 1;
};
```

Builder rules:

- `runtimeTopK = min(1, decisions.size())` for MVP.
- First kept decision is canonical and drives per-op schedule attrs.
- Decision ID format:
  - `<kernelId>.decision.<ordinal>`
- Apply to every linalg op in the same `KernelPatternView`:
  - `ascend.v2.schedule.family`
  - `ascend.v2.schedule.template`
  - `ascend.v2.schedule.decision_id`
  - `ascend.v2.schedule.runtime_top_k`

Expected report:

```text
ScheduleDecisionSet:
  kernel = kernel_0
  decisions = 2
  runtime_top_k = 1
  selected = kernel_0.decision.0
```

**Test:**

`ascend-schedule-decision-set.mlir` checks:

- two ops in one kernel receive the same decision id
- runtime top K attr exists
- existing `ascend-schedule-mvp.mlir` checks still pass

**Verification:**

Focused lit in xvm/docker.

## Task 8: Schedule Cache Model

**Owner:** fresh worker subagent after Task 7 is reviewed.

**Purpose:** Add deterministic `ShapeBucketCache` and `TuningResultCache` modeling without persistent storage.

**Files:**

- Add `ScheduleCache.h`
- Add `ScheduleCache.cpp`
- Modify `SchedulePass.cpp`
- Add `ascend-schedule-cache.mlir`

**Implementation details:**

Add:

```c++
struct ShapeBucketKey {
  std::string kernelId;
  std::string family;
  SmallVector<int64_t> resultShape;
};

struct TuningResultKey {
  ShapeBucketKey bucket;
  std::string templateName;
  SmallVector<int64_t> tileShape;
};

struct ScheduleCacheReport {
  unsigned shapeBucketLookups = 0;
  unsigned shapeBucketMisses = 0;
  unsigned tuningLookups = 0;
  unsigned tuningMisses = 0;
  unsigned negativeCacheEntries = 0;
};
```

MVP behavior:

- Build keys for every selected decision.
- Treat first lookup for a key as miss.
- Treat repeated key in the same pass run as hit.
- Record negative cache entry for no-template and guard-budget-pruned cases.
- Use deterministic text serialization:
  - `kernel_0|vector_static_2d|4x8`
  - dynamic dim as `?`

Expected report:

```text
ScheduleCache:
  shape_bucket_lookups = 2
  shape_bucket_misses = 1
  tuning_lookups = 2
  tuning_misses = 1
  negative_cache_entries = 0
```

**Test:**

`ascend-schedule-cache.mlir` includes two identical vector kernels in one function and checks one hit through lookup/miss counts.

**Verification:**

Focused lit in xvm/docker.

## Task 9: StructuredLoweringDriver MVP

**Owner:** fresh worker subagent after Task 8 is reviewed.

**Purpose:** Establish a stable schedule-to-structured-lowering boundary without implementing Layer 4 memory realization.

**Files:**

- Add `StructuredLoweringDriver.h`
- Add `StructuredLoweringDriver.cpp`
- Modify `SchedulePass.cpp`
- Add `ascend-schedule-structured-lowering.mlir`

**Implementation details:**

Add a driver that receives `KernelPatternView`, `ScheduleProblem`, and selected `ScheduleDecisionSet`.

MVP behavior:

- Verify every scheduled pattern has at least one decision.
- Verify every op in the pattern has the same selected decision id.
- Attach `ascend.v2.schedule.structured_lowering = "loop_skeleton_v0"` to each scheduled op.
- Emit report:

```text
StructuredLowering:
  kernel = kernel_0
  skeleton = loop_skeleton_v0
  verified_ops = 2
```

This marker is the contract for later explicit loop/materialization work. It does not rewrite linalg ops in Phase 2.

**Test:**

`ascend-schedule-structured-lowering.mlir` checks the marker and report.

**Verification:**

Focused lit in xvm/docker.

## Task 10: Full Phase 2 Verification, Review, And Tracking

**Owner:** main agent coordinates with subagents.

**Purpose:** Close Phase 2 with quality evidence and tracking board updates.

**Files:**

- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

**Required validation commands in xvm/docker:**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
  /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-schedule-*.mlir'
```

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
  /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-v2-pipeline-mvp.mlir'
```

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
  cmake --build /home/niu/code/llvm-project/llvm/build --target check-afir'
```

**Review requirements:**

- Spec review against `docs/Ascend-MLIR-Detailed-Implementation-V2-4.zh.md`.
- Code review focused on:
  - deterministic ordering
  - no upstream MLIR edits
  - no global mutable cache that changes lit determinism
  - no regression in existing schedule attr values
  - no accidental edits to `AGENTS.md`
- Test review:
  - count newly added lit files
  - count focused schedule tests
  - record `check-afir` result

**Tracking update content:**

Update Phase 2 board with:

- completed task list
- changed file summary
- added test files count
- focused lit result
- `check-afir` result
- docker/xvm command evidence
- residual risks and explicit Phase 3 handoff

## Execution Protocol

For each development task:

1. Main agent dispatches exactly one fresh worker subagent with the task section and file ownership.
2. Worker edits only the owned files and reports changed paths.
3. Main agent reviews the diff.
4. Main agent runs or requests xvm/docker focused validation.
5. Main agent fixes integration issues locally if small, otherwise dispatches a focused follow-up worker.
6. Main agent updates the tracking board after each accepted task.

The main agent must not claim a task is complete until xvm/docker validation for that task has passed or the exact blocker is recorded.

## Expected Final State

After Phase 2:

- Schedule pass operates on `KernelPatternView` groups rather than independent ops.
- Axis coalescing, problem building, template matching, search, guards, decision sets, cache modeling, and structured lowering marker each have focused lit coverage.
- Existing MVP schedule attr values remain stable.
- `check-afir` passes with the same unsupported-test profile or better.
- `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md` records development, testing, docker verification, code review, and residual risks.
