# Ascend Symbol Axis Design Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the four remaining design-alignment blocks for symbol-axis scheduling: candidate-level tileable/reduction axes, symbol-derived shape constraints, symbolic template/search parameters with guard-budget/TopK behavior, and full axis propagation coverage for layout/reshape/split/concat/branch/merge/softmax multi-reduction paths.

**Architecture:** Keep `lib/Conversion/Ascend/Kernelize/Analysis/SymbolAxisSpace.{h,cpp}` as the only parser/translator for `ascend.symbol_constraints`. Add focused Schedule-side contract/projection helpers that consume `SymbolAxisSpace`, `KernelPatternView`, and `CoalescedAxisInfo` without reparsing symbol attributes locally. Preserve existing Schedule reports and runtime metadata, extending them with stable symbol names and axis IDs only where the design requires observable behavior.

**Tech Stack:** MLIR C++17 conversion passes, `afir-opt`, LLVM lit/FileCheck, existing Ascend gtest binaries, xvm verification under `/home/niu/code/Ascend-MLIR`.

---

## File Structure

- Create `lib/Conversion/Ascend/Schedule/ScheduleAxisContract.h`
  - Owns candidate-level `tileableAxes`, `requiredReductionAxes`, and propagation diagnostics derived from `CoalescedAxisInfo` plus symbol axis identity.
- Create `lib/Conversion/Ascend/Schedule/ScheduleAxisContract.cpp`
  - Implements symbol-axis set construction, role-aware propagation filtering, and report formatting helpers.
- Modify `lib/Conversion/Ascend/Schedule/ScheduleTypes.h`
  - Adds `AxisSet`, `ScheduleAxisContract`, `SymbolicTileParamSpec`, and symbol names on `ScheduleTileParam` if not already carried through.
- Modify `lib/Conversion/Ascend/Schedule/ScheduleProblemBuilder.cpp`
  - Builds and prints `tileable_axes`, `required_reduction_axes`, and symbol-derived `dim_equal(...)` constraints.
- Modify `lib/Conversion/Ascend/Schedule/ScheduleTemplateImplementation.cpp`
  - Uses `ScheduleAxisContract` to enumerate only legal tile axes and to build symbolic tile candidates keyed by axis symbol names.
- Modify `lib/Conversion/Ascend/Schedule/ScheduleSearch.cpp`
  - Emits symbol-aware candidate/decision guards, applies guard-budget pruning to symbolic guard fragments, and keeps deterministic TopK ordering.
- Modify `lib/Conversion/Ascend/Schedule/ScheduleDecision.cpp`
  - Carries selected symbolic axis parameter names into tile params and tail plans.
- Modify `lib/Conversion/Ascend/Schedule/AxisCoalescer.cpp`
  - Extends propagation coverage for layout transform, reshape, split/concat, branch/merge, and softmax multi-reduction paths while preserving the existing raw-axis fallback.
- Modify `lib/Conversion/Ascend/CMakeLists.txt`
  - Adds the new Schedule helper source.
- Modify `test/tools/check_ascend_conversion_architecture.sh`
  - Guards against new symbol parser duplication outside `Kernelize/Analysis/SymbolAxisSpace`.
- Add or modify lit tests:
  - `test/Conversion/ascend-schedule-symbol-axis-contract.mlir`
  - `test/Conversion/ascend-schedule-symbol-shape-constraints.mlir`
  - `test/Conversion/ascend-schedule-symbolic-search-space.mlir`
  - `test/Conversion/ascend-schedule-axis-propagation-complex.mlir`
  - Existing focused tests: `ascend-schedule-axis-coalescing.mlir`, `ascend-schedule-problem.mlir`, `ascend-schedule-search.mlir`, `ascend-schedule-symbolic-tile-params.mlir`

## Task 1: Candidate-Level Symbol Axis Contract

**Files:**
- Create: `lib/Conversion/Ascend/Schedule/ScheduleAxisContract.h`
- Create: `lib/Conversion/Ascend/Schedule/ScheduleAxisContract.cpp`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleTypes.h`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleProblemBuilder.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`
- Test: `test/Conversion/ascend-schedule-symbol-axis-contract.mlir`

- [ ] **Step 1: Write the failing lit test**

Add `test/Conversion/ascend-schedule-symbol-axis-contract.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @symbol_axis_contract_add_reduce(
    %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %out: tensor<?xf32>)
    -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %n = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %empty0 = tensor.empty(%m, %n) : tensor<?x?xf32>
  %add = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%empty0 : tensor<?x?xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %v = arith.addf %x, %y : f32
      linalg.yield %v : f32
    } -> tensor<?x?xf32>

  %red = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%add : tensor<?x?xf32>)
      outs(%out : tensor<?xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %v = arith.addf %acc, %x : f32
      linalg.yield %v : f32
    } -> tensor<?xf32>
  return %red : tensor<?xf32>
}

// CHECK: ScheduleProblem:
// CHECK:   kernel = kernel_0
// CHECK:   tileable_axes = [arg0_dim0]
// CHECK:   required_reduction_axes = [arg0_dim1]
// CHECK:   axis_constraints = [
// CHECK:     axis=0 roles=[bind_core,kernel_loop,vectorize] tail=masked_tail sym=arg0_dim0
// CHECK:     axis=1 roles=[full_reduction] tail=full_extent sym=arg0_dim1
// CHECK:   ]
```

- [ ] **Step 2: Run RED on xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-schedule-symbol-axis-contract.mlir'
```

Expected: `FAIL` because `ScheduleProblem` does not print `tileable_axes` or `required_reduction_axes`.

- [ ] **Step 3: Add the data model**

Add to `ScheduleTypes.h`:

```cpp
struct SymbolicAxisRef {
  unsigned logicalAxisId = 0;
  std::string symbolName;
  AxisKind kind = AxisKind::Unknown;
};

struct ScheduleAxisContract {
  SmallVector<SymbolicAxisRef, 4> tileableAxes;
  SmallVector<SymbolicAxisRef, 2> requiredReductionAxes;
  SmallVector<std::string, 2> propagationConstraints;
};
```

Add `ScheduleAxisContract axisContract;` to `ScheduleProblem`.

- [ ] **Step 4: Implement `ScheduleAxisContract` construction**

Create `ScheduleAxisContract.h` with:

```cpp
FailureOr<ScheduleAxisContract>
buildScheduleAxisContract(const KernelPatternView &pattern,
                          const CoalescedAxisInfo &axes);

void printScheduleAxisList(ArrayRef<SymbolicAxisRef> axes,
                           llvm::raw_ostream &os);
```

Create `ScheduleAxisContract.cpp` with these initial rules:

- Seed from `axes.logicalAxes`.
- Parallel axes enter `tileableAxes`.
- Reduction axes enter `requiredReductionAxes`.
- If the same symbol appears as reduction anywhere in the pattern, remove it from `tileableAxes` and keep it in `requiredReductionAxes`.
- Use `symbolName` when present, otherwise use `axis<logicalAxisId>` for report stability.

- [ ] **Step 5: Wire the builder and report**

In `ScheduleProblemBuilder.cpp`, call:

```cpp
FailureOr<ScheduleAxisContract> axisContract =
    buildScheduleAxisContract(pattern, axes);
if (failed(axisContract))
  return failure();
problem.axisContract = std::move(*axisContract);
```

Print after `structure_constraints`:

```cpp
os << "  tileable_axes = ";
printScheduleAxisList(problem.axisContract.tileableAxes, os);
os << "\n";
os << "  required_reduction_axes = ";
printScheduleAxisList(problem.axisContract.requiredReductionAxes, os);
os << "\n";
```

- [ ] **Step 6: Run GREEN**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-schedule-symbol-axis-contract.mlir test/Conversion/ascend-schedule-problem.mlir test/Conversion/ascend-schedule-axis-coalescing.mlir'
```

Expected: all listed tests `PASS`.

- [ ] **Step 7: Commit**

```bash
git add lib/Conversion/Ascend/CMakeLists.txt lib/Conversion/Ascend/Schedule/ScheduleTypes.h lib/Conversion/Ascend/Schedule/ScheduleProblemBuilder.cpp lib/Conversion/Ascend/Schedule/ScheduleAxisContract.h lib/Conversion/Ascend/Schedule/ScheduleAxisContract.cpp test/Conversion/ascend-schedule-symbol-axis-contract.mlir
git commit -m "feat: derive schedule symbol axis contract"
```

## Task 2: Symbol Constraints to ScheduleProblem Shape Constraints

**Files:**
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleProblemBuilder.cpp`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleTypes.h`
- Test: `test/Conversion/ascend-schedule-symbol-shape-constraints.mlir`

- [ ] **Step 1: Write the failing lit test**

Add `test/Conversion/ascend-schedule-symbol-shape-constraints.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @symbol_shape_constraints(
    %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %out: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %add = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%out : tensor<?x?xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %v = arith.addf %x, %y : f32
      linalg.yield %v : f32
    } -> tensor<?x?xf32>
  return %add : tensor<?x?xf32>
}

// CHECK: ScheduleProblem:
// CHECK:   shape_constraints = [d0 dynamic, d1 dynamic, dim_equal(arg0_dim0), dim_equal(arg0_dim1)]
```

- [ ] **Step 2: Run RED on xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-schedule-symbol-shape-constraints.mlir'
```

Expected: `FAIL`; only `d0 dynamic, d1 dynamic` are printed.

- [ ] **Step 3: Reuse `SymbolAxisSpace` without reparsing locally**

In `ScheduleProblemBuilder.cpp`, include:

```cpp
#include "Conversion/Ascend/Kernelize/Analysis/SymbolAxisSpace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
```

Add helper:

```cpp
void appendSymbolShapeConstraints(Operation *primaryOp,
                                  SmallVectorImpl<std::string> &constraints);
```

The helper:
- Gets `func::FuncOp` from `primaryOp->getParentOfType<func::FuncOp>()`.
- Calls `kernelize::buildSymbolAxisSpace(func)`.
- For each axis with `memberCount > 1`, appends `dim_equal(<symbolName>)`.
- Keeps insertion order from `FunctionAxisSpace.axes`.

- [ ] **Step 4: Wire constraints after result shape constraints**

In `buildScheduleProblem`, keep:

```cpp
appendShapeConstraints(problem.resultShape, problem.shapeConstraints);
appendSymbolShapeConstraints(primaryOp, problem.shapeConstraints);
```

- [ ] **Step 5: Run GREEN**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-schedule-symbol-shape-constraints.mlir test/Conversion/ascend-schedule-problem.mlir test/Conversion/ascend-normalize-symbol-constraints.mlir'
```

Expected: all listed tests `PASS`.

- [ ] **Step 6: Commit**

```bash
git add lib/Conversion/Ascend/Schedule/ScheduleProblemBuilder.cpp test/Conversion/ascend-schedule-symbol-shape-constraints.mlir
git commit -m "feat: lift symbol constraints into schedule problem"
```

## Task 3: Symbolic ScheduleSearch and Template Parameters

**Files:**
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleTypes.h`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleTemplateImplementation.cpp`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleSearch.cpp`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleDecision.cpp`
- Test: `test/Conversion/ascend-schedule-symbolic-search-space.mlir`
- Test: `test/Conversion/ascend-schedule-symbolic-tile-params.mlir`

- [ ] **Step 1: Write the failing lit test**

Add `test/Conversion/ascend-schedule-symbolic-search-space.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @symbolic_dynamic_vector(%arg0: tensor<?x?xf16>,
                                   %arg1: tensor<?x?xf16>,
                                   %out: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %add = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%out : tensor<?x?xf16>) {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %v = arith.addf %x, %y : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>
  return %add : tensor<?x?xf16>
}

// CHECK: ScheduleSearch:
// CHECK:   generated =
// CHECK:   kept = 4
// CHECK:   compile_time_top_k = 4
// CHECK:   pruned_by_guard_budget =
// CHECK:   selected_candidate_guards =
// CHECK-SAME: T_arg0_dim0 <= arg0_dim0
// CHECK-SAME: T_arg0_dim1 <= arg0_dim1
// CHECK: ScheduleDecisionSet:
// CHECK:   tile_params = [name=T_arg0_dim0 axis=0 binding=runtime
// CHECK-SAME: [name=T_arg0_dim1 axis=1 binding=runtime
```

- [ ] **Step 2: Run RED on xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-schedule-symbolic-search-space.mlir'
```

Expected: `FAIL`; tile params use `TB_M/TB_N` and guards use `d0/d1` or `a0/a1`, not `T_<symbol>`.

- [ ] **Step 3: Extend tile param naming**

In `ScheduleDecision.cpp`, change `defaultTileParamName` to prefer symbol names:

```cpp
if (tileIndex < problem.axes.logicalAxes.size()) {
  StringRef symbolName = problem.axes.logicalAxes[tileIndex].symbolName;
  if (!symbolName.empty())
    return (llvm::Twine("T_") + symbolName).str();
}
```

Keep existing `t_K`, `TB_M`, `TB_N`, `Tb_M`, `Tb_N` fallback behavior for static/no-symbol cases.

- [ ] **Step 4: Emit symbol-aware candidate guards**

In `ScheduleSearch.cpp`, add `appendSymbolicTileGuards`:

```cpp
void appendSymbolicTileGuards(const ScheduleProblem &problem,
                              const TileShape &tileShape,
                              SmallVectorImpl<ScheduleGuard> &guards);
```

For every runtime tile axis with symbol name `S` and non-full tile:
- Add `T_S > 0`.
- Add `T_S <= S`.
- Add `T_S % semantic_align == 0` when `semanticAlignmentGranularity > 0`.

Do not add these guards for extent-bound full reduction axes.

- [ ] **Step 5: Make guard budget count symbolic fragments**

Keep `getGuardCount(instance)` as the only budget gate. Ensure `appendSymbolicTileGuards` pushes actual `ScheduleGuard` objects into `candidateGuards` before the budget check in `searchScheduleInstancesWithStats`.

- [ ] **Step 6: Keep deterministic TopK**

Update `isLowerRankedInstance` tie-breakers to compare:

```cpp
if (lhs.candidateGuards.size() != rhs.candidateGuards.size())
  return lhs.candidateGuards.size() < rhs.candidateGuards.size();
```

before family/name lexical tie-breaks. This preserves stable TopK while preferring smaller guard sets.

- [ ] **Step 7: Run GREEN**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-schedule-symbolic-search-space.mlir test/Conversion/ascend-schedule-search.mlir test/Conversion/ascend-schedule-symbolic-tile-params.mlir test/Conversion/ascend-schedule-guards.mlir'
```

Expected: all listed tests `PASS`.

- [ ] **Step 8: Commit**

```bash
git add lib/Conversion/Ascend/Schedule/ScheduleTypes.h lib/Conversion/Ascend/Schedule/ScheduleTemplateImplementation.cpp lib/Conversion/Ascend/Schedule/ScheduleSearch.cpp lib/Conversion/Ascend/Schedule/ScheduleDecision.cpp test/Conversion/ascend-schedule-symbolic-search-space.mlir test/Conversion/ascend-schedule-symbolic-tile-params.mlir
git commit -m "feat: drive schedule search from symbolic axes"
```

## Task 4: Complex Axis Propagation Coverage

**Files:**
- Modify: `lib/Conversion/Ascend/Schedule/AxisCoalescer.cpp`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleAxisContract.cpp`
- Test: `test/Conversion/ascend-schedule-axis-propagation-complex.mlir`
- Existing tests: `test/Conversion/ascend-full-pipeline-add-broadcast-concat.mlir`, `test/Conversion/ascend-full-pipeline-softmax.mlir`, `test/Conversion/ascend-kernelize-transpose.mlir`

- [ ] **Step 1: Write the failing lit test**

Add `test/Conversion/ascend-schedule-axis-propagation-complex.mlir` with four split-input sections:

```mlir
// RUN: afir-opt %s --split-input-file --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @transpose_preserves_symbol_axes(%arg0: tensor<?x?xf16>,
                                           %out: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %t = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf16>)
      outs(%out : tensor<?x?xf16>) {
    ^bb0(%x: f16, %o: f16):
      linalg.yield %x : f16
    } -> tensor<?x?xf16>
  return %t : tensor<?x?xf16>
}

// CHECK-LABEL: func.func @transpose_preserves_symbol_axes
// CHECK: tileable_axes = [arg0_dim1, arg0_dim0]

// -----

func.func @concat_dim0_barrier(%a: tensor<?x8xf16>, %b: tensor<?x8xf16>)
    -> tensor<?x8xf16> {
  %c = tensor.concat dim(0) %a, %b : (tensor<?x8xf16>, tensor<?x8xf16>) -> tensor<?x8xf16>
  return %c : tensor<?x8xf16>
}

// CHECK-LABEL: func.func @concat_dim0_barrier
// CHECK: structure_constraints =
// CHECK-SAME: concat_axis_barrier

// -----

func.func @reshape_static_bridge(%arg0: tensor<?x8xf16>) -> tensor<?xf16> {
  %flat = tensor.collapse_shape %arg0 [[0, 1]] : tensor<?x8xf16> into tensor<?xf16>
  return %flat : tensor<?xf16>
}

// CHECK-LABEL: func.func @reshape_static_bridge
// CHECK: structure_constraints =
// CHECK-SAME: reshape_static_bridge

// -----

func.func @softmax_two_reductions_share_tile_axis(%arg0: tensor<?x?xf32>,
                                                  %out: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %n = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %empty0 = tensor.empty(%m) : tensor<?xf32>
  %max = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<?x?xf32>)
      outs(%empty0 : tensor<?xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %v = arith.maximumf %x, %acc : f32
      linalg.yield %v : f32
    } -> tensor<?xf32>
  %empty1 = tensor.empty(%m, %n) : tensor<?x?xf32>
  %sub = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %max : tensor<?x?xf32>, tensor<?xf32>)
      outs(%empty1 : tensor<?x?xf32>) {
    ^bb0(%x: f32, %m: f32, %o: f32):
      %v = arith.subf %x, %m : f32
      linalg.yield %v : f32
    } -> tensor<?x?xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%sub : tensor<?x?xf32>)
      outs(%empty0 : tensor<?xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %v = arith.addf %x, %acc : f32
      linalg.yield %v : f32
    } -> tensor<?xf32>
  %div = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%sub, %sum : tensor<?x?xf32>, tensor<?xf32>)
      outs(%out : tensor<?x?xf32>) {
    ^bb0(%x: f32, %s: f32, %o: f32):
      %v = arith.divf %x, %s : f32
      linalg.yield %v : f32
    } -> tensor<?x?xf32>
  return %div : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @softmax_two_reductions_share_tile_axis
// CHECK: tileable_axes = [arg0_dim0]
// CHECK: required_reduction_axes = [arg0_dim1]
```

- [ ] **Step 2: Run RED on xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-schedule-axis-propagation-complex.mlir'
```

Expected: `FAIL` on at least concat/reshape/softmax checks, because current propagation does not emit those contract constraints.

- [ ] **Step 3: Add layout transform propagation**

In `AxisCoalescer.cpp`, keep the current linalg indexing map path, but for projection/permutation maps:
- Accept affine dim permutation maps.
- Use `SymbolAxisSpace::opAxisMap` first.
- Fall back to raw-axis map only when the symbol map has no entry.

- [ ] **Step 4: Add view-like static bridge propagation**

In `ScheduleAxisContract.cpp`, when pattern ops include `tensor.collapse_shape` or `tensor.expand_shape`:
- If reassociation is static and all folded dimensions except one are static constants, append `reshape_static_bridge`.
- If multiple dynamic symbols collapse into one axis, remove the collapsed axis from `tileableAxes` and append `reshape_dynamic_barrier`.

- [ ] **Step 5: Add split/concat propagation**

When pattern ops include `tensor.concat`:
- If concat dimension is static and not tileable, append `concat_axis_barrier`.
- If concat dimension maps to a tileable symbol, remove that symbol from `tileableAxes`.
- Preserve other symbols as tileable.

- [ ] **Step 6: Add branch/merge propagation**

When ops have `ascend.branch_group` or `ascend.merge_group`:
- Group ops by group id.
- Require same ordered symbol list for all branches in a group.
- On mismatch, remove mismatched symbols from `tileableAxes` and append `branch_merge_axis_mismatch`.

- [ ] **Step 7: Add softmax multi-reduction consistency**

For patterns with two or more reduction ops:
- Collect each reduction op's reduction symbols.
- Require all reduction ops to agree on the reduction symbol set.
- Add agreed symbols to `requiredReductionAxes`.
- Remove agreed symbols from `tileableAxes`.
- If reduction symbols differ, emit `softmax_reduction_axis_mismatch` and make the candidate unschedulable by returning failure from the contract builder.

- [ ] **Step 8: Run GREEN**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-schedule-axis-propagation-complex.mlir test/Conversion/ascend-schedule-axis-coalescing.mlir test/Conversion/ascend-full-pipeline-add-broadcast-concat.mlir test/Conversion/ascend-full-pipeline-softmax.mlir test/Conversion/ascend-kernelize-transpose.mlir'
```

Expected: all listed tests `PASS`.

- [ ] **Step 9: Commit**

```bash
git add lib/Conversion/Ascend/Schedule/AxisCoalescer.cpp lib/Conversion/Ascend/Schedule/ScheduleAxisContract.cpp test/Conversion/ascend-schedule-axis-propagation-complex.mlir
git commit -m "feat: cover complex symbol axis propagation"
```

## Task 5: Architecture Guard and Full Verification

**Files:**
- Modify: `test/tools/check_ascend_conversion_architecture.sh`
- Verify: conversion lit, schedule gtests, runtime smoke gates as needed.

- [ ] **Step 1: Strengthen architecture guard**

Add checks:

```bash
require_file "lib/Conversion/Ascend/Schedule/ScheduleAxisContract.h"
require_pattern "lib/Conversion/Ascend/Schedule/ScheduleProblemBuilder.cpp" "buildScheduleAxisContract"
require_pattern "lib/Conversion/Ascend/Schedule/ScheduleProblemBuilder.cpp" "appendSymbolShapeConstraints"
reject_pattern "lib/Conversion/Ascend/Schedule" "parseSymbolConstraintAttr"
```

The only allowed `parseSymbolConstraintAttr` owner remains `lib/Conversion/Ascend/Kernelize/Analysis/SymbolAxisSpace.cpp` and verifier code under `lib/Conversion/Ascend/Common` or `lib/Conversion/Ascend/Normalize`.

- [ ] **Step 2: Run architecture test**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && bash test/tools/check_ascend_conversion_architecture.sh .'
```

Expected: exit code `0`.

- [ ] **Step 3: Run focused xvm build**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt AscendSymbolConstraintsTest AscendKernelizeOpInterfaceTest AscendKernelPatternTest AscendScheduleDecisionTest AscendScheduleTemplateRegistryTest'
```

Expected: exit code `0`.

- [ ] **Step 4: Run focused conversion lit**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-kernelize-*.mlir test/Conversion/ascend-schedule-*.mlir test/Conversion/ascend-normalize-symbol-constraints.mlir test/Conversion/ascend-full-pipeline-gather-elementwise-fusion.mlir test/Conversion/ascend-full-pipeline-relu-broadcast-transpose.mlir test/Conversion/ascend-full-pipeline-broadcast-add-reduce.mlir test/Conversion/ascend-full-pipeline-rank2-elementwise-add.mlir test/Conversion/ascend-full-pipeline-add-broadcast-concat.mlir test/Conversion/ascend-full-pipeline-softmax.mlir'
```

Expected: all discovered tests `PASS`.

- [ ] **Step 5: Run gtests**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./build/bin/AscendSymbolConstraintsTest && ./build/bin/AscendKernelizeOpInterfaceTest && ./build/bin/AscendKernelPatternTest && ./build/bin/AscendScheduleDecisionTest && ./build/bin/AscendScheduleTemplateRegistryTest'
```

Expected: all test binaries exit `0`.

- [ ] **Step 6: Run simulation gate before any NPU claim**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && bash test/tools/runtime/run_simbackend_examples.sh'
```

Expected: exit code `0`, except an explicitly documented longrun-only case may be recorded as skipped/legacy if it repeats the known `add-broadcast-concat` simulator hang rather than a validation failure.

- [ ] **Step 7: Commit verification guard**

```bash
git add test/tools/check_ascend_conversion_architecture.sh
git commit -m "test: guard symbol axis schedule architecture"
```

## Completion Audit

- The goal is not complete until all four design blocks are implemented and verified in current state.
- Evidence required:
  - `ScheduleProblem` report contains `tileable_axes`, `required_reduction_axes`, and symbol-derived `dim_equal(...)` constraints for dynamic symbol tests.
  - `ScheduleSearch` report and `ScheduleDecisionSet` expose `T_<symbol>` tile params and symbol-aware guards.
  - Complex propagation lit covers transpose, reshape, concat, branch/merge, and multi-reduction softmax behavior.
  - Architecture guard proves Schedule does not locally parse `ascend.symbol_constraints`.
  - xvm focused build, focused conversion lit, and gtests pass.
  - xvm sim gate is run before any real-NPU validation claim.
