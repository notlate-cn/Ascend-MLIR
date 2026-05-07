# Ascend MLIR V2 MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first executable V2 compiler spine: target profile loading, entry normalization, MVP kernelization, fixed schedule lowering, and debug/verifier hooks.

**Architecture:** Keep the existing `afir-opt` driver and conversion pass registry while adding V2 passes under the current infrastructure. The MVP produces a runnable vertical slice for Linalg/Tensor input through normalized IR, kernel boundary marking, fixed schedule loop structure, and existing downstream bufferize/AscendC prototype passes.

**Tech Stack:** C++17, MLIR passes/TableGen, LLVM support libraries, lit/FileCheck, existing `afir-opt` and `check-afir` targets.

---

## Scope

This plan covers the first MVP only:

- V2-8 target hardware modeling MVP for `Ascend910B2`
- V2-2 Normalize pass and verifier
- V2-3 Kernelize MVP for single-primary-op candidates
- V2-4 Schedule MVP with fixed rule scheduling
- V2-7 debug/report hooks for those layers
- V2-9 pipeline registration through the existing `afir-opt` driver

This plan intentionally does not implement full candidate merge, horizontal fusion, multi-primary composite candidates, complete schedule search, full `MemoryRealizationPlan`, host tiling shared library generation, or runtime manifest DAG execution. Those become follow-up plans after this MVP is green.

## File Structure

- Create: `include/Target/Ascend/TargetProfile.h`
  - target data structs and query APIs
- Create: `include/Target/Ascend/CannTargetProfileLoader.h`
  - CANN `.ini` profile loader interface
- Create: `lib/Target/Ascend/CMakeLists.txt`
  - target model library build rules
- Create: `lib/Target/Ascend/CannTargetProfileLoader.cpp`
  - minimal `.ini` parser and `Ascend910B2` profile construction
- Create: `lib/Target/Ascend/TargetProfile.cpp`
  - target verifier and query implementation
- Create: `include/Conversion/AscendV2/Normalize/NormalizePass.h`
  - `--ascend-normalize` constructor
- Create: `lib/Conversion/AscendV2/Normalize/NormalizePass.cpp`
  - entry dialect whitelist and normalize verifier
- Create: `include/Conversion/AscendV2/Kernelize/KernelizePass.h`
  - `--ascend-kernelize` constructor
- Create: `lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp`
  - dependency walk, role classification, MVP kernel pattern marking
- Create: `include/Conversion/AscendV2/Schedule/SchedulePass.h`
  - `--ascend-schedule` constructor
- Create: `lib/Conversion/AscendV2/Schedule/SchedulePass.cpp`
  - fixed schedule loop skeleton and schedule attributes
- Create: `include/Conversion/AscendV2/Debug/DebugOptions.h`
  - shared debug option parsing and report helpers
- Create: `lib/Conversion/AscendV2/Debug/DebugOptions.cpp`
  - debug stage/report implementation
- Modify: `include/Conversion/Passes.td`
  - register V2 pass names and options
- Modify: `include/Conversion/Passes.h`
  - include V2 pass headers
- Modify: `lib/Conversion/CMakeLists.txt`
  - add `AscendV2` subdirectories
- Modify: `lib/Target/CMakeLists.txt`
  - add `Ascend` target model subdirectory
- Modify: `tools/afir-opt/afir-opt.cpp`
  - no new driver yet; ensure V2 passes are registered through existing conversion registration
- Create: `test/Target/ascend-target-profile.mlir`
  - target loading diagnostics smoke test
- Create: `test/Conversion/ascend-normalize.mlir`
  - Normalize success and rejection cases
- Create: `test/Conversion/ascend-kernelize-mvp.mlir`
  - single-primary kernel marking tests
- Create: `test/Conversion/ascend-schedule-mvp.mlir`
  - fixed schedule lowering tests
- Create: `test/Conversion/ascend-v2-pipeline-mvp.mlir`
  - vertical MVP pipeline smoke test

## Task 1: Add Target Profile MVP

**Files:**
- Create: `include/Target/Ascend/TargetProfile.h`
- Create: `include/Target/Ascend/CannTargetProfileLoader.h`
- Create: `lib/Target/Ascend/CMakeLists.txt`
- Create: `lib/Target/Ascend/TargetProfile.cpp`
- Create: `lib/Target/Ascend/CannTargetProfileLoader.cpp`
- Modify: `lib/Target/CMakeLists.txt`
- Test: `test/Target/ascend-target-profile.mlir`

- [ ] **Step 1: Write the failing lit test**

```mlir
// RUN: afir-opt --ascend-print-target-profile='soc=Ascend910B2 cann-root=/home/niu/Ascend/20260323_newest/cann' %s 2>&1 | FileCheck %s

module {}

// CHECK: TargetProfile
// CHECK: soc = "Ascend910B2"
// CHECK: memory_place = "GM"
// CHECK: memory_place = "L1"
// CHECK: memory_place = "UB"
// CHECK: intrinsic = "Intrinsic_data_move_out2l1"
```

- [ ] **Step 2: Run the test and confirm it fails because the pass is unknown**

Run:

```bash
cmake --build build --target afir-opt -j2
build/bin/afir-opt --ascend-print-target-profile='soc=Ascend910B2 cann-root=/home/niu/Ascend/20260323_newest/cann' test/Target/ascend-target-profile.mlir
```

Expected: failure mentioning unknown pass or option.

- [ ] **Step 3: Add target model data types**

Implement `TargetProfile.h` with these MVP types:

```cpp
namespace mlir::ascend {
enum class MemoryPlace { GM, L2, L1, L0A, L0B, L0C, UB };

struct TargetIdentity {
  std::string socVersion;
  std::string shortSocVersion;
  std::string npuArch;
};

struct TargetHardwareInfo {
  int64_t aiCoreCount = 0;
  int64_t cubeCoreCount = 0;
  int64_t vectorCoreCount = 0;
  int64_t l1SizeBytes = 0;
  int64_t ubSizeBytes = 0;
  bool supportBF16 = false;
  bool supportFixpipe = false;
};

struct TargetIntrinsicInfo {
  std::string name;
  SmallVector<std::string> dtypes;
};

struct TargetProfile {
  TargetIdentity identity;
  TargetHardwareInfo hardware;
  DenseMap<MemoryPlace, int64_t> capacityBytes;
  SmallVector<std::pair<MemoryPlace, MemoryPlace>> movementPaths;
  SmallVector<TargetIntrinsicInfo> intrinsics;
};

LogicalResult verifyTargetProfile(const TargetProfile &profile,
                                  raw_ostream &os);
StringRef stringifyMemoryPlace(MemoryPlace place);
} // namespace mlir::ascend
```

- [ ] **Step 4: Implement the CANN loader**

Implement `CannTargetProfileLoader::load(cannRoot, socVersion)` to:

- open `${cannRoot}/aarch64-linux/data/platform_config/${socVersion}.ini`
- parse section/key/value lines
- read `[version]`, `[SoCInfo]`, `[AICoreSpec]`, `[AICoreMemoryRates]`, `[AICoreintrinsicDtypeMap]`
- fill the MVP `TargetProfile`
- verify nonzero `ai_core_cnt`, `l1_size`, `ub_size`
- include required paths `GM->L1`, `L1->L0A`, `L1->L0B`, `L0C->UB`, `UB->GM`

- [ ] **Step 5: Add a diagnostic print pass**

Register `--ascend-print-target-profile` in `Passes.td` as a module pass with options:

```tablegen
Option<"soc", "soc", "std::string", /*default=*/"\"Ascend910B2\"",
       "CANN SoC version">
Option<"cannRoot", "cann-root", "std::string", /*default=*/"\"\"",
       "CANN root path">
```

The pass loads the profile and prints a stable text summary to stderr.

- [ ] **Step 6: Run target tests**

Run:

```bash
cmake --build build --target afir-opt -j2
llvm-lit -v build/test/Target/ascend-target-profile.mlir
```

Expected: PASS on the xvm/CANN path above. If local CANN is unavailable, mark the test with `// REQUIRES: ascend_env` and set the root from `ASCEND_TOOLKIT_HOME`.

- [ ] **Step 7: Commit**

```bash
git add include/Target/Ascend lib/Target/Ascend lib/Target/CMakeLists.txt \
        include/Conversion/Passes.td test/Target/ascend-target-profile.mlir
git commit -m "target: add Ascend target profile MVP"
```

## Task 2: Add V2 Pass Skeleton And Debug Options

**Files:**
- Create: `include/Conversion/AscendV2/Debug/DebugOptions.h`
- Create: `lib/Conversion/AscendV2/Debug/DebugOptions.cpp`
- Create: `lib/Conversion/AscendV2/CMakeLists.txt`
- Modify: `include/Conversion/Passes.td`
- Modify: `include/Conversion/Passes.h`
- Modify: `lib/Conversion/CMakeLists.txt`
- Test: `test/Conversion/ascend-v2-pipeline-mvp.mlir`

- [ ] **Step 1: Write the failing pass registration test**

```mlir
// RUN: afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule %s | FileCheck %s

module {
  func.func @empty() {
    return
  }
}

// CHECK: func.func @empty
```

- [ ] **Step 2: Run and confirm unknown passes**

Run:

```bash
cmake --build build --target afir-opt -j2
build/bin/afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule test/Conversion/ascend-v2-pipeline-mvp.mlir
```

Expected: failure mentioning unknown V2 passes.

- [ ] **Step 3: Add shared debug options**

Implement:

```cpp
namespace mlir::ascend::v2 {
enum class DebugStage { None, Normalize, Kernelize, Schedule, All };

struct DebugOptions {
  DebugStage stage = DebugStage::None;
  bool dumpReport = false;
};

DebugStage parseDebugStage(StringRef value);
bool shouldDump(DebugOptions options, DebugStage stage);
void emitStageHeader(raw_ostream &os, DebugStage stage, StringRef passName);
} // namespace mlir::ascend::v2
```

- [ ] **Step 4: Register empty V2 passes**

Add `AscendNormalizePass`, `AscendKernelizePass`, and `AscendSchedulePass` in `Passes.td`. Each pass accepts:

```tablegen
Option<"debugStage", "debug-stage", "std::string", /*default=*/"\"none\"",
       "V2 debug stage: none, normalize, kernelize, schedule, all">
Option<"dumpReport", "dump-report", "bool", /*default=*/"false",
       "Dump V2 analysis report to stderr">
```

- [ ] **Step 5: Run the pipeline test**

Run:

```bash
cmake --build build --target afir-opt -j2
llvm-lit -v build/test/Conversion/ascend-v2-pipeline-mvp.mlir
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add include/Conversion/AscendV2 lib/Conversion/AscendV2 \
        include/Conversion/Passes.td include/Conversion/Passes.h \
        lib/Conversion/CMakeLists.txt test/Conversion/ascend-v2-pipeline-mvp.mlir
git commit -m "conversion: add Ascend V2 pass skeleton"
```

## Task 3: Implement Normalize MVP

**Files:**
- Create: `include/Conversion/AscendV2/Normalize/NormalizePass.h`
- Create: `lib/Conversion/AscendV2/Normalize/NormalizePass.cpp`
- Modify: `lib/Conversion/AscendV2/CMakeLists.txt`
- Test: `test/Conversion/ascend-normalize.mlir`

- [ ] **Step 1: Write Normalize tests**

```mlir
// RUN: afir-opt --ascend-normalize %s | FileCheck %s
// RUN: not afir-opt --ascend-normalize %s --split-input-file 2>&1 | FileCheck %s --check-prefix=ERR

func.func @valid(%arg0: tensor<4x8xf16>, %arg1: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %empty = tensor.empty() : tensor<4x8xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>)
    outs(%empty : tensor<4x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<4x8xf16>
  return %out : tensor<4x8xf16>
}

// CHECK-LABEL: func.func @valid
// CHECK: ascend.v2.normalized

// -----

"test.unknown"() : () -> ()

// ERR: unsupported dialect before Kernelize
```

- [ ] **Step 2: Run and confirm failure**

Run:

```bash
cmake --build build --target afir-opt -j2
llvm-lit -v build/test/Conversion/ascend-normalize.mlir
```

Expected: FAIL before implementation.

- [ ] **Step 3: Implement whitelist verification**

In `NormalizePass.cpp`, walk the module and reject operations whose dialect namespace is not one of:

```cpp
llvm::StringSet<> allowed = {"builtin", "func", "tensor", "linalg",
                             "arith", "math", "affine"};
```

Allow `affine` only for affine maps attached to Linalg syntax and avoid introducing affine ops.

- [ ] **Step 4: Mark normalized functions**

Set `ascend.v2.normalized = true` on each `func.func` that passes validation.

- [ ] **Step 5: Run Normalize tests**

Run:

```bash
cmake --build build --target afir-opt -j2
llvm-lit -v build/test/Conversion/ascend-normalize.mlir
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add include/Conversion/AscendV2/Normalize \
        lib/Conversion/AscendV2/Normalize \
        lib/Conversion/AscendV2/CMakeLists.txt \
        test/Conversion/ascend-normalize.mlir
git commit -m "conversion: implement Ascend V2 normalize MVP"
```

## Task 4: Implement Kernelize MVP

**Files:**
- Create: `include/Conversion/AscendV2/Kernelize/KernelizePass.h`
- Create: `lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp`
- Modify: `lib/Conversion/AscendV2/CMakeLists.txt`
- Test: `test/Conversion/ascend-kernelize-mvp.mlir`

- [ ] **Step 1: Write Kernelize tests**

```mlir
// RUN: afir-opt --ascend-normalize --ascend-kernelize='dump-report=true debug-stage=kernelize' %s 2>&1 | FileCheck %s

func.func @elementwise(%arg0: tensor<4x8xf16>, %arg1: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %empty = tensor.empty() : tensor<4x8xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>)
    outs(%empty : tensor<4x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<4x8xf16>
  return %out : tensor<4x8xf16>
}

// CHECK: Kernelize report
// CHECK: op_role = "vector"
// CHECK: kernel_pattern = "kernel_0"
// CHECK: primary_ops = 1
// CHECK: ascend.v2.kernel
```

- [ ] **Step 2: Run and confirm failure**

Run:

```bash
cmake --build build --target afir-opt -j2
llvm-lit -v build/test/Conversion/ascend-kernelize-mvp.mlir
```

Expected: FAIL before role/kernel attributes exist.

- [ ] **Step 3: Implement MVP role classification**

Classify:

- `linalg.matmul` and `linalg.batch_matmul` as `cube`
- `linalg.generic` with only parallel iterators as `vector`
- `linalg.generic` with reduction iterators as `reduction`
- unsupported structured ops as `unsupported`

Attach:

```text
ascend.v2.op_role = "vector|reduction|cube|unsupported"
```

- [ ] **Step 4: Implement single-primary kernel marking**

For each supported `linalg` op, assign:

```text
ascend.v2.kernel = "kernel_N"
ascend.v2.primary = true
```

Do not merge kernels in this MVP. Preserve deterministic operation order by walking functions in IR order.

- [ ] **Step 5: Add verifier checks**

Reject `--ascend-kernelize` if a function lacks `ascend.v2.normalized`. Reject supported `linalg` ops that fail role classification.

- [ ] **Step 6: Run Kernelize tests**

Run:

```bash
cmake --build build --target afir-opt -j2
llvm-lit -v build/test/Conversion/ascend-kernelize-mvp.mlir
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add include/Conversion/AscendV2/Kernelize \
        lib/Conversion/AscendV2/Kernelize \
        lib/Conversion/AscendV2/CMakeLists.txt \
        test/Conversion/ascend-kernelize-mvp.mlir
git commit -m "conversion: implement Ascend V2 kernelize MVP"
```

## Task 5: Implement Fixed Schedule MVP

**Files:**
- Create: `include/Conversion/AscendV2/Schedule/SchedulePass.h`
- Create: `lib/Conversion/AscendV2/Schedule/SchedulePass.cpp`
- Modify: `lib/Conversion/AscendV2/CMakeLists.txt`
- Test: `test/Conversion/ascend-schedule-mvp.mlir`

- [ ] **Step 1: Write Schedule tests**

```mlir
// RUN: afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='dump-report=true debug-stage=schedule' %s 2>&1 | FileCheck %s

func.func @elementwise(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> {
  %empty = tensor.empty() : tensor<64xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty : tensor<64xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>
  return %out : tensor<64xf16>
}

// CHECK: Schedule report
// CHECK: schedule_family = "vector_static_1d"
// CHECK: schedule_template = "single_tile_per_block"
// CHECK: ascend.v2.schedule.family = "vector_static_1d"
```

- [ ] **Step 2: Run and confirm failure**

Run:

```bash
cmake --build build --target afir-opt -j2
llvm-lit -v build/test/Conversion/ascend-schedule-mvp.mlir
```

Expected: FAIL before schedule attributes exist.

- [ ] **Step 3: Implement schedule problem extraction**

For each op with `ascend.v2.kernel`, extract:

- result rank
- iterator types
- static shape if available
- `ascend.v2.op_role`

Emit report lines for `ScheduleProblem`.

- [ ] **Step 4: Implement fixed schedule decision**

For MVP:

- vector rank-1 or rank-2 gets `schedule_family = "vector_static_1d"` or `"vector_static_2d"`
- reduction gets `schedule_family = "reduction_static"`
- cube gets `schedule_family = "cube_static_matmul"`
- all get `schedule_template = "single_tile_per_block"`

Attach:

```text
ascend.v2.schedule.family
ascend.v2.schedule.template
ascend.v2.schedule.decision_id
```

- [ ] **Step 5: Add schedule verifier**

Reject ops without `ascend.v2.kernel`. Reject unsupported `ascend.v2.op_role`.

- [ ] **Step 6: Run Schedule tests**

Run:

```bash
cmake --build build --target afir-opt -j2
llvm-lit -v build/test/Conversion/ascend-schedule-mvp.mlir
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add include/Conversion/AscendV2/Schedule \
        lib/Conversion/AscendV2/Schedule \
        lib/Conversion/AscendV2/CMakeLists.txt \
        test/Conversion/ascend-schedule-mvp.mlir
git commit -m "conversion: implement Ascend V2 schedule MVP"
```

## Task 6: Add Vertical MVP Pipeline Test

**Files:**
- Modify: `test/Conversion/ascend-v2-pipeline-mvp.mlir`
- Optionally modify: `docs/Ascend-MLIR-Detailed-Implementation-V2-9.zh.md`

- [ ] **Step 1: Expand the pipeline smoke test**

Update the test to run:

```mlir
// RUN: afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule %s | FileCheck %s
```

Use one elementwise function and check:

```text
ascend.v2.normalized
ascend.v2.op_role
ascend.v2.kernel
ascend.v2.schedule.family
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
cmake --build build --target afir-opt -j2
llvm-lit -v build/test/Conversion/ascend-normalize.mlir \
            build/test/Conversion/ascend-kernelize-mvp.mlir \
            build/test/Conversion/ascend-schedule-mvp.mlir \
            build/test/Conversion/ascend-v2-pipeline-mvp.mlir
```

Expected: PASS.

- [ ] **Step 3: Run full regression**

Run:

```bash
cmake --build build --target check-afir -j2
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add test/Conversion/ascend-v2-pipeline-mvp.mlir \
        docs/Ascend-MLIR-Detailed-Implementation-V2-9.zh.md
git commit -m "test: add Ascend V2 MVP pipeline coverage"
```

Only add the doc file if the pass naming table was updated.

## Implementation Order

1. Task 2 first if the worker wants a pure pass skeleton before target loading.
2. Task 1 can run independently once pass registration conventions are clear.
3. Task 3 requires Task 2.
4. Task 4 requires Task 3.
5. Task 5 requires Task 4.
6. Task 6 requires Tasks 3-5.

The recommended execution order is Task 2, Task 1, Task 3, Task 4, Task 5, Task 6.

## Verification Commands

Use focused commands during development:

```bash
cmake --build build --target afir-opt -j2
llvm-lit -v build/test/Conversion/ascend-normalize.mlir
llvm-lit -v build/test/Conversion/ascend-kernelize-mvp.mlir
llvm-lit -v build/test/Conversion/ascend-schedule-mvp.mlir
```

Before declaring the branch complete:

```bash
cmake --build build --target check-afir -j2
```

If CANN target profile tests are enabled:

```bash
ASCEND_TOOLKIT_HOME=/home/niu/Ascend/20260323_newest/cann \
llvm-lit -v build/test/Target/ascend-target-profile.mlir
```

## Follow-Up Plans After MVP

- Kernelize V2 full candidate analysis: closure, primitive seed/expand, merge, horizontal fusion, and constrained local selection.
- Schedule V2 full search: `ScheduleProblem`, `ScheduleDecisionSet`, `compileTimeTopK`, `runtimeTopK`, cache and guard handling.
- Realize V2 plan objects: `PlacementPlan`, `StaticMemoryPlan`, `MovementPlan`, `MemoryRealizationPlan`.
- Translate/runtime artifacts: host tiling ABI, `tiling_space.json`, runtime manifest, multi-kernel DAG.

## Self-Review

- Spec coverage: this plan maps V2-8, V2-2, V2-3 MVP, V2-4 MVP, V2-7 debug hooks, and V2-9 pipeline registration to concrete tasks. V2-5, V2-6, and the full runtime portion of V2-9 are explicitly deferred.
- Placeholder scan: no task uses open-ended placeholder work; each task names files, tests, commands, and expected behavior.
- Type consistency: public names use the agreed V2 terms: `TargetProfile`, `TargetMemoryModel` subset through `TargetProfile`, `ascend-normalize`, `ascend-kernelize`, `ascend-schedule`, `op_role`, `kernel`, and `schedule`.
