# Symbolic Tile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the static selected-tile lowering contract with role-generic symbolic tile metadata and runtime tiling fields.

**Architecture:** Schedule still ranks concrete candidate defaults, but it emits `tile_params` as the authoritative contract. Runtime artifact emission exports those params to tiling metadata. Compute selected-tile lowering consumes tile fields when present and keeps `selected_tile_shape` only as a legacy fallback.

**Tech Stack:** MLIR C++ passes, CANN artifact manifest emission, lit/FileCheck tests, xvm verification.

---

### Task 1: Document And Attribute Contract

**Files:**
- Modify: `include/Conversion/Ascend/Common/Attributes.h`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleTypes.h`
- Modify: `docs/Ascend-MLIR-Detailed-Implementation-V2-4.zh.md`

- [x] Add shared attribute names `ascend.schedule.tile_params` and `ascend.schedule.tile_binding`.
- [x] Add schedule data structs for symbolic tile params, including logical axis, name, binding, default, upper bound, extent, roles, and primitive uses.
- [x] Update the V2 schedule decision section to state that `selected_tile_shape` is legacy/debug only.

### Task 2: RED Tests For Schedule Metadata

**Files:**
- Create: `test/Conversion/ascend-schedule-symbolic-tile-params.mlir`

- [x] Add vector/all-parallel, reduction, and cube lit kernels.
- [x] Cover memory role with a direct ScheduleDecision unit test, because the
      lit pipeline cannot force a copy-body generic into a memory role without
      the upstream kernelize contract selecting it as memory.
- [x] Run Schedule and check for `ascend.schedule.tile_binding = "symbolic"`.
- [x] Check role-derived `ascend.schedule.tile_params`.
- [x] Check `selected_tile_shape` is not the only tile metadata.

### Task 3: Emit Symbolic Tile Metadata

**Files:**
- Modify: `lib/Conversion/Ascend/Schedule/StructuredLoweringDriver.cpp`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleDecision.cpp`

- [x] Build tile param entries from `ScheduleProblem.axes.axisScheduleConstraints` and selected decision defaults.
- [x] Preserve `tile_params` in op-level, legacy function-level, and per-kernel schedule metadata.
- [x] Keep legacy `selected_tile_shape` emission for compatibility.
- [x] Extend the Schedule report with symbolic tile params.

### Task 4: RED Tests For Artifact Export

**Files:**
- Modify: `test/Target/cann-translate-runtime-artifacts.mlir`
- Modify: `test/Target/cann-translate-runtime-artifacts-multi.mlir`

- [x] Add `ascend.schedule.tile_params` to the test kernels.
- [x] Check manifest `tilingParams.tile_params` preserves name, binding, default, upper_bound, roles, and primitive uses.
- [x] Check generated host tiling code assigns default tile values to matching `TilingData` fields.

### Task 5: Export Runtime Tile Params

**Files:**
- Modify: `lib/Target/CannKernel/CannRuntimeArtifacts.cpp`

- [x] Validate `tile_params` schema.
- [x] Emit `tile_params` in artifact manifest schedule entries and kernel entries.
- [x] Generate host tiling field defaults from `tile_params`.
- [x] Keep old `selected_tile_shape` manifest output for legacy consumers.

### Task 6: RED Tests For Runtime Tile Consumption

**Files:**
- Create: `test/Conversion/ascend-compute-lower-symbolic-tile.mlir`

- [x] Add rank-2 all-parallel and rank-2 reduction tests with `tile_params`.
- [x] Check `scf.for` steps are derived from function i64 tile arguments.
- [x] Check the old static selected-tile path remains accepted.

### Task 7: Consume Runtime Tile Params In Compute Lowering

**Files:**
- Modify: `lib/Conversion/Ascend/Translate/KernelIR/ComputeSelectedTileLowering.cpp`

- [x] Add helpers to read tile params from op attrs.
- [x] Add function i64 tile arguments when symbolic params are present and no matching argument exists.
- [x] Use runtime tile arg values for selected all-parallel and reduction loop steps.
- [x] Keep transpose/static fallback on `selected_tile_shape` until its dedicated symbolic lowering is implemented.

### Task 8: Verification

**Commands:**
- `ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt AscendScheduleDecisionTest'`
- `ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Conversion/ascend-schedule-symbolic-tile-params.mlir test/Conversion/ascend-compute-lower-symbolic-tile.mlir'`
- `ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./build/bin/AscendScheduleDecisionTest --gtest_filter=AscendScheduleDecisionTest.*'`
- `ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/Target/cann-translate-runtime-artifacts.mlir test/Target/cann-translate-runtime-artifacts-multi.mlir'`

Expected result: all focused tests pass.
