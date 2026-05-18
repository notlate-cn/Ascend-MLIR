# Ascend Dynamic Workspace Expression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the dynamic-shape workspace ABI gap so Realize can stamp symbolic workspace expressions and CANN artifacts can consume them through manifest, tiling-space JSON, and host `_GetWorkspaceSize`.

**Architecture:** Keep static `cann.workspace_size_bytes` as the fixed-size fast path. Add a parallel `cann.workspace_size_expr` string attr for compiler-generated symbolic expressions over tiling shape fields such as `dim_arg0_0`; CANN artifact emission preserves the JSON expression and lowers field names to `shape_args[N]` only in generated host C++.

**Tech Stack:** MLIR C++ passes, Ascend Realize planner, CANN runtime artifact emitter, LIT/FileCheck, xvm verification via `ssh xvm@orb`.

---

## File Structure

- Modify `include/Conversion/Ascend/Common/Attributes.h`
  - Add `cann.workspace_size_expr` shared attr name.
- Modify `lib/Conversion/Ascend/Realize/RealizeTypes.h`
  - Carry optional dynamic byte expressions on buffer facts, workspace slots, and `StaticMemoryPlan`.
- Modify `lib/Conversion/Ascend/Realize/BufferizationDriver.cpp`
  - Build byte expressions for dynamic tensor values from `tensor.empty` sizes and `tensor.dim %argN, %cD`.
- Modify `lib/Conversion/Ascend/Realize/StaticMemoryPlanner.cpp`
  - Preserve symbolic workspace expressions when static byte counts are unknown.
- Modify `lib/Conversion/Ascend/Realize/RealizePass.cpp`
  - Stamp `cann.workspace_size_expr` for dynamic plans and preserve `cann.workspace_size_bytes` for fully static plans.
- Modify `lib/Conversion/Ascend/Realize/RealizeReport.cpp`
  - Print expression presence and value for focused RED/GREEN evidence.
- Modify `lib/Target/CannKernel/CannRuntimeArtifacts.cpp`
  - Read `cann.workspace_size_expr`, emit it to JSON, and translate known tiling shape fields into `shape_args[N]` in host C++.
- Add `test/Conversion/ascend-realize-dynamic-workspace-expr.mlir`
  - RED/GREEN Realize coverage for dynamic workspace expression stamping.
- Add `test/Target/cann-translate-runtime-artifacts-dynamic-workspace.mlir`
  - RED/GREEN CANN artifact coverage for dynamic expression consumption.
- Add `test/Target/cann-translate-runtime-artifacts-dynamic-workspace-invalid.mlir`
  - Negative coverage for unknown tiling shape fields in host expression lowering.
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`
  - Record the closure and xvm verification.

## Tasks

### Task 1: RED Tests

- [ ] Add `test/Conversion/ascend-realize-dynamic-workspace-expr.mlir` with a dynamic vector temporary `tensor<?x128xf16>` whose size is `dim_arg0_0 * 128 * 2`.
- [ ] Add `test/Target/cann-translate-runtime-artifacts-dynamic-workspace.mlir` with `cann.workspace_size_expr = "dim_arg0_0 * 128 * 2"`.
- [ ] Run focused LIT in xvm and confirm both fail because `cann.workspace_size_expr` is not implemented yet.

### Task 2: Realize Expression Propagation

- [ ] Add `kCannWorkspaceSizeExprAttr`.
- [ ] Extend buffer facts and static memory plan types with `byteSizeExpr` / `workspaceSizeExpr`.
- [ ] Infer dynamic tensor byte expressions from tensor result shapes.
- [ ] Aggregate per-kernel workspace expression terms in `stampWorkspaceSizeAttrs`.
- [ ] Run the Realize LIT and confirm GREEN.

### Task 3: CANN Artifact Consumption

- [ ] Extend `WorkspaceInfo` to preserve a dynamic expression string.
- [ ] Keep JSON `workspace_size_expr` / `workspaceSizeExpr` equal to the symbolic expression.
- [ ] Emit host `_GetWorkspaceSize` by replacing known tiling fields with `shape_args[N]`.
- [ ] Run the CANN artifact LIT and confirm GREEN.

### Task 4: Verification And Tracking

- [ ] Run focused xvm targets:

```bash
ninja -C build -j6 afir-opt afir-translate
/home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v \
  build/test/Conversion/ascend-realize-dynamic-workspace-expr.mlir \
  build/test/Target/cann-translate-runtime-artifacts-dynamic-workspace.mlir \
  build/test/Conversion/ascend-realize-workspace-layout.mlir \
  build/test/Target/cann-translate-runtime-artifacts.mlir
```

- [ ] Run broader xvm checks:

```bash
ctest --test-dir build -R Ascend --output-on-failure
/home/niu/code/llvm-project/llvm/build/bin/llvm-lit -q build/test/Conversion build/test/Target
```

- [ ] Run host/xvm static guards:

```bash
git diff --check
./test/tools/check_ascend_no_v2_code_naming.sh
./test/tools/check_ascend_public_headers.sh
```

- [ ] Update tracking doc with RED/GREEN evidence and the remaining non-goals.

## Acceptance Criteria

- Static workspace ABI output is unchanged for existing tests.
- Dynamic workspace plans emit `cann.workspace_size_expr` without pretending a static byte count exists.
- CANN tiling-space JSON, runtime manifest, and host tiling helper consume the same expression.
- Host C++ only emits expressions over known tiling shape fields and fails closed on unknown identifiers.
- Dynamic workspace closure does not expand into a general dynamic allocator or unrelated materialization refactor.
