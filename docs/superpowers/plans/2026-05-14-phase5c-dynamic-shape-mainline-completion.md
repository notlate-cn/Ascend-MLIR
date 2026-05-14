# Phase 5C+ Dynamic Shape Mainline Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the current design-vs-implementation gap for the dynamic-shape Ascend mainline and advance from example-driven success to reusable mechanisms.

**Architecture:** Keep the existing Phase 0 -> Phase 5 pipeline intact. Add the smallest durable mechanisms first: explicit tracking, non-duplicated schedule decisions, structured marker metadata, target-aware tile hooks, multi-kernel manifest scaffolding, and a transformer acceptance path that can grow without special-casing the whole graph.

**Tech Stack:** MLIR C++ passes, Ascend Schedule/Realize/Backend/CANN translation, LIT/FileCheck, runtime-session SimBackend, xvm/docker verification via `ssh xvm@orb`.

---

## Scope

This plan completes the six agreed steps:

1. Add a design-vs-implementation gap board to tracking.
2. Remove redundant guard storage from `ScheduleDecision`.
3. Add a structured marker IR MVP for guards and tail plans.
4. Add target-aware tile hook scaffolding and route the current 32-element defaults through it.
5. Add multi-kernel runtime manifest scaffolding without changing the current one-kernel examples.
6. Add a transformer dynamic mainline acceptance path that records the remaining unsupported pieces explicitly.

The plan does not require modifying the CANN installation directory. Host-side edits happen in this repo; build and runtime validation happen in xvm at `/home/niu/code/Ascend-MLIR`.

## File Structure

- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`
  - Add a design-vs-implementation gap board and mark this plan as active.
- Add `test/tools/check_ascend_schedule_decision_contract.sh`
  - Static regression guard for no duplicate guard fields in `ScheduleDecision`.
- Modify `include/Conversion/Ascend/Schedule/ScheduleTypes.h`
  - Remove `ScheduleDecision::candidateGuards` and `ScheduleDecision::decisionGuards`.
  - Add `TargetTilePolicy` and route default tile constants through a named policy.
- Modify `lib/Conversion/Ascend/Schedule/ScheduleDecision.cpp`
  - Print guard counts from `ScheduleDecision::instance`.
  - Keep `tailPlans` as a real refinement result.
- Modify `lib/Conversion/Ascend/Schedule/ScheduleSearch.cpp`
  - Replace raw `kDefaultParallelTile` consumers with target-aware policy helpers.
- Modify `include/Conversion/Ascend/Common/Attributes.h`
  - Add shared attr names for `ascend.schedule.guard_markers`, `ascend.schedule.tail_markers`, and `ascend.schedule.target_tile_policy`.
- Modify `lib/Conversion/Ascend/Schedule/StructuredLoweringDriver.cpp`
  - Serialize guard markers and tail markers to scheduled ops and parent `func.func`.
- Modify `test/Conversion/ascend-schedule-structured-lowering.mlir`
  - Check marker attrs are emitted.
- Modify `lib/Target/CannKernel/CannRuntimeArtifacts.cpp`
  - Emit `kernel_entries` manifest scaffolding for one or more scheduled kernels while preserving existing fields.
- Modify `test/Target/cann-translate-runtime-artifacts.mlir`
  - Check `kernel_entries` contains the existing one-kernel case.
- Modify `examples/transformer/run-mainline.sh`
  - Add a transformer dynamic smoke entrypoint that runs the supported prefix and records unsupported full-graph gaps without claiming full transformer completion.
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`
  - Record verification rows for this plan.

## Task 1: Tracking Gap Board

**Files:**
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Add the gap board**

Add a section named `## Design vs Implementation Gap Board` with rows for:

```markdown
| Area | Design Target | Current Implementation | Status | Next Closure Step |
|---|---|---|---|---|
| ScheduleDecision ownership | `ScheduleDecision` refines `ScheduleInstance` without copying instance fields | `ScheduleDecision` currently owns `instance`; guard counts must be read through `instance` | Closing | Remove duplicate guard fields |
| Structured markers | `TailPlanMarker` / guard markers / cache markers are explicit IR carriers | Schedule currently emits selected tile and tail metadata attrs; guard/tail marker attrs are MVP scope | Closing | Emit guard/tail marker attrs |
| Target-driven tile | Tile selection consumes target memory / intrinsic / cost model | Current role-driven `[32,N]` policy is routed through a target tile policy hook | Closing | Replace raw constants with target policy |
| Multi-kernel manifest | Runtime manifest can describe multiple kernel entries | Current artifacts model one primary kernel per function | Planned | Emit one-entry `kernel_entries` scaffold |
| Transformer dynamic | Full transformer graph compiles through new mainline | Supported examples and transformer smoke exist; full graph still needs DAG/unsupported-op closure | Planned | Add explicit transformer acceptance entrypoint |
```

- [ ] **Step 2: Verify markdown diff**

Run:

```bash
git diff -- docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md
git diff --check -- docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md
```

Expected: no whitespace errors.

## Task 2: ScheduleDecision Guard Ownership

**Files:**
- Add: `test/tools/check_ascend_schedule_decision_contract.sh`
- Modify: `include/Conversion/Ascend/Schedule/ScheduleTypes.h`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleDecision.cpp`

- [ ] **Step 1: Write the failing static regression**

Create `test/tools/check_ascend_schedule_decision_contract.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if sed -n '/struct ScheduleDecision {/,/};/p' include/Conversion/Ascend/Schedule/ScheduleTypes.h |
   grep -E 'SmallVector<ScheduleGuard> +(candidateGuards|decisionGuards)' >/dev/null; then
  echo "ScheduleDecision must not duplicate ScheduleInstance guard vectors" >&2
  exit 1
fi

if grep -R -nE 'decision[.]candidateGuards|decision[.]decisionGuards|selectedDecision[.]candidateGuards|selectedDecision[.]decisionGuards' \
    include/Conversion/Ascend lib/Conversion/Ascend test/Conversion >/dev/null; then
  echo "ScheduleDecision guard users must read through decision.instance" >&2
  exit 1
fi
```

- [ ] **Step 2: Verify RED**

Run:

```bash
bash test/tools/check_ascend_schedule_decision_contract.sh
```

Expected: FAIL with `ScheduleDecision must not duplicate ScheduleInstance guard vectors`.

- [ ] **Step 3: Remove duplicate fields**

In `include/Conversion/Ascend/Schedule/ScheduleTypes.h`, remove these fields from `struct ScheduleDecision`:

```cpp
SmallVector<ScheduleGuard> candidateGuards;
SmallVector<ScheduleGuard> decisionGuards;
```

In `lib/Conversion/Ascend/Schedule/ScheduleDecision.cpp`, remove:

```cpp
decision.candidateGuards = instance.candidateGuards;
decision.decisionGuards = instance.decisionGuards;
```

Update report printing to:

```cpp
const ScheduleInstance &selectedInstance = selectedDecision.instance;
os << "  candidate_guards = "
   << selectedInstance.candidateGuards.size() << "\n";
os << "  decision_guards = " << selectedInstance.decisionGuards.size()
   << "\n";
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
bash test/tools/check_ascend_schedule_decision_contract.sh
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build afir-opt && PATH=/home/niu/code/llvm-project/llvm/build/bin:$PATH /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion --filter="ascend-schedule-(decision-set|structured-lowering|guards|search)"'
```

Expected: static check passes and focused LIT passes.

## Task 3: Structured Guard and Tail Marker MVP

**Files:**
- Modify: `include/Conversion/Ascend/Common/Attributes.h`
- Modify: `include/Conversion/Ascend/Schedule/ScheduleTypes.h`
- Modify: `lib/Conversion/Ascend/Schedule/StructuredLoweringDriver.cpp`
- Modify: `test/Conversion/ascend-schedule-structured-lowering.mlir`

- [ ] **Step 1: Write failing LIT checks**

Add checks to `test/Conversion/ascend-schedule-structured-lowering.mlir`:

```mlir
// CHECK-SAME: ascend.schedule.guard_markers
// CHECK-SAME: ascend.schedule.tail_markers
```

Expected before implementation: FAIL because those attrs do not exist.

- [ ] **Step 2: Add shared attr names**

Add to `include/Conversion/Ascend/Common/Attributes.h`:

```cpp
inline constexpr llvm::StringLiteral kScheduleGuardMarkersAttr =
    "ascend.schedule.guard_markers";
inline constexpr llvm::StringLiteral kScheduleTailMarkersAttr =
    "ascend.schedule.tail_markers";
inline constexpr llvm::StringLiteral kScheduleTargetTilePolicyAttr =
    "ascend.schedule.target_tile_policy";
```

Add `using` declarations in `include/Conversion/Ascend/Schedule/ScheduleTypes.h`.

- [ ] **Step 3: Serialize guard markers**

In `StructuredLoweringDriver.cpp`, build an array of dictionaries from `selectedDecision.instance.decisionGuards`:

```cpp
{ kind = "...", domain = "...", dim = i64, value = i64, text = "..." }
```

Use `candidate` guards only for schedule selection; marker IR records decision guards because these are the structured runtime conditions.

- [ ] **Step 4: Serialize tail markers**

Reuse `selectedDecision.tailPlans` to build an array of dictionaries:

```cpp
{ axis = i64, selected = "...", align = i64, buffering = "...", guard = bool }
```

Attach `guard_markers` and `tail_markers` to each scheduled op and preserve the first complete set on the parent `func.func`.

- [ ] **Step 5: Verify marker attrs**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build afir-opt && PATH=/home/niu/code/llvm-project/llvm/build/bin:$PATH /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion --filter="ascend-schedule-structured-lowering"'
```

Expected: LIT passes and IR contains both marker attrs.

## Task 4: Target-Aware Tile Hook MVP

**Files:**
- Modify: `include/Conversion/Ascend/Schedule/ScheduleTypes.h`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleSearch.cpp`
- Modify: `lib/Conversion/Ascend/Schedule/StructuredLoweringDriver.cpp`
- Modify: `test/Conversion/ascend-schedule-search.mlir`

- [ ] **Step 1: Write failing checks**

Add FileCheck expectations that scheduled ops carry:

```mlir
// CHECK-SAME: ascend.schedule.target_tile_policy = "target_default_32"
```

Expected before implementation: FAIL because the attr is missing.

- [ ] **Step 2: Add tile policy data**

Add:

```cpp
struct TargetTilePolicy {
  std::string policyId = "target_default_32";
  int64_t defaultParallelTile = 32;
};
```

Add `TargetTilePolicy targetTilePolicy;` to `ScheduleProblem`.

- [ ] **Step 3: Replace raw tile constant access**

Keep `32` as the MVP policy value, but read it through:

```cpp
static int64_t getDefaultParallelTile(const ScheduleProblem &problem) {
  return problem.targetTilePolicy.defaultParallelTile;
}
```

Do not query CANN directly in this step; this is the hook that later consumes `TargetMemoryModel`.

- [ ] **Step 4: Emit target policy metadata**

Attach `ascend.schedule.target_tile_policy = "target_default_32"` to scheduled ops and parent function in `StructuredLoweringDriver.cpp`.

- [ ] **Step 5: Verify**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build afir-opt && PATH=/home/niu/code/llvm-project/llvm/build/bin:$PATH /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion --filter="ascend-schedule-(search|structured-lowering|vector-bounded-tile)"'
```

Expected: focused schedule LIT passes.

## Task 5: Multi-Kernel Manifest Scaffold

**Files:**
- Modify: `lib/Target/CannKernel/CannRuntimeArtifacts.cpp`
- Modify: `test/Target/cann-translate-runtime-artifacts.mlir`

- [ ] **Step 1: Write failing manifest checks**

In `test/Target/cann-translate-runtime-artifacts.mlir`, add checks for:

```json
"kernel_entries": [
  {
    "kernel_id": "kernel_0",
    "entry_index": 0,
    "selected_tile_shape": [
```

Expected before implementation: FAIL because the manifest has only top-level schedule metadata.

- [ ] **Step 2: Emit one-entry scaffold**

In `CannRuntimeArtifacts.cpp`, when schedule metadata is present, add:

```json
"kernel_entries": [
  {
    "kernel_id": "kernel_0",
    "entry_index": 0,
    "selected_tile_shape": ...,
    "tail_policies": ...,
    "tail_plan": ...
  }
]
```

Preserve the existing top-level `selected_tile_shape`, `tail_policies`, and `tail_plan` fields for compatibility.

- [ ] **Step 3: Verify target tests**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build afir-translate && PATH=/home/niu/code/llvm-project/llvm/build/bin:$PATH /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Target --filter="cann-translate-runtime-artifacts"'
```

Expected: runtime artifact tests pass.

## Task 6: Transformer Dynamic Acceptance Entry

**Files:**
- Create or modify: `examples/transformer/run-mainline.sh`
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Inspect current transformer inputs**

Run:

```bash
find examples/transformer -maxdepth 2 -type f -print
```

Expected: identify `transformer_dynamic.mlir` and any existing data/scripts.

- [ ] **Step 2: Add explicit smoke script**

Create `examples/transformer/run-mainline.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
BUILD_DIR="$DIR/build_mainline"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

"$AFIR_OPT" "$DIR/transformer_dynamic.mlir" \
  --ascend-normalize \
  -o "$BUILD_DIR/step1_normalized.mlir"

"$AFIR_OPT" "$BUILD_DIR/step1_normalized.mlir" \
  --ascend-kernelize \
  -o "$BUILD_DIR/step2_kernelized.mlir"

if "$AFIR_OPT" "$BUILD_DIR/step2_kernelized.mlir" \
  --ascend-compute-lower \
  -o "$BUILD_DIR/full_codegen_from_prefix_unexpected.mlir" \
  2> "$BUILD_DIR/full_codegen_from_prefix.stderr"; then
  echo "transformer_dynamic.full_codegen=unexpected-pass" >&2
  exit 1
fi

if ! grep -q "unsupported" "$BUILD_DIR/full_codegen_from_prefix.stderr"; then
  echo "transformer_dynamic.full_codegen=missing-unsupported-diagnostic" >&2
  cat "$BUILD_DIR/full_codegen_from_prefix.stderr" >&2
  exit 1
fi

echo "transformer_dynamic.mainline_prefix=pass"
echo "transformer_dynamic.full_codegen=deferred"
echo "transformer_dynamic.next_gap=unsupported_op_closure"
```

This script must not claim full transformer codegen until the full graph reaches CANN translation and runtime sim.

- [ ] **Step 3: Verify transformer prefix**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/transformer/run-mainline.sh'
```

Expected output includes `transformer_dynamic.mainline_prefix=pass`.

- [ ] **Step 4: Update tracking**

Add a row:

```markdown
| transformer_dynamic mainline prefix | `In Progress` | Normalize + Kernelize prefix smoke exists; full Schedule/Realize/Codegen still waits for multi-kernel DAG and unsupported-op closure | `examples/transformer/run-mainline.sh` |
```

## Final Verification

Run:

```bash
git diff --check
bash test/tools/check_ascend_schedule_decision_contract.sh
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build afir-opt afir-translate runtime-session mix-compiler'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && PATH=/home/niu/code/llvm-project/llvm/build/bin:$PATH /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion --filter="ascend-"'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && PATH=/home/niu/code/llvm-project/llvm/build/bin:$PATH /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Target --filter="cann-translate-runtime-artifacts"'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash test/tools/runtime/run_simbackend_examples.sh broadcast-add-reduce relu-broadcast-transpose add-broadcast-concat gather-elementwise-fusion split-relu-brc-add-mul matmul-add-leakyrelu'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/transformer/run-mainline.sh'
```

Expected:

- Static checks pass.
- Ascend conversion LIT passes.
- Runtime artifact LIT passes.
- 6-example SimBackend smoke passes.
- Transformer prefix smoke reports `transformer_dynamic.mainline_prefix=pass`.

## Self-Review

- No step claims full transformer CANN/runtime completion before it exists.
- Multi-kernel manifest is scaffolded as a backward-compatible one-entry array.
- Target-aware tiling is a hook using the current default value, not a fake full cost model.
- `ScheduleDecision` guard ownership matches V2-4: guard vectors live in `ScheduleInstance`; `ScheduleDecision` keeps only refinement fields.
