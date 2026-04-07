# Generic Mix Region Codegen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current supported-mix special-case translator path with a genuinely generic region-driven mix codegen architecture while preserving the currently passing xvm path during migration.

**Architecture:** Introduce a `MixPartitionPlan` that captures ordered cube/vector/boundary regions and explicit boundary-crossing facts. Use that plan to drive generic region emitters, while keeping the current supported-mix path only as a temporary fallback until the generic route proves itself on the validated example.

**Tech Stack:** MLIR/LLVM C++, `afir-translate`, AscendC mix kernel codegen, xvm/orb simulator validation, git

---

### Task 1: Introduce `MixPartitionPlan` Data Model

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Test: `examples/matmul-add-leakyrelu/run.sh`

- [ ] **Step 1: Add explicit partition-plan structs**

Add region-plan types near the existing mix analysis structures:

```c++
struct MixBoundaryValue {
  Value value;
  MixPartitionKind producer;
  MixPartitionKind consumer;
};

struct MixRegionPlan {
  MixPartitionKind kind;
  SmallVector<Operation *> ops;
  SmallVector<MixBoundaryValue> inputs;
  SmallVector<MixBoundaryValue> outputs;
};

struct MixPartitionPlan {
  SmallVector<MixRegionPlan> regions;
  bool empty() const { return regions.empty(); }
};
```

Keep the initial model minimal. Do not add speculative fields.

- [ ] **Step 2: Add a plan builder that derives initial regions from the existing summary**

Add a helper:

```c++
static MixPartitionPlan buildInitialMixPartitionPlan(
    func::FuncOp funcOp, const MixPartitionSummary &summary);
```

Initial behavior:
- create one cube region from `summary.cubeOps`
- create one boundary region from `summary.boundaryOps`
- create one vector region from `summary.vectorOps`
- omit empty regions

This is not the final generic algorithm; it is the migration bridge.

- [ ] **Step 3: Rebuild translator and confirm no behavior change**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build --target afir-translate -j4'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./build/bin/afir-translate -mlir-to-cann examples/matmul-add-leakyrelu/step7_cann.mlir | grep -n "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)\\|CrossCoreSetFlag<0x2, PIPE_FIX>(3)\\|CrossCoreWaitFlag(3)\\|LeakyRelu"'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log | tail -n 20'
```

Expected:
- grep lines are present
- final lines include zero diffs and `PASS`

- [ ] **Step 4: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp
git commit -m "Introduce mix partition plan scaffolding"
```

### Task 2: Make Boundary Crossings Explicit

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Test: `examples/matmul-add-leakyrelu/run.sh`

- [ ] **Step 1: Record explicit boundary-crossing facts**

Extend the plan builder so it captures crossing values for the current shape:

```c++
static SmallVector<MixBoundaryValue>
collectMixBoundaryValues(const MixPartitionSummary &summary);
```

Initial scope:
- identify values whose producer and consumer partitions differ
- record producer/consumer partition kinds

- [ ] **Step 2: Attach boundary facts to the boundary region**

Populate the boundary region’s `inputs`/`outputs` using `MixBoundaryValue`.
Do not try to solve arbitrary storage placement yet. Just make the crossing explicit in the plan.

- [ ] **Step 3: Rebuild translator and rerun xvm validation**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build --target afir-translate -j4'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./build/bin/afir-translate -mlir-to-cann examples/matmul-add-leakyrelu/step7_cann.mlir | grep -n "CrossCoreSetFlag<0x2, PIPE_FIX>(3)\\|CrossCoreWaitFlag(3)"'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log | tail -n 20'
```

Expected:
- both cross-core lines still appear
- final lines include zero diffs and `PASS`

- [ ] **Step 4: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp
git commit -m "Record explicit mix boundary values"
```

### Task 3: Introduce Region-Driven Emission API

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Test: `examples/matmul-add-leakyrelu/run.sh`

- [ ] **Step 1: Add region-driven emitter entry points**

Add:

```c++
static void emitMixCubeRegion(raw_ostream &os,
                              const MixRegionPlan &region,
                              const SupportedMixKernelConfig &config,
                              const MixTaskKindDescriptor &desc);
static void emitMixBoundaryRegion(raw_ostream &os,
                                  const MixRegionPlan &region,
                                  const SupportedMixKernelConfig &config,
                                  const MixTaskKindDescriptor &desc);
static void emitMixVectorRegion(raw_ostream &os,
                                const MixRegionPlan &region,
                                const SupportedMixKernelConfig &config,
                                const MixTaskKindDescriptor &desc);
```

Initially these may delegate to the existing supported-mix emit helpers, but the plan region must become an explicit input.

- [ ] **Step 2: Route current supported emission through region-driven API**

Update the mix emitter to:
- build the initial partition plan
- pick the first cube/boundary/vector regions from that plan
- call the new region-driven emitters

The old helper bodies can remain as implementation details behind the new API.

- [ ] **Step 3: Rebuild translator and rerun xvm validation**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build --target afir-translate -j4'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./build/bin/afir-translate -mlir-to-cann examples/matmul-add-leakyrelu/step7_cann.mlir | grep -n "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)\\|mm.SetBias\\|LeakyRelu"'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log | tail -n 20'
```

Expected:
- all grep lines appear
- final lines include zero diffs and `PASS`

- [ ] **Step 4: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp
git commit -m "Route supported mix emission through region plans"
```

### Task 4: Demote the Legacy Supported Path to Fallback

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Test: `examples/matmul-add-leakyrelu/run.sh`

- [ ] **Step 1: Separate capability gating for generic and legacy paths**

Add distinct helpers:

```c++
static bool canLowerGenericMixPlan(const MixPartitionPlan &plan);
static bool canLowerLegacySupportedMix(func::FuncOp funcOp,
                                       const MixPartitionSummary &summary);
```

For this stage:
- `canLowerGenericMixPlan(...)` should return true only for the currently validated region shape
- `canLowerLegacySupportedMix(...)` should preserve today’s narrow fallback logic

- [ ] **Step 2: Make the generic route the primary path**

Update the top-level mix flow to:
1. build summary
2. build plan
3. try generic region-driven emission first
4. fall back to legacy supported path only if generic lowering says no
5. otherwise error clearly

- [ ] **Step 3: Rebuild translator and rerun xvm validation**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build --target afir-translate -j4'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./build/bin/afir-translate -mlir-to-cann examples/matmul-add-leakyrelu/step7_cann.mlir | grep -n "CrossCoreSetFlag<0x2, PIPE_FIX>(3)\\|CrossCoreWaitFlag(3)\\|LeakyRelu"'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log | tail -n 20'
```

Expected:
- grep lines are present
- final lines include zero diffs and `PASS`

- [ ] **Step 4: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp
git commit -m "Make generic mix region lowering primary path"
```

### Task 5: Expand Translator Test Coverage Around the New Generic Path

**Files:**
- Modify: `test/Target/cann-translate-mix.mlir`
- Modify: `test/Target/cann-translate-mix-relu.mlir`
- Modify: `test/Target/cann-translate-mix-no-bias.mlir`
- Test: `examples/matmul-add-leakyrelu/run.sh`

- [ ] **Step 1: Add checks that reflect region-driven lowering, not only final strings**

Add or tighten checks so the translator tests verify:
- mix task type
- cube/vector region structure
- boundary sync lines
- relu/leaky-relu variant behavior

Do not weaken any existing checks.

- [ ] **Step 2: Manually inspect translator output on xvm**

Because xvm may not provide `FileCheck`, run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./build/bin/afir-translate -mlir-to-cann examples/matmul-add-leakyrelu/step7_cann.mlir | sed -n "1,120p"'
```

Expected:
- output structure matches the strengthened checks

- [ ] **Step 3: Final xvm end-to-end confirmation**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log | tail -n 20'
```

Expected:
- final lines include `max_abs_diff=0.000000e+00`
- final lines include `mean_abs_diff=0.000000e+00`
- final lines include `PASS`

- [ ] **Step 4: Commit**

```bash
git add test/Target/cann-translate-mix.mlir test/Target/cann-translate-mix-relu.mlir test/Target/cann-translate-mix-no-bias.mlir
git commit -m "Expand generic mix translator coverage"
```
