# Non-Mix Pipeline Regression Recovery Design

## Goal

Recover the five example pipelines that previously ran end-to-end on CPU simulation with correct accuracy, but regressed after mix-mode support was introduced:

- [`examples/add-broadcast-concat/run.sh`](/Volumes/GM9/code/Ascend-MLIR/examples/add-broadcast-concat/run.sh)
- [`examples/broadcast-add-reduce/run.sh`](/Volumes/GM9/code/Ascend-MLIR/examples/broadcast-add-reduce/run.sh)
- [`examples/gather-elementwise-fusion/run.sh`](/Volumes/GM9/code/Ascend-MLIR/examples/gather-elementwise-fusion/run.sh)
- [`examples/relu-broadcast-transpose/run.sh`](/Volumes/GM9/code/Ascend-MLIR/examples/relu-broadcast-transpose/run.sh)
- [`examples/split-relu-brc-add-mul/run.sh`](/Volumes/GM9/code/Ascend-MLIR/examples/split-relu-brc-add-mul/run.sh)

The recovery must be mechanism-level, not sample-specific:

- no branching on example names
- no patching generated kernels by file name
- no new hardcoded fallback paths in runtime or translator
- no weakening of mix functionality that already passes

## Problem Statement

Recent work introduced:

- `kernel_kind` analysis based on `ascendc.unit`
- mix partition planning
- generic mix region lowering in [`CannTranslation.cpp`](/Volumes/GM9/code/Ascend-MLIR/lib/Target/CannKernel/CannTranslation.cpp)
- runtime and compiler changes to support packed mix execution

Those changes were validated on the mix example [`examples/matmul-add-leakyrelu/run.sh`](/Volumes/GM9/code/Ascend-MLIR/examples/matmul-add-leakyrelu/run.sh), but they also changed code paths shared with ordinary vector-only pipelines.

The five affected examples are expected to remain non-mix. If they no longer pass, the likely failure modes are:

- a vector example is misclassified as mix
- a vector example keeps `kernel_kind=vec` but still enters mix-specific translation or runtime logic
- generic translator changes changed output for non-mix kernels
- runtime/compiler/validator behavior was unintentionally coupled to mix metadata or ABI assumptions
- environment issues mask real regressions and need to be separated from functional failures

## Success Criteria

For each of the five examples, on xvm:

```bash
bash -lc 'source examples/env.sh && bash examples/<case>/run.sh --log'
```

must satisfy all of the following:

1. the pipeline completes without shell failure
2. `.bin` output is generated when the example is supposed to compile a kernel
3. validator/runtime executes rather than failing due to wrong pipeline routing
4. final accuracy check passes with the example's intended tolerances

Additionally:

- [`examples/matmul-add-leakyrelu/run.sh`](/Volumes/GM9/code/Ascend-MLIR/examples/matmul-add-leakyrelu/run.sh) must remain passing
- the fix must be expressed as shared mechanism changes, not per-example special handling

## Non-Goals

- broadening mix support to new `cube + vector` patterns
- redesigning example scripts beyond what is needed to restore their original contract
- hiding environment misconfiguration by adding sample-local library path hacks
- replacing the existing validator or compiler CLI contracts

## Design Principles

### 1. Classify Early, Route Once

Kernel routing must be decided in MLIR based on semantic facts:

- `vec`
- `cube`
- `mix`

Translator and runtime must consume that classification. They must not rediscover it from generated C++, file names, or example names.

### 2. Non-Mix Pipelines Must Be Isolated from Mix Logic

If an example is semantically vector-only, then:

- it must not enter mix emission
- it must not require mix manifests
- it must not depend on mix runtime launch metadata

Mix support may coexist with vector support, but it must not alter vector behavior unless the kernel is actually mix.

### 3. Environment Failures and Functional Failures Must Be Separated

The xvm investigation must record whether a failure is due to:

- missing tool or Python dependency
- missing runtime library
- compilation failure
- runtime launch failure
- wrong numerical result

Only functional failures should drive translator/runtime fixes. Environment failures should be documented separately.

### 4. Fix at Shared Boundaries

Any accepted fix must live in one of these shared layers:

- MLIR pass pipeline
- translator
- runtime/compiler/validator
- common example harness behavior

Fixes must not be accepted if they rely on:

- checking the example directory name
- checking the kernel symbol string against known samples
- injecting sample-only generated source rewrites

## Scope Breakdown

### Target Pipelines

The five target examples are all expected to remain non-mix after the mix work:

- add-broadcast-concat
- broadcast-add-reduce
- gather-elementwise-fusion
- relu-broadcast-transpose
- split-relu-brc-add-mul

One of them currently already shows a distinct environment/runtime failure on xvm:

- add-broadcast-concat reaches compile, then fails in runtime initialization because `libascend_hal.so` cannot be loaded through `libascendcl.so`

This is useful because it demonstrates that the investigation must not collapse all failures into one root cause.

### Comparison Baseline

The expected reference behavior is:

- before the mix work, these pipelines could run fully automatically
- generated kernel code was sufficient unless an example already documented a known hand-fixed path
- CPU simulation and accuracy validation passed

The active passing baseline that must be preserved is:

- `matmul-add-leakyrelu` mix path on xvm

## Proposed Investigation Architecture

### 1. Example Health Matrix

Create a structured matrix for the five examples with these fields:

- expected kernel kind
- first failing stage
- last successful artifact
- whether `step7_cann.mlir` is produced
- whether `step8_kernel.cpp` contains unexpected mix structures
- whether compile succeeds
- whether runtime initializes
- whether accuracy passes
- root-cause bucket

This matrix becomes the single source of truth during the recovery.

### 2. Shared Diagnostics, Not Sample Debugging

For each example, inspect the same checkpoints:

- Stage 3 or earliest stage retaining `ascendc.unit`
- Stage 7 / `step7_kernel.mlir`
- Stage 7b / `step7_cann.mlir`
- Stage 8 / generated C++
- compile result
- validator/runtime result

The purpose is not to debug each sample independently. The purpose is to determine which shared boundary changed.

### 3. Priority Order of Root-Cause Search

Investigation should proceed in this order:

1. kernel-kind misclassification
2. translator mix-path leakage into non-mix codegen
3. runtime/compiler/validator mix-path leakage into non-mix execution
4. example harness assumptions broken by shared tooling changes
5. environment-only issues

This ordering biases toward mechanism bugs before script-local symptoms.

### 4. Recovery Strategy

The repair strategy is:

- make non-mix routing explicit and strict
- restore the previous non-mix translator/runtime behavior through shared gates
- keep mix functionality intact by ensuring its path only triggers for true mix kernels

If a failure is environment-only, record it but do not fold it into functional repair commits.

## Expected Fix Themes

The most likely shared repairs are expected to fall into one or more of these categories:

### A. Tighten `kernel_kind` Propagation or Interpretation

Examples:

- preserve `vec` classification correctly through later passes
- ensure translator only enters mix lowering for `kernel_kind=mix`
- ensure absence of `ascendc.unit=AiCore.Cube` cannot accidentally trigger mix logic

### B. Re-Separate Translator Paths

Examples:

- restore the plain vector emitter path as the only path for non-mix kernels
- make generic mix lowering opt-in on semantic classification only
- prevent helper reuse from accidentally emitting mix shell or task metadata for non-mix kernels

### C. Re-Separate Runtime Paths

Examples:

- keep non-mix compiler/validator launch on the plain `.bin` path
- keep mix manifest and launch-metadata consumption out of non-mix flows
- ensure packed mix execution is only used when the artifact is actually mix

### D. Normalize Harness Expectations

Examples:

- fix common environment assumptions in one place
- ensure examples consistently invoke the same compiler/validator contracts
- avoid sample-specific runtime library workarounds

## Deliverables

This recovery effort should produce:

1. a spec and implementation plan
2. a structured result matrix for the five examples on xvm
3. shared mechanism fixes
4. xvm verification logs showing:
   - all five non-mix examples recovered or explicitly blocked by environment-only issues
   - mix example still passing

## Exit Condition

The effort is complete when:

- the five target examples are revalidated on xvm
- their failures, if any remain, are classified precisely
- all functional regressions caused by mix work are fixed through shared mechanisms
- the validated mix example remains green
- no new sample-name or pattern-name hardcoding has been introduced
