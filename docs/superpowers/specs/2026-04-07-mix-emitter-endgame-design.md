# Mix Emitter Endgame Design

## Goal

Finish the remaining mix-translation work in a single directed effort without expanding supported semantics beyond the already validated `matmul + bias + relu/leaky_relu` path. The result should leave `afir-translate` structured around explicit analysis and region-emission boundaries rather than a growing pile of supported-case string templates.

## Current State

The mix path is already stable for the validated example:

- `step3` annotates `ascendc.kernel_kind`
- `afir-translate` emits final mix-ready `step8_kernel.cpp`
- `RuntimeMix` consumes final source instead of materializing `step8b`
- xvm end-to-end execution passes with `max_abs_diff=0`, `mean_abs_diff=0`

`lib/Target/CannKernel/CannTranslation.cpp` has also already been decomposed substantially:

- signature predicate
- partition predicate
- boundary predicate
- task-kind/config inference
- kernel shell helpers
- AIC/AIV emit helpers

What remains is not “make it work” but “finish making the implementation replaceable.” Today, the supported mix path is still fundamentally a fixed template with increasingly clean helper seams. The end state should preserve current behavior while making the next step, a true partition-driven region emitter, mechanically local.

## Non-Goals

This effort does not expand feature support to:

- new mix task ratios
- non-`relu`/`leaky_relu` vector epilogues
- new runtime ABI shapes
- new host/runtime functionality
- arbitrary mix kernels beyond the currently validated supported shape

Those are follow-on efforts after the structure is fully cleaned up.

## Recommended Approach

Use a two-layer endgame:

1. finish the current supported mix path as a fully isolated “reference emitter”
2. place a thin partition-driven region-emission boundary in front of it

This preserves the passing path while removing the last structural blockers to a future general emitter.

## Final Architecture

### 1. Supported Mix Analysis Layer

`CannTranslation.cpp` should expose explicit helpers for every currently supported fact:

- `hasSupportedMixFunctionSignature(...)`
- `hasSupportedMixPartitions(...)`
- boundary classification helpers
- bias presence inference
- epilogue inference
- task-kind inference

These helpers must be analysis-only. They should not emit code, and they should not contain template-string fragments.

### 2. Supported Mix Configuration Layer

The analysis helpers should feed a compact `SupportedMixKernelConfig` that contains only semantic facts needed by emission:

- `taskKind`
- `hasBiasAdd`
- `epilogueKind`
- `leakyReluAlpha`

Task-kind-derived execution constants should stay centralized through a single descriptor entry point so that task type spelling, cross-core mode, vector ratio, and flag id remain impossible to drift apart.

### 3. Region Emitter Boundary

The current AIC/AIV template should be expressed as three explicit emission responsibilities:

- `emitMixCubeRegion(...)`
- `emitMixVectorRegion(...)`
- `emitMixBoundarySync(...)`

During this effort, these helpers may still emit the current supported shape only. The important change is that the translator body no longer “is” the template; instead, it orchestrates region emission through named boundaries.

This is the key endgame move. Once this boundary exists, replacing the internals with truly partition-driven emission becomes a local change instead of another whole-function rewrite.

### 4. Stable Kernel Shell

The kernel shell should remain separate from region emission:

- includes and namespaces
- tiling-copy helper
- kernel signature
- common prologue/epilogue wrapper

The shell should not know details of bias, epilogue choice, or AIC/AIV internals beyond calling region emitters.

## Execution Model

The supported mix lowering stays intentionally narrow:

1. verify supported function signature
2. verify supported partition structure
3. infer supported mix config
4. emit shell
5. emit cube region
6. emit synchronization boundary
7. emit vector region

If any supported-shape requirement is not met, translation should continue to fail clearly and early rather than silently widening semantics.

## Testing and Validation

Every structural change must preserve two kinds of evidence:

1. translator output evidence
- `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`
- `mm.SetBias(biasGM);` when bias is present
- `CrossCoreSetFlag<0x2, PIPE_FIX>(3)`
- `CrossCoreWaitFlag(3)`
- `uint32_t count = ... / 2`
- `LeakyRelu(... 0.001000f ...)` or `Relu(...)` for the relu variant

2. xvm end-to-end evidence
- `bash examples/matmul-add-leakyrelu/run.sh --log`
- final `max_abs_diff=0`
- final `mean_abs_diff=0`
- `PASS`

Translator regression files should continue to lock the current supported behavior. The xvm run remains the final proof that translator structure changes did not disturb runtime execution.

## Remaining Work Breakdown

### Phase 1: Finish Structural Decomposition

Remove the last local lambdas and mixed-responsibility helper bodies from supported mix emission. The target is a file where each helper has one clear job and the top-level supported mix path reads as orchestration instead of implementation detail.

### Phase 2: Introduce Explicit Region-Emission Interface

Create named region-emission helpers for cube region, vector region, and boundary synchronization. Initially these helpers can still emit the current supported shape.

### Phase 3: Align Analysis with Emission Inputs

Make sure the config and region emitters consume only explicit analysis outputs, not accidental template knowledge. This mostly means tightening the remaining helper signatures and eliminating hidden assumptions.

### Phase 4: Reduce “Supported Mix” to a Capability Gate

After the region boundary is in place, the supported path should read as:

- capability check
- config inference
- region emission

At that point, the next project can focus on broadening support instead of cleaning up structure.

## Risks

### Over-refactoring without semantic gain

Mitigation: every change remains xvm-verified and commit-sized. No broad speculative rewrites.

### Accidentally widening supported semantics

Mitigation: preserve explicit supported-shape checks and fail closed.

### Losing runtime-correct string details during refactor

Mitigation: keep translator-output checks and the xvm numerical pass as mandatory evidence after every stage.

## Success Criteria

This effort is complete when all of the following are true:

- current xvm `run.sh` path still passes with zero diff
- supported mix translation is organized into analysis/config/shell/region boundaries
- AIC/AIV emission no longer relies on hidden inline structure
- task-kind-derived execution constants remain centralized
- supported-mix capability checks are explicit and isolated
- the remaining delta to a future general partition-driven emitter is local rather than architectural
