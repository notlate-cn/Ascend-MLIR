# Generic Mix Region Codegen Design

## Goal

Replace the current supported-mix special-case emitter with a genuinely generic mix codegen path that can handle arbitrary kernels containing both `cube` and `vector` class operations.

The end state is:

- mix detection is derived from IR semantics, not pattern names
- translator consumes a region-level mix partition plan
- AIC/AIV emission is driven by partition contents, not by a fixed `matmul + relu/leaky_relu` template
- the current supported-mix implementation becomes a temporary fallback and is eventually removable

## Problem Statement

The current mix path in [`CannTranslation.cpp`](/Volumes/GM9/code/Ascend-MLIR/.worktrees/mix-kind-translate/lib/Target/CannKernel/CannTranslation.cpp) is structurally cleaner than before, but it is still semantically narrow.

What is hardcoded today:

- fixed function signature shape
- fixed task kind `MixAic1To2`
- fixed cube-side emission around a matmul object
- fixed vector-side emission around a relu/leaky-relu queue pipeline
- fixed boundary behavior around `cGM` plus a single cross-core flag pair

This is acceptable as a validated bridge for the current example, but it is not a general mix toolchain.

If the requirement is “support arbitrary `cube + vector` op computation”, then the current emitter architecture is still wrong at the last step:

- analysis is partially generic
- emission is still scenario-specific

## Design Principles

### 1. Partition-Driven, Not Pattern-Driven

Codegen must not branch on “matmul + bias + relu” style scenario identities.

Instead:

- `step3` / `step7` classify `kernel_kind = mix`
- each op is assigned to `cube`, `vector`, or `boundary`
- emission walks those partitions

### 2. Region Is the Unit of Emission

The translator should not treat individual core ops as the emission anchor. The correct abstraction is a region:

- `cube region`
- `boundary region`
- `vector region`

Each region includes:

- its core ops
- helper ops needed only by that region
- the region-local buffer / queue / tensor setup required to execute it

### 3. Boundary Is an Explicit ABI

Values that cross from cube execution to vector execution must be materialized explicitly.

That means the partition plan must define:

- which values cross the boundary
- their storage class
- synchronization requirements
- execution order between producer and consumer regions

The current implicit “write to `cGM`, then `CrossCoreSetFlag/WaitFlag`” behavior should become one possible boundary lowering strategy, not a hardcoded law of the translator.

### 4. Emission Must Be Layered

The generic mix codegen path should be layered as:

1. `kernel kind analysis`
2. `mix partition analysis`
3. `mix partition plan construction`
4. `region lowering strategy selection`
5. `AIC/AIV code emission`

No layer should reach backward and rediscover facts already computed by an earlier layer.

## Proposed Architecture

### 1. Kernel Kind Analysis

This remains based on `ascendc.unit`:

- only `AiCore.Vector` -> `vec`
- only `AiCore.Cube` -> `cube`
- both -> `mix`

This is already the correct source of truth and should remain unchanged.

### 2. Mix Partition Summary -> Mix Partition Plan

The current `MixPartitionSummary` is too weak for generic codegen. It answers:

- do cube ops exist
- do vector ops exist
- do boundary ops exist

That is enough for capability detection, but not for generic emission.

It must be replaced or extended by a `MixPartitionPlan` that records:

- ordered region list
- per-region op membership
- region kind: `cube`, `vector`, `boundary`
- region inputs and outputs
- boundary-crossing values
- storage assignment for boundary values
- synchronization requirements

This plan is the key new artifact.

### 3. Region Ownership Rules

Ownership should come from `ascendc.unit` first, then def-use closure.

Rules:

- ops with `ascendc.unit = AiCore.Cube` belong to the cube side
- ops with `ascendc.unit = AiCore.Vector` belong to the vector side
- helper ops without explicit unit are absorbed into the nearest side they serve
- if a value is produced on one side and consumed on the other, the crossing is represented explicitly in the boundary plan

This avoids falling back to op-name pattern matching as the main classifier.

### 4. Boundary Lowering Strategy

The boundary plan must decide, per crossing:

- storage location:
  - GM-backed tensor
  - queue/buffer backed tensor
  - other supported storage form
- producer side flush / copy behavior
- consumer side load / copy behavior
- synchronization primitive

For the current validated path, the first generic strategy can still be:

- materialize crossing tensor in GM
- producer writes GM
- `CrossCoreSetFlag`
- consumer `CrossCoreWaitFlag`
- consumer reads GM

But this strategy must be represented explicitly as a lowering decision, not embedded in a single special-case emitter.

### 5. Generic Region Emitters

The translator should introduce region-oriented emitters:

- `emitMixCubeRegion(...)`
- `emitMixBoundaryRegion(...)`
- `emitMixVectorRegion(...)`

These emitters must consume partition-plan regions, not special-case semantic booleans like `hasBiasAdd`.

Within them, operation emission should reuse the same lower-level op emission facilities used elsewhere in `CannTranslation.cpp` as much as possible, rather than building large handwritten string templates.

### 6. Strategy-Fallback Model

During migration, the translator can support two mix paths:

1. generic region codegen
2. legacy supported-mix fallback

Routing rule:

- if `MixPartitionPlan` is fully lowerable by the generic path, use it
- otherwise, if the kernel matches the already validated narrow supported shape, use the current fallback
- otherwise emit a clear “mix lowering not yet supported” error

This keeps the passing example stable while generic support is brought online incrementally.

## Migration Plan

### Phase 1: Introduce `MixPartitionPlan`

Build the new plan alongside the current summary without changing emission yet.

Success condition:

- translator can print or internally inspect a correct region plan for the current mix example

### Phase 2: Lower Boundary Plan Explicitly

Implement an explicit boundary representation and one concrete lowering strategy:

- GM crossing tensor
- cross-core flag synchronization

Success condition:

- current example can be expressed through the generic boundary representation

### Phase 3: Generic Cube/Vector Region Emitters

Make cube and vector emitters consume partition-plan regions rather than hardcoded scenario booleans.

Success condition:

- current example translates through generic region emitters, while preserving identical runtime behavior

### Phase 4: Fallback Demotion

Keep the legacy supported-mix path only as fallback.

Success condition:

- current example takes the generic route by default
- legacy route is no longer on the main path

### Phase 5: Expand Supported Mix Semantics

Only after the above is complete, start broadening support:

- more vector epilogues
- more cube/vector interleavings
- more boundary patterns
- more task kinds and ratios

## Testing Strategy

### Translator Structure Tests

Add translator tests that validate:

- mix partition plan construction
- boundary crossing detection
- generic region emission structure

### Variant Coverage

Retain and expand current mix translator variants:

- bias + leaky relu
- bias + relu
- no bias + leaky relu

Then add new variants that stress the generic path without changing runtime behavior goals.

### Runtime Validation

xvm execution remains mandatory for every migration stage:

```bash
source examples/env.sh
bash examples/matmul-add-leakyrelu/run.sh --log
```

Required result:

- `max_abs_diff=0`
- `mean_abs_diff=0`
- `PASS`

## Risks

### Region plan too weak

If the new plan records only region membership but not boundary/storage/sync facts, emission will fall back into hidden template logic.

### Boundary abstraction too ambitious too early

Trying to solve every possible cross-region storage pattern at once will slow the migration. Start with one explicit strategy and keep the abstraction extensible.

### Regressing the passing path during migration

This is why the legacy fallback should remain until the generic path proves itself on xvm.

## Success Criteria

This project is complete when:

- generic mix codegen is driven by `MixPartitionPlan`
- translator no longer depends on a handwritten `matmul + relu/leaky_relu` template for the main mix path
- current validated example runs through the generic path
- legacy supported-mix code is either removable or clearly isolated as fallback
- the architecture can naturally extend to arbitrary `cube + vector` kernels without adding more scenario-specific emitters
