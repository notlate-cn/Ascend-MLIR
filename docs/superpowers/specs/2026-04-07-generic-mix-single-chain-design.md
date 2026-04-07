# Generic Mix Single-Chain Design

## Goal

Make mix lowering genuinely generic for kernels that contain both `AiCore.Cube` and `AiCore.Vector` work, while constraining the first implementation stage to a single linear mix chain:

- one cube region
- explicit boundary transfer
- one vector region

This replaces the current "supported mix kernel" template as the primary long-term direction without attempting full arbitrary-region scheduling in the same step.

## Problem

The current mix path is no longer sample-name driven, but it is still semantically specialized:

- function signature assumptions remain narrow
- task kind is effectively fixed
- the generated kernel shell still assumes a single supported matmul-style mix form
- AIC/AIV emission is template-driven rather than region-driven

That is acceptable as an intermediate compatibility path, but it is not a good long-term foundation for arbitrary cube+vector computations.

## Non-Goals

This stage does not attempt to support:

- arbitrary alternating `cube -> vector -> cube -> vector` execution chains
- automatic task kind search
- arbitrary multi-stage boundary scheduling
- all possible mixed ABI shapes

Those remain follow-up work after the first generic single-chain path is stable.

## Target Scope

The first generic mix stage supports kernels that satisfy all of the following:

1. `ascendc.kernel_kind = "mix"`
2. the partition plan can be linearized as `cube -> boundary -> vector`
3. exactly one cube compute region and one vector compute region participate in execution
4. boundary values are explicit and can be materialized through the existing mix shared-transfer model
5. region-local ops are already individually translatable by the existing cube/vector emit helpers

## Architectural Direction

### 1. Analysis becomes the source of truth

The translator must stop inferring mix behavior from a narrow supported-kernel template. Instead:

- `step3` continues to set `ascendc.kernel_kind`
- later lowering preserves `ascendc.unit`
- translation builds a `MixPartitionPlan`
- emission consumes that plan directly

The plan must be designed for future multi-region growth even if the first executable path only accepts a single chain.

### 2. `MixPartitionPlan` is future-facing

`MixPartitionPlan` should already model:

- ordered regions
- region kind (`cube`, `vector`, `boundary`)
- region-local operations
- explicit boundary inputs and outputs
- enough metadata to validate whether a plan is executable by the current stage

The first stage uses a restricted validator:

- accept exactly one cube region
- accept exactly one vector region
- accept one or more explicit boundary values between them
- reject anything else with a clear "unsupported generic mix shape" diagnostic

### 3. Boundary transfer is a first-class ABI concept

Boundary handling cannot remain implicit inside a handwritten emitter template. The generic path must explicitly represent:

- which values cross from cube to vector
- producer region and consumer region
- storage/transfer class used to bridge the regions
- synchronization points required before the consumer executes

The first stage may reuse the existing transfer strategy, but only through an explicit boundary-emission layer.

### 4. Region-driven emission becomes the primary path

The translator should be organized as:

- kernel shell emission
- cube region emission
- boundary transfer emission
- vector region emission

The current supported mix template may remain temporarily as fallback, but the new generic single-chain region path becomes the primary route for acceptable plans.

### 5. Runtime remains a consumer only

No RuntimeMix component should regain responsibility for generating or rewriting final mix source.

The contract stays:

- analysis in MLIR
- final source emission in translation
- runtime compiles and executes emitted source only

## Data Model

### `MixPartitionPlan`

`MixPartitionPlan` must remain general enough for multi-region follow-up work. At minimum it needs:

- ordered `regions`
- `MixRegionPlan.kind`
- `MixRegionPlan.ops`
- `MixRegionPlan.inputs`
- `MixRegionPlan.outputs`

### `MixBoundaryValue`

Boundary values should remain explicit and deduplicated. They need:

- the SSA value crossing the boundary
- producer partition kind
- consumer partition kind

The first stage assumes cube-to-vector transfer only, but the type must not encode that assumption.

## Translator Behavior

### Eligibility

The generic single-chain path should run only when all of the following are true:

- kernel kind is `mix`
- a valid single-chain region plan exists
- the signature is still representable by the current mix runtime ABI
- each region contains only ops the current emitters know how to lower

If any condition fails, the translator should either:

- fall back to the current legacy supported-mix path if that path is still intentionally retained, or
- emit a clear unsupported diagnostic

### Emission order

For the first stage the emitted structure is still logically:

1. kernel shell
2. AIC branch for cube region
3. boundary transfer and synchronization glue
4. AIV branch for vector region

What changes is that each branch is driven by the region plan rather than by a pre-authored scenario template.

## Error Handling

The translator should reject unsupported generic mix shapes early and explicitly. Error text should identify the violated restriction, for example:

- more than one cube region
- more than one vector region
- no explicit boundary values
- unsupported op present in cube region
- unsupported op present in vector region
- plan cannot be linearized as `cube -> boundary -> vector`

This is preferable to silently routing into a narrow template that only accidentally works.

## Testing Strategy

### Analysis tests

Add tests that verify:

- `ascendc.kernel_kind` remains `mix`
- `ascendc.unit` survives to the stage used by translation
- partition planning produces one cube region, one vector region, and explicit boundary values for the current mix example

### Translation tests

Add tests that verify the new generic single-chain path emits:

- mix kernel shell
- AIC branch
- boundary synchronization/transfer
- AIV branch

These tests should not depend on sample names.

### End-to-end tests

Keep the existing xvm end-to-end validation for:

- `examples/matmul-add-leakyrelu/run.sh`

And retain regression checks ensuring non-mix examples still stay on the plain path.

## Migration Plan

1. Keep the current supported-mix emitter as fallback only during transition.
2. Introduce a generic single-chain eligibility check over `MixPartitionPlan`.
3. Route eligible plans through the generic region-driven path.
4. Re-verify xvm end-to-end correctness.
5. Once stable, shrink legacy supported-mix code to fallback-only or remove it if the generic path fully subsumes it.

## Success Criteria

This design is successful when:

- mix lowering no longer depends on a handwritten scenario template as the primary path
- the first generic stage supports any single-chain `cube -> boundary -> vector` kernel whose region-local ops are otherwise translatable
- `examples/matmul-add-leakyrelu/run.sh` still passes end-to-end on xvm
- non-mix pipelines remain unaffected
- the resulting data model and emitter boundaries are ready for a later multi-region implementation
