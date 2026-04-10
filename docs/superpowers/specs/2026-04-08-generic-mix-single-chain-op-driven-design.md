# Generic Mix Single-Chain Op-Driven Design

## Goal

Keep the first-stage `mix` scope constrained to a single executable chain:

- one `cube` region
- one explicit `boundary` region
- one `vector` region

But replace the remaining fixed vector/cube template assumptions with an op-driven emission model. The translator should consume region-local ops from `MixPartitionPlan` and dispatch them through reusable emitters, instead of assuming a hardcoded `matmul + bias + relu/leakyrelu` shape.

## Non-Goals

This stage does not attempt to solve:

- multi-region linear chains such as `cube -> vector -> cube -> vector`
- arbitrary mix DAG scheduling
- automatic task-kind search
- fully generic shell/ABI lowering

The shell, ABI, and current single-chain mix task-kind constraints may remain narrow in this phase. The change is focused on region-local op emission.

## Current Limitation

The current `generic single-chain` path is structurally stronger than before, but emission still relies on supported-mix shell helpers that assume a narrow semantic shape. Even though validation and boundary selection are now more generic, the emitted body is not yet truly driven by the region-local op list.

That leaves two problems:

1. region-local extensibility is poor
2. future multi-region work would still be blocked on template-heavy body emission

## Target Architecture

### 1. Keep `MixPartitionPlan` as the source of truth

`MixPartitionPlan` remains the primary structural model for the mix translator:

- ordered regions
- region kind
- region-local ops
- explicit boundary inputs/outputs
- validated single-chain crossings

The op-driven emitter must consume this plan directly.

### 2. Split shell emission from region-op emission

The mix translator should be separated into two layers:

- shell/ABI/task-kind layer
- region-op emission layer

The shell layer is still allowed to enforce current single-chain ABI restrictions. The region-op layer must not assume a fixed semantic pattern beyond what is required by the op emitters themselves.

### 3. Introduce op-driven region emission

For each region kind, translator emission should iterate region-local ops and dispatch them through dedicated emit helpers.

Conceptually:

- `emitMixCubeRegionOps(...)`
- `emitMixBoundaryRegionOps(...)`
- `emitMixVectorRegionOps(...)`

Each helper should:

- visit ops in region order
- map supported ops to AscendC emission
- reject unsupported ops with an explicit diagnostic

This means the design is generic even if implementation support is still incremental.

## Important Constraint

The design must not define vector support by a closed whitelist at the architecture level.

It is acceptable for the implementation to initially support only the subset of vector ops already covered by existing emitters, but the framework must be open-ended:

- support is determined by op-dispatch coverage
- unsupported ops fail explicitly
- the architecture does not encode “vector region means only add/mul/max/broadcast/relu”

The same rule applies to cube region emission.

## Boundary Region

The boundary layer remains explicit and first-class:

- selected crossing pairs remain validated
- payload feasibility remains explicit
- synchronization remains explicit

This phase does not generalize boundary strategy beyond the currently supported transfer model, but the boundary region must continue to be emitted as its own layer rather than hidden inside cube/vector helpers.

## Diagnostics

Unsupported generic single-chain shapes should continue to fail during validation.

Supported single-chain shapes with unsupported region-local ops should fail during emission with targeted diagnostics, for example:

- unsupported vector op in single-chain mix emitter
- unsupported cube op in single-chain mix emitter
- unsupported boundary op in single-chain mix emitter

This distinction matters:

- validation failures mean the structure is unsupported
- emission failures mean the structure is acceptable, but the current op emitter coverage is incomplete

## Testing Strategy

### Positive coverage

Retain:

- current mix translator fixtures
- `matmul-add-leakyrelu` end-to-end example

### Negative coverage

Retain:

- `mix-island` structural negative fixture

Add:

- targeted translator negatives that prove unsupported region-local ops fail explicitly at emission time, not by falling into sample-specific fallback behavior

## Phase Boundary

This phase is complete when:

1. single-chain routing still uses `MixPartitionPlan`
2. shell lowering remains stable
3. region bodies are emitted through op-driven dispatch
4. unsupported ops fail explicitly
5. current mix example still passes on xvm

After that, the next phase can expand from single-chain to multi-region linear chains without rewriting the emitter architecture again.
