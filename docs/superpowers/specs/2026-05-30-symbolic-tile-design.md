# Symbolic Tile Design

## Problem

The current Schedule implementation materializes a concrete
`ascend.schedule.selected_tile_shape` and downstream lowering uses that value as
the loop step. This turns a schedule candidate into a compile-time constant and
prevents autotuning and host tiling from selecting tile sizes for dynamic shapes.

The design target is a symbolic tile contract:

- Schedule describes tile parameters and legality constraints.
- Autotuner and host tiling choose runtime tile values within those constraints.
- Kernel lowering consumes tile parameters, not static selected tile values.
- Shape buckets are not required for the first implementation slice.

## Contract

`ascend.schedule.tile_params` is the primary tile contract. It is an array of
dictionary attributes, one entry per scheduled logical tile dimension:

```mlir
ascend.schedule.tile_params = [
  {
    name = "TB_M",
    axis = 0 : i64,
    axis_kind = "parallel",
    binding = "runtime",
    default = 70 : i64,
    upper_bound = 70 : i64,
    extent = 70 : i64,
    roles = ["kernel_loop", "vectorize"],
    primitive_uses = ["data_copy", "vector_compute", "write_back"]
  }
]
```

Field meanings:

- `name`: host tiling field name. It must match the generated `TilingData`
  field used by the kernel.
- `axis`: logical schedule axis index.
- `axis_kind`: `parallel`, `reduction`, or `unknown`.
- `binding`:
  - `runtime`: host tiling provides the tile value at launch time.
  - `extent`: the tile is the runtime extent of that axis, not an autotuned
    split variable.
  - `static_fallback`: compatibility fallback for legacy kernels that cannot
    consume runtime tile fields yet.
- `default`: fallback tile value used by host tiling when no tuning record
  applies.
- `upper_bound`: resource legality bound. Autotuner search must not produce a
  tile above this value unless a later resource model proves a larger value
  legal.
- `extent`: static extent if known, `?` encoded as MLIR dynamic sentinel
  otherwise.
- `roles`: axis execution roles from Schedule.
- `primitive_uses`: primitive uses that constrain legality and tail policy.

`ascend.schedule.tile_binding = "symbolic"` marks that `tile_params` is the
authoritative lowering contract. `selected_tile_shape` can still be emitted for
legacy compatibility and report readability, but new lowering must not require
it.

## Role Coverage

The contract is role-generic. It is derived from `ScheduleProblem` and
`AxisScheduleConstraint`, not from operation names.

- Vector and injective/all-parallel kernels use runtime parallel-axis tile
  params.
- Reduction kernels use runtime params for parallel or chunked reduction axes;
  full reduction axes are `extent` bindings.
- Cube kernels expose M/N/K logical axis tile params. Backends may delegate the
  actual values to the CANN matmul tiling API, but the schedule contract remains
  explicit.
- Memory kernels expose full-axis or runtime copy tile params according to their
  axis constraints.
- Layout-transform kernels use the same logical-axis fields; unsupported maps
  must fail closed rather than invent static tiles.

## Host Tiling

Host tiling receives `tile_params` through artifact metadata. For the first
slice, shape buckets are deliberately omitted:

```text
exact runtime shape + tuning DB hit -> tuned tile value
otherwise -> default value clamped by extent and upper_bound
```

The generated host tiling helper must write each runtime tile field into
`TilingData`. `GetBlockDim` and workspace sizing may use the same tile fields
when needed.

## Lowering Rules

When `tile_binding = "symbolic"`:

- Structured lowering must preserve `tile_params`.
- Compute lowering must materialize loop steps from the corresponding tiling
  fields.
- Tail lowering must use `min(tile, extent - offset)` for runtime tile values.
- A pass may use `upper_bound` to size static resources, but it must not use
  `upper_bound` as the loop step.

Legacy rule:

- `selected_tile_shape` remains accepted for existing static tests and kernels.
- New dynamic tests must assert that `tile_params` exists and is exported into
  runtime artifact tiling params.

## Non-Goals

- Shape bucket design.
- Full autotuner cost-history schema.
- Replacing CANN matmul API tiling internals.
- Removing all legacy `selected_tile_shape` tests in one patch.

## Acceptance Criteria

- Schedule emits `tile_params` for vector, reduction, cube, and memory schedule
  families.
- Runtime artifact manifest includes `tile_params`.
- Host tiling C++ initializes symbolic tile fields from defaults when no tuning
  record is present.
- Compute selected-tile lowering can consume runtime tile params for supported
  rank-2 all-parallel and reduction kernels.
- Existing static tests keep passing through the legacy fallback path.
