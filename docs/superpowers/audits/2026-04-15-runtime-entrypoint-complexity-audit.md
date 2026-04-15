# Runtime Entrypoint Complexity Audit

## Scope

- Object: `runtime-session`, C API, `autotuner`, and execution backends.
- Semantics: callers should consume runtime artifact contracts, not mix internal stages.
- Result: current visible mix-specific surfaces are classified as acceptable, watch, or follow-up.

## Findings

| Object | Current Surface | Classification | Result |
| --- | --- | --- | --- |
| `runtime-session` | parses `--kernel-kind=mix`; configures sibling `mix-tiling-helper` through `Runtime/ToolDiscovery` | acceptable | helper discovery is runtime support, not local CLI logic |
| C API | selects compiled artifact path; materializes user tiling bytes | acceptable | no mix staging dependency |
| `autotuner` | accepts `kernel-kind=mix`; packs tiling candidates | acceptable | no direct mix backend internals |
| `RuntimeSessionRequestBuilder` | loads mix metadata and applies `blockDim` / `tiling.bin` defaults | acceptable | artifact contract boundary |
| `SimBackend` / `NpuBackend` | branch on mix artifact and call packed mix runner | watch | execution still exposes packed mix launch seam |
| `ExecutionRunner` | has `runPackedMixFile` API | follow-up | mix-specific runner seam remains visible |

## Decision

- Keep `RuntimeSessionRequestBuilder` as the boundary for mix metadata to invocation defaults.
- Keep `ArtifactCompiler` as the boundary for mix vs vec/cube compile dispatch.
- Do not move mix staging into CLI, C API, autotuner, or execution summary code.
- Defer `ExecutionRunner::runPackedMixFile` cleanup until packed mix launch can be represented as a generic artifact launch.

## Next Work

- Design a generic dynamic-library artifact launch interface before changing `ExecutionRunner`.
- Keep timing-stage names internal to diagnostics; callers should not parse them for behavior.
