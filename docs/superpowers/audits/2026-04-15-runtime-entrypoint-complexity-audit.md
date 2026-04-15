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
| `SimBackend` / `NpuBackend` | branch on mix artifact and call generic dynamic-library runner | acceptable | mix artifact maps to shared object plus launch symbol |
| `ExecutionRunner` | has `runDynamicLibraryArtifact` API; keeps `runPackedMixFile` compatibility wrapper | watch | active seam is generic; old API can be deleted after external compatibility audit |

## Decision

- Keep `RuntimeSessionRequestBuilder` as the boundary for mix metadata to invocation defaults.
- Keep `ArtifactCompiler` as the boundary for mix vs vec/cube compile dispatch.
- Do not move mix staging into CLI, C API, autotuner, or execution summary code.
- Treat `runDynamicLibraryArtifact` as the runtime-native shared-library launch seam.
- Retain `runPackedMixFile` only as a compatibility wrapper until external consumers are audited.

## Next Work

- Audit external consumers of `runPackedMixFile`; delete the compatibility wrapper if none remain.
- Keep timing-stage names internal to diagnostics; callers should not parse them for behavior.
