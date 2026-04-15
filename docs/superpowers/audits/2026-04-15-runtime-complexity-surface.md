# Runtime Complexity Surface

## Scope

- Object: `lib/Runtime` and `include/Runtime`.
- Semantics: runtime is now a platform layer, not a single runner.
- Result: complexity is acceptable only when hidden behind stable runtime-native contracts.

## Complexity Sources

- `Artifact`: compile request, manifest, artifact loading, vec/cube and mix backend dispatch.
- `Mix`: AscendC source analysis, direct-source compile, tiling artifact emission, host stub generation, legacy fallback.
- `Execution`: SIM/NPU runners, task graph, output comparison, profile retention.
- `Frontend`: CLI and C API request assembly through `RuntimeFrontendCore`.

## Current Risk

- Mix-specific behavior can leak upward if callers begin depending on compile stages or fallback internals.
- Timing comparisons can become misleading when frontend compile, artifact compile, and SIM invocation are mixed.
- More runtime features will increase maintenance cost unless each one lands behind an existing contract.

## Recommended Next Work

- Keep documenting contract inputs and outputs before performance changes.
- Keep all mix-specific staging under `Runtime/Mix` and `Runtime/Artifact`.
- Add new behavior through `RuntimeFrontendCore` only when both CLI and C API need it.
- Treat cache as a future optimization after contract boundaries are stable.
