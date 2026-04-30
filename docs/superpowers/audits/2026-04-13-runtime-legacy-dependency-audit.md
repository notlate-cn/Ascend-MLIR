# Runtime Legacy Dependency Audit

> Historical note: this audit records the pre-deletion dependency picture and
> the transition status of the old `Legacy` units. It should no longer be read
> as an active blocker list for current runtime work.

## Scope

This audit records concrete current references to the legacy runtime units below, based on `rg` inventory plus source inspection:

- `Legacy/Compiler`
- `Legacy/Executor`
- `Legacy/SimValidator`
- `Legacy/HostRunnerGen`
- `Legacy/CompatRuntime`

It covers direct includes, implementation dependencies, and test coverage that kept the legacy surface alive during migration. It does not change behavior or propose deletions.

## Direct References

### Legacy/Compiler

> Update (2026-04-14): `Legacy/Compiler` and its public shim have now been
> removed. This section is retained as historical audit context for the
> pre-deletion state.

| Consumer | File | Dependency Kind | Notes |
|---|---|---|---|
| Public shim | `include/Runtime/Compiler.h` | direct wrapper | Re-exports `Runtime/Legacy/Compiler.h`. |
| Artifact compiler API | `include/Runtime/Artifact/ArtifactCompiler.h` | direct wrapper | `ArtifactCompiler` is still typed against legacy compiler config/ABI. |
| Runtime build wiring | `lib/Runtime/CMakeLists.txt` | build-layer dependency | Still compiles `Legacy/Compiler.cpp` into the runtime library. |
| Artifact compiler implementation | `lib/Runtime/Artifact/ArtifactCompiler.cpp` | implementation dependency | Constructs `CompilerConfig`, instantiates `Compiler`, and calls `Compile()`. |
| Mix compiler frontend | `tools/mix-compiler/mix_compiler_main.cpp` | frontend dependency | Builds a mix `ArtifactCompileRequest` and drives `ArtifactCompiler` directly. |
| Runtime-session builder API | `include/Runtime/Artifact/RuntimeSessionRequestBuilder.h` | adapter dependency | Runtime-session request construction is still built on `ArtifactCompiler`. |
| Runtime-session builder implementation | `lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp` | implementation dependency | Converts runtime-session requests into `ArtifactCompileRequest` and calls `ArtifactCompiler`. |
| Runtime-session frontend | `tools/runtime-session/runtime_session_main.cpp` | CLI dependency | Uses the runtime-session request builder, which still routes compile requests through `ArtifactCompiler`. |
| C API smoke test | `test/tools/runtime/test_capi_runtime.cpp` | direct test consumer | Compiles an example kernel through the C API, which reaches `ArtifactCompiler`. |
| C API runtime shim | `lib/CAPI/Runtime/Runtime.cpp` | adapter dependency | Calls `ArtifactCompiler` after building a compat compile request. |
| Runtime unit test | `test/tools/runtime/test_runtime.cpp` | direct test consumer | Exercises `Compiler::Compile()` for mix artifact generation. |
| Runtime helper test | `test/tools/runtime/test_taskgraph_runtime.cpp` | direct test consumer | Covers `normalizeCompiledArtifact()` and `prepareCompileOutputDir()`, both declared in `Legacy/Compiler.h`. |

### Legacy/Executor

> Update (2026-04-14): `Legacy/Executor` and its public shim have now been
> removed. This section is retained as historical audit context for the
> pre-deletion state.

| Consumer | File | Dependency Kind | Notes |
|---|---|---|---|
| Public shim | `include/Runtime/Executor.h` | direct wrapper | Re-exports `Runtime/Legacy/Executor.h`. |
| Simulator backend | `lib/Runtime/Execution/SimBackend.cpp` | implementation dependency | Uses `Executor::MAGIC_*`, constructs `Executor`, and runs binary or packed mix artifacts. |
| NPU backend | `lib/Runtime/Execution/NpuBackend.cpp` | implementation dependency | Uses the same executor path for real-device execution. |
| Validator shim | `include/Runtime/SimValidator.h` | transitive dependency | `SimValidator` still requires `Executor&` in its legacy API. |
| Runtime unit test | `test/tools/runtime/test_runtime.cpp` | direct test consumer | Verifies `RunPackedMixFile()` error handling through `Executor`. |

### Legacy/SimValidator

| Consumer | File | Dependency Kind | Notes |
|---|---|---|---|
| Public shim | `include/Runtime/SimValidator.h` | direct wrapper | Re-exports `Runtime/Legacy/SimValidator.h`. |
| Simulator backend | `lib/Runtime/Execution/SimBackend.cpp` | implementation dependency | Validates simulator outputs after execution. |
| NPU backend | `lib/Runtime/Execution/NpuBackend.cpp` | implementation dependency | Uses `SimValidator::CompareOnly()` after real-device execution. |

### Legacy/HostRunnerGen

> Update (2026-04-14): `Legacy/HostRunnerGen`, its public shim, and its
> dedicated tests have now been removed. This section is retained as historical
> audit context for the pre-deletion state.

| Consumer | File | Dependency Kind | Notes |
|---|---|---|---|
| Public shim | `include/Runtime/HostRunnerGen.h` | direct wrapper | Re-exports `Runtime/Legacy/HostRunnerGen.h`. |
| Runtime build wiring | `lib/Runtime/CMakeLists.txt` | build-layer dependency | Still compiles `Legacy/HostRunnerGen.cpp` into the runtime library. |
| Runtime unit test | `test/tools/runtime/test_runtime.cpp` | direct test consumer | Exercises generation, compiled runner behavior, and `--bin` enforcement. |
| Runner generator test | `test/tools/runner/test_runner_gen.cpp` | direct test consumer | Generates runners for vec/cube/mix and checks emitted source behavior. |

### Legacy/CompatRuntime

> Update (2026-04-14): `Legacy/CompatRuntime` and its public shim have now
> been removed. This section is retained as historical audit context for the
> pre-deletion state.

| Consumer | File | Dependency Kind | Notes |
|---|---|---|---|
| Public shim | `include/Runtime/CompatRuntime.h` | direct wrapper | Re-exports `Runtime/Legacy/CompatRuntime.h`. |
| C API runtime | `lib/CAPI/Runtime/Runtime.cpp` | adapter dependency | Builds compat compile requests and validator/session inputs through compat helpers. |
| Taskgraph runtime tests | `test/tools/runtime/test_taskgraph_runtime.cpp` | adapter/test dependency | Verifies compat compile-request mapping, compat run-manifest building, and tiling-path preparation. |

## By Consumer

- `include/Runtime/Artifact/ArtifactCompiler.h`, `lib/Runtime/Artifact/ArtifactCompiler.cpp`, `tools/autotuner/autotuner_main.cpp`, and `lib/CAPI/Runtime/Runtime.cpp` no longer depend on `Legacy/Compiler`; any compile-path coupling there is now historical, not current.
- `lib/Runtime/Execution/SimBackend.cpp` and `lib/Runtime/Execution/NpuBackend.cpp` no longer depend on `Legacy/Executor` or `Legacy/SimValidator`; those seams are now historical, not current.
- `test/tools/runtime/test_runtime.cpp` and `test/tools/runtime/test_taskgraph_runtime.cpp` no longer exercise `Legacy/Compiler`-backed helpers directly; `Legacy/Compiler`, `Legacy/Executor`, and `Legacy/HostRunnerGen` are now historical, not current, test dependencies.
- `lib/Runtime/CMakeLists.txt` no longer wires any deleted legacy implementation unit into the runtime library.
- `lib/CAPI/Runtime/Runtime.cpp` no longer routes compile-path compatibility through `Legacy/CompatRuntime`; it now constructs runtime-native requests directly.
- `test/tools/runtime/test_taskgraph_runtime.cpp` no longer depends on `Legacy/CompatRuntime` helper coverage.
- `tools/autotuner/autotuner_main.cpp` no longer includes or directly orchestrates `Compiler`, `Executor`, `SimValidator`, or `HostRunnerGen`; its remaining `Legacy/Compiler` coupling is indirect through `ArtifactCompiler`.

## Bucket Classification

### Bucket A: Must Keep For Now

- none currently

### Bucket B: Candidate For Boundary Shrink

- none currently

### Bucket C: Delete After Migration

- `Legacy/Executor`
  - Migration is complete; the file and shim have now been removed.
- `Legacy/SimValidator`
  - Migration is complete; the file and shim have now been removed.
- `Legacy/HostRunnerGen`
  - Migration is complete; the file, shim, and dedicated tests have now been removed.
- `Legacy/Compiler`
  - Migration is complete; the file, shim, and final retained test coverage have now been removed.
- `Legacy/CompatRuntime`
  - Migration is complete; the file and shim have now been removed.

## Open Questions

- none currently; active planning should now focus on runtime-native boundaries rather than legacy dependency removal.
