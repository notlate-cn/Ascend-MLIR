# Runtime Legacy Dependency Audit

## Scope

This audit records concrete current references to the legacy runtime units below, based on `rg` inventory plus source inspection:

- `Legacy/Compiler`
- `Legacy/Executor`
- `Legacy/SimValidator`
- `Legacy/HostRunnerGen`
- `Legacy/CompatRuntime`

It covers direct includes, implementation dependencies, and test coverage that still keeps the legacy surface alive. It does not change behavior or propose deletions.

## Direct References

### Legacy/Compiler

| Consumer | File | Dependency Kind | Notes |
|---|---|---|---|
| Public shim | `include/Runtime/Compiler.h` | direct wrapper | Re-exports `Runtime/Legacy/Compiler.h`. |
| Artifact compiler API | `include/Runtime/Artifact/ArtifactCompiler.h` | direct wrapper | `ArtifactCompiler` is still typed against legacy compiler config/ABI. |
| Artifact compiler implementation | `lib/Runtime/Artifact/ArtifactCompiler.cpp` | implementation dependency | Constructs `CompilerConfig`, instantiates `Compiler`, and calls `Compile()`. |
| Runtime unit test | `test/tools/runtime/test_runtime.cpp` | direct test consumer | Exercises `Compiler::Compile()` for mix artifact generation. |
| Runtime helper test | `test/tools/runtime/test_taskgraph_runtime.cpp` | direct test consumer | Covers `normalizeCompiledArtifact()` and `prepareCompileOutputDir()`, both declared in `Legacy/Compiler.h`. |

### Legacy/Executor

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

| Consumer | File | Dependency Kind | Notes |
|---|---|---|---|
| Public shim | `include/Runtime/HostRunnerGen.h` | direct wrapper | Re-exports `Runtime/Legacy/HostRunnerGen.h`. |
| Runtime build wiring | `lib/Runtime/CMakeLists.txt` | build-layer dependency | Still compiles `Legacy/HostRunnerGen.cpp` into the runtime library. |
| Runtime unit test | `test/tools/runtime/test_runtime.cpp` | direct test consumer | Exercises generation, compiled runner behavior, and `--bin` enforcement. |
| Runner generator test | `test/tools/runner/test_runner_gen.cpp` | direct test consumer | Generates runners for vec/cube/mix and checks emitted source behavior. |

### Legacy/CompatRuntime

| Consumer | File | Dependency Kind | Notes |
|---|---|---|---|
| Public shim | `include/Runtime/CompatRuntime.h` | direct wrapper | Re-exports `Runtime/Legacy/CompatRuntime.h`. |
| C API runtime | `lib/CAPI/Runtime/Runtime.cpp` | adapter dependency | Builds compat compile requests and validator/session inputs through compat helpers. |
| Taskgraph runtime tests | `test/tools/runtime/test_taskgraph_runtime.cpp` | adapter/test dependency | Verifies compat compile-request mapping, compat run-manifest building, and tiling-path preparation. |

## By Consumer

- `include/Runtime/Artifact/ArtifactCompiler.h`, `lib/Runtime/Artifact/ArtifactCompiler.cpp`, `tools/autotuner/autotuner_main.cpp`, and `lib/CAPI/Runtime/Runtime.cpp` all depend on `Legacy/Compiler` indirectly through `ArtifactCompiler`.
- `lib/Runtime/Execution/SimBackend.cpp` and `lib/Runtime/Execution/NpuBackend.cpp` depend on `Legacy/Executor` directly for binary registration, launch, and magic selection.
- `lib/Runtime/Execution/SimBackend.cpp` and `lib/Runtime/Execution/NpuBackend.cpp` depend on `Legacy/SimValidator` directly for output comparison after execution.
- `test/tools/runtime/test_runtime.cpp` and `test/tools/runtime/test_taskgraph_runtime.cpp` still exercise `Legacy/Compiler` directly, while `test/tools/runtime/test_runtime.cpp` remains the runtime unit test file that directly exercises `Legacy/Executor` and `Legacy/HostRunnerGen`.
- `test/tools/runner/test_runner_gen.cpp` is dedicated `Legacy/HostRunnerGen` coverage and does not touch the newer runtime session stack.
- `lib/Runtime/CMakeLists.txt` still wires all five `Legacy/*` implementation units into the runtime library, including `Legacy/HostRunnerGen.cpp`.
- `lib/CAPI/Runtime/Runtime.cpp` still routes compile-path compatibility through `Legacy/CompatRuntime` helpers before calling `ArtifactCompiler`.
- `test/tools/runtime/test_taskgraph_runtime.cpp` is the main direct consumer of `Legacy/CompatRuntime` helpers and also checks that the compat layer maps into runtime-native request objects correctly.
- `tools/autotuner/autotuner_main.cpp` no longer includes or directly orchestrates `Compiler`, `Executor`, `SimValidator`, or `HostRunnerGen`; its remaining `Legacy/Compiler` coupling is indirect through `ArtifactCompiler`.

## Bucket Classification

### Bucket A: Must Keep For Now

- `Legacy/Compiler`
  - Required by `include/Runtime/Artifact/ArtifactCompiler.h` and `lib/Runtime/Artifact/ArtifactCompiler.cpp`.
  - Still exercised directly by `test/tools/runtime/test_runtime.cpp` and `test/tools/runtime/test_taskgraph_runtime.cpp`.
- `Legacy/Executor`
  - Required by `lib/Runtime/Execution/SimBackend.cpp` and `lib/Runtime/Execution/NpuBackend.cpp`.
  - Still exposed through `include/Runtime/SimValidator.h`.
- `Legacy/SimValidator`
  - Required by `lib/Runtime/Execution/SimBackend.cpp` and `lib/Runtime/Execution/NpuBackend.cpp`.

### Bucket B: Candidate For Boundary Shrink

- `Legacy/HostRunnerGen`
  - Referenced by `include/Runtime/HostRunnerGen.h`, `lib/Runtime/CMakeLists.txt`, `test/tools/runtime/test_runtime.cpp`, and `test/tools/runner/test_runner_gen.cpp`.
  - No runtime-native execution path currently depends on it.
- `Legacy/CompatRuntime`
  - Still used by `lib/CAPI/Runtime/Runtime.cpp` and `test/tools/runtime/test_taskgraph_runtime.cpp`.
  - Already sits at the adapter boundary, so it is a shrink candidate before deeper legacy deletion.

### Bucket C: Delete After Migration

- none yet

## Open Questions

- Can `tools/autotuner/autotuner_main.cpp` stay fully on `ArtifactCompiler` and `ExecutionSession`, or will it need a new runtime-native builder once `Legacy/Compiler` is removed?
- Should `lib/CAPI/Runtime/Runtime.cpp` keep using `buildCompatCompileRequest()` and `buildCompatSingleTaskRunManifest()`, or should those compatibility helpers be replaced with direct runtime-native construction?
- Do `test/tools/runtime/test_runtime.cpp` and `test/tools/runner/test_runner_gen.cpp` need to remain as explicit `Legacy/HostRunnerGen` coverage, or should they move to a runtime-native runner path first?
- Is `include/Runtime/SimValidator.h` intended to remain a public shim after `Legacy/Executor` is no longer part of the validation API?
