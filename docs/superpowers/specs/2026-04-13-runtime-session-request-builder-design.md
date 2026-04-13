# Runtime Session Request Builder Design

## Goal

Reduce `tools/runtime-session/runtime_session_main.cpp` to CLI concerns only by moving runtime input assembly into a reusable library helper.

This stage only extracts request-building logic. It does not change CLI flags, output format, profile behavior, or execution semantics.

## Scope

In scope:
- Extract artifact-loading and task-graph-building logic out of `runtime_session_main.cpp`
- Introduce a small library-side builder for:
  - loading an artifact from `--artifact-root`
  - compiling an artifact from `--kernel ...`
  - building a `TaskGraph` from `--run-manifest`
  - building a single-task graph from a single `KernelArtifact`
- Keep `runtime-session` responsible for:
  - parsing CLI flags
  - calling the builder
  - planning/running `ExecutionSession`
  - printing summaries and errors

Out of scope:
- Changing CLI flags or user-visible output
- Refactoring summary-printing helpers out of the CLI
- Changing `ArtifactCompiler`, `RunManifest`, `ExecutionSession`, or backend behavior
- Introducing new CLI tools or APIs

## Problem

`runtime_session_main.cpp` still combines three responsibilities:
1. CLI parsing
2. Runtime object assembly
3. CLI result presentation

The first and third belong in the CLI. The second does not.

The remaining assembly helpers in `runtime_session_main.cpp` are:
- `loadArtifactFromRoot(...)`
- `prepareArtifact()`
- `prepareManifestGraph()`
- `buildGraph(...)`
- supporting manifest/artifact parsing helpers used only for assembly

This makes the CLI file continue to grow and keeps reusable runtime input construction trapped in one executable.

## Design

### New Builder Type

Add a small library helper, tentatively:
- `RuntimeSessionRequestBuilder`

Preferred location:
- `include/Runtime/Artifact/RuntimeSessionRequestBuilder.h`
- `lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp`

This keeps it close to artifact/manfiest assembly instead of execution/output code.

### Builder Responsibilities

The builder provides a library boundary for constructing runtime inputs.

It owns:
- manifest.txt parsing for artifact roots
- artifact path resolution relative to artifact root
- manifest-to-`KernelArtifact` conversion
- compile-request-to-`KernelArtifact` conversion
- run-manifest-to-`TaskGraph` conversion
- single-artifact-to-single-task-graph conversion

It does not own:
- command-line parsing
- execution
- output formatting
- retained profile lifecycle

### Proposed API Surface

Keep the surface small and explicit.

Core request structs:
- `RuntimeArtifactRequest`
  - either `artifactRoot` or compile fields (`kernel`, `kernelKind`, `outputDir`, etc.)
- `RuntimeGraphRequest`
  - either `runManifestPath` or `KernelArtifact + taskId`

Core methods:
- `loadArtifactFromRoot(...) -> Expected<KernelArtifact>`
- `prepareArtifact(...) -> Expected<KernelArtifact>`
- `prepareGraphFromManifest(...) -> Expected<pair<ExecutionBackendKind, TaskGraph>>`
- `buildSingleTaskGraph(...) -> Expected<TaskGraph>`

The builder may also expose a higher-level convenience method later, but not in this stage.

## CLI After Refactor

`runtime_session_main.cpp` should keep:
- `cl::opt` definitions
- summary/error printers
- `main()` orchestration
- retained-profile lifecycle calls

It should stop owning:
- artifact manifest parsing
- artifact request branching logic
- task-graph assembly details

That means current helpers such as `readManifest`, `requireManifestValue`, `resolveArtifactPath`, `locateManifestPath`, `loadArtifactFromRoot`, `prepareArtifact`, `buildGraph`, and `prepareManifestGraph` move out of the CLI file into the builder implementation, except for any truly CLI-only formatting helper.

## Error Semantics

Behavior must remain unchanged.

The builder should preserve existing error messages where practical, especially for:
- missing manifest fields
- unsupported kernel kind
- unsupported mix resource type
- invalid artifact root
- invalid CLI source combinations

`runtime-session` should continue to print errors exactly through its existing error-summary path.

## Testing

Add focused runtime tests for the builder instead of expanding CLI tests.

Required coverage:
- artifact-root manifest -> `KernelArtifact`
- compile request -> `KernelArtifact`
- run manifest -> backend kind + `TaskGraph`
- single artifact -> single task graph
- mix resource type propagation remains intact

Verification remains:
- `bash test/tools/runtime/run_runtime.sh` on xvm

## Migration Plan

One small step:
1. Introduce builder headers/cpp
2. Move request-assembly logic into builder
3. Update `runtime-session` to call builder
4. Keep all CLI-visible behavior unchanged
5. Re-run focused runtime verification on xvm

## Acceptance Criteria

This stage is done when:
- `runtime_session_main.cpp` no longer contains runtime request assembly helpers
- the builder owns artifact and graph assembly logic
- xvm `run_runtime.sh` still passes unchanged
- no CLI flag or output regression is introduced
