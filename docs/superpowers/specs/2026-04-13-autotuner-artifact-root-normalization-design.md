# Autotuner Artifact Root Normalization Design

**Goal:** Make `autotuner --artifact-root` use the same canonical runtime artifact loading semantics as `runtime-session`, especially for mix artifact metadata such as `mix_resource_type`.

**Scope:** This spec covers only autotuner artifact-root loading. It does not redesign `ArtifactCompiler`, `SimBackend`, `NpuBackend`, or the broader `Legacy` cleanup sequence.

## Problem

`tools/autotuner/autotuner_main.cpp` still has a local artifact-root manifest loader. That local path does not preserve the same metadata semantics as the runtime’s canonical artifact loader.

In particular:
- the autotuner-local loader hardcodes `artifact.mixResourceType = MixResourceType::Unknown`
- the runtime’s canonical loader in `RuntimeSessionRequestBuilder.cpp` parses `mix_resource_type` and applies the current runtime conventions for artifact manifests
- `ExecutionSession::canScheduleTask(...)` rejects mix tasks when `mixResourceType` is `Unknown`

This means `autotuner --artifact-root` can misclassify valid mix artifacts even though the rest of the runtime stack already knows how to load them correctly.

## Desired Outcome

When `autotuner` is given `--artifact-root`:
- it should load artifacts through the same canonical runtime artifact-loading logic used by `runtime-session`
- vec/cube artifact behavior should remain unchanged
- mix artifacts should preserve `mix_resource_type`
- no separate autotuner-specific artifact manifest parsing logic should remain for that path

## Design Choice

There are three possible approaches:

1. Replace the local autotuner loader with the canonical runtime loader. Recommended.
2. Keep the local loader and manually port more runtime manifest rules into it.
3. Introduce a second shared helper just for autotuner.

Option 1 is the right choice because:
- the canonical loader already exists
- the runtime-session path already relies on it
- duplicating artifact manifest semantics again would recreate the same drift we just audited

## Proposed Change

`autotuner_main.cpp` should stop using its local `locateManifestPath(...)` / `loadArtifactFromRoot(...)` path when `--artifact-root` is provided.

Instead, it should call the shared runtime API:
- `loadRuntimeSessionArtifactFromRoot(...)`

That function already provides:
- supported manifest discovery
- required-field validation
- `kernel_kind` parsing
- `mix_resource_type` parsing
- artifact-root-relative path resolution

The autotuner should continue owning:
- `--kernel` source-build behavior via `ArtifactCompiler`
- tiling-space loading
- candidate graph construction
- profiling extraction and output JSON

Only the artifact-root loading path is normalized in this change.

## Behavioral Constraints

This change must not:
- alter the autotuner CLI surface
- change output JSON schema
- change profiling extraction logic
- change vec/cube source-build behavior
- broaden into `ArtifactCompiler` refactoring

It should only remove local artifact-root manifest semantics drift.

## Testing Impact

The implementation should add focused coverage proving:
- a mix artifact-root loaded through autotuner preserves `MixResourceType::Mix1C1V` (or the manifest-declared value)
- the normalized path no longer produces `MixResourceType::Unknown` for valid mix manifests
- existing artifact-root loading behavior for vec artifacts still works

Because this is an autotuner runtime-path change, verification should include:
- focused unit coverage for the loader path
- xvm autotuner smoke for at least one vec path
- if practical, a focused mix artifact-root path check to prove the metadata no longer regresses before scheduling

## Non-Goals

This spec does not:
- remove `Legacy/Compiler`
- change `ArtifactCompiler`
- change `ExecutionSession`
- change `SimBackend` or `NpuBackend`
- redesign autotuner result scoring

Those remain separate follow-up tracks.
