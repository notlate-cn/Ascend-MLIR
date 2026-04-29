# Runtime Legacy Cleanup Reclassification Design

## Goal

Refresh the cleanup classification in:

- `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`

so it reflects the current runtime architecture after these two seam removals:

- `ArtifactCompiler -> Legacy/Compiler` for vec/cube builds
- `SimBackend/NpuBackend -> Legacy/Executor` direct backend dependency

## Motivation

The current cleanup-candidate audit was written before the latest runtime cutovers. It still encodes now-stale assumptions about what is on the critical path for second-round `Legacy` cleanup.

Specifically:

- `Legacy/Compiler` is no longer required by `ArtifactCompiler`
- `Legacy/Executor` is no longer directly referenced by `SimBackend` / `NpuBackend`
- `Legacy/SimValidator` was already removed from direct runtime backend usage in the prior comparator extraction round

That means the cleanup candidate buckets now need to be reclassified before any further deletion or shrink work is planned.

## Non-Goals

- Do not produce a new dependency audit from scratch.
- Do not change runtime behavior.
- Do not delete any legacy code in this round.
- Do not decide the final `Legacy/Compiler` extraction strategy yet.

## Scope

This round only updates the cleanup candidate document so it answers, with current source evidence:

- what is now safe to consider for cleanup planning
- what still needs migration first
- what still must not be touched yet
- what verification gates apply to each category

## Reclassification Principles

### Source of truth

Use these as the current evidence base:

- `docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md`
- `docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md`
- current source state in `lib/Runtime/Artifact`, `lib/Runtime/Execution`, and `AGENTS.md`

### Bucket meanings

The refreshed buckets should be interpreted as:

- `Safe Now`
  - items whose previous direct runtime-critical blockers are gone
  - safe here means “eligible for focused cleanup planning next,” not “delete immediately”

- `Needs Migration First`
  - items with real remaining consumers or adapter boundaries that still need one more explicit cutover

- `Do Not Touch Yet`
  - items with dedicated test/workflow value and no replacement path yet

## Expected Reclassification Direction

### Legacy/Compiler

This should no longer be classified as blocked by `ArtifactCompiler`.

Its remaining status should be driven by:

- any non-`ArtifactCompiler` consumers still found in source
- whether those consumers justify one more extraction pass or a much smaller retained legacy surface

### Legacy/Executor

This should no longer be classified as a direct runtime-backend blocker.

Its new status should explicitly reflect:

- it now sits behind `DefaultExecutionRunner`
- further cleanup work is still possible, but must start from the adapter seam rather than the backends

### Legacy/SimValidator

This should be reclassified based on the fact that the direct backend seam is already gone.

Its remaining status should be decided by any other surviving consumers, not by outdated backend references.

### Legacy/CompatRuntime

This likely remains in `Needs Migration First`, because it still marks a compatibility boundary for C API and taskgraph tests.

### Legacy/HostRunnerGen

This likely remains in `Do Not Touch Yet`, unless current source evidence shows it has lost its dedicated consumer/testing value.

## Deliverable

Update:

- `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`

so that:

- bucket membership reflects the current post-cutover source state
- each bucket entry cites the current reason it is there
- verification requirements are updated to match the new seams

## Acceptance Criteria

- the cleanup candidate audit no longer claims:
  - `ArtifactCompiler` directly depends on `Legacy/Compiler`
  - `SimBackend` / `NpuBackend` directly depend on `Legacy/Executor`
- the audit explicitly reflects the new adapter boundaries:
  - `VecCubeArtifactBackend`
  - `DefaultExecutionRunner`
  - runtime-native comparator path already removing direct `Legacy/SimValidator` backend use
- the resulting buckets give a coherent next-step basis for second-round `Legacy` cleanup planning

