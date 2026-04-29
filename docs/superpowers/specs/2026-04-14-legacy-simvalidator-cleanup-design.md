# Legacy SimValidator Cleanup Design

> Date: 2026-04-14
> Status: Proposed

## Goal

Shrink `Legacy/SimValidator` after the runtime backends have already moved to the runtime-native output comparator, so that only truly unavoidable residual validation helpers remain.

## Non-Goals

- No `Executor` rewrite in this slice.
- No `SimBackend` / `NpuBackend` behavior change.
- No autotuner redesign.
- No broad deletion of `Legacy` files without a residual-consumer check.

## Current Problem

The main runtime execution path no longer depends directly on `Legacy/SimValidator`:

- `SimBackend` uses the runtime-native output comparator.
- `NpuBackend` uses the runtime-native output comparator.

But `Legacy/SimValidator` still exists as a retained implementation unit with:

- `CompareOnly(...)`
- `ValidateBinary(...)`
- simulator cycle-count parsing and diff helpers

That means the file is no longer a main-path blocker, but it still looks larger than its true remaining role.

## Recommended Approach

Treat this as a residual-surface cleanup, not a full validator rewrite.

The implementation should answer two questions:

1. Which parts of `Legacy/SimValidator` still have real consumers?
2. Which parts can move into smaller runtime-native helpers or be dropped as dead surface?

The expected outcome is either:

- a much smaller retained compatibility/helper unit, or
- a narrow extraction plan for the last useful pieces

but not an all-at-once deletion.

## Residual Surface

`Legacy/SimValidator` currently contains two categories of logic:

### 1. Output comparison

This category has already been migrated in the main runtime path.

The runtime-native output comparator now owns:

- mismatch detection
- structural validation semantics
- backend-facing validation error propagation

So `CompareOnly(...)` should no longer be treated as a required runtime-backend seam.

### 2. Simulator cycle-count and retained validation helpers

The remaining useful pieces are likely the small utilities around:

- parsing simulator summary logs
- packaging diff/cycle results for residual legacy validation paths

These may still justify one more extraction if there are non-backend consumers that actually need them.

## Recommended Cleanup Order

### Step 1: Audit residual consumers

Re-run source inventory for:

- `Runtime/SimValidator.h`
- `SimValidator`
- `CompareOnly(`
- `ValidateBinary(`

Then classify each remaining consumer as:

- explicit legacy compatibility
- removable test residue
- future helper extraction target

### Step 2: Remove dead comparison-facing surface

If any comparison-facing methods are no longer consumed outside the file itself, remove or narrow them instead of keeping them alive as generic validator API.

### Step 3: Decide whether cycle parsing deserves extraction

If simulator cycle parsing is still useful outside retained legacy code, extract that logic into a smaller runtime-native helper.

If not, keep it local to the retained legacy file.

## Alternatives Considered

### 1. Delete `Legacy/SimValidator` immediately

Rejected.

The main seam is already gone, but immediate deletion risks conflating dead comparison surface with any still-useful residual helper logic.

### 2. Leave `Legacy/SimValidator` untouched

Rejected.

Now that the main runtime seam is gone, this is the right time to shrink the remaining surface and make the residual role explicit.

## Testing And Verification

The implementation plan should require:

1. source inventory after the shrink:
   - `rg -n "Runtime/SimValidator.h|\\bSimValidator\\b|CompareOnly\\(|ValidateBinary\\(" include lib tools test --glob '!build*' --glob '!externals/**'`
2. focused runtime verification on xvm:
   - `bash test/tools/runtime/run_runtime.sh`
3. if any non-backend consumer is migrated, verify its replacement path explicitly

## Acceptance Criteria

This design is complete when:

- the residual `Legacy/SimValidator` consumer set is explicit
- dead comparison-facing surface is removed or clearly isolated
- any surviving helper logic has a defensible retained role
- xvm focused runtime verification remains green

## Scope Guardrails

This slice must not expand into:

- `Legacy/Executor` deletion
- `CompatRuntime` redesign
- profiling schema changes
- autotuner feature work
