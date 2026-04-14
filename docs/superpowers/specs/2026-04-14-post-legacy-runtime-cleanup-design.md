# Post-Legacy Runtime Cleanup Design

## Goal

Now that the major `lib/Runtime/Legacy` implementation units have been deleted, stop treating runtime work as legacy-cleanup work and re-establish the repository as a runtime-native baseline.

This cleanup is not a new feature pass. It is a consolidation pass that removes stale legacy framing from the tree and leaves the next development stage focused on runtime-native evolution.

## Scope

In scope:
- remove empty or obsolete `Legacy/` directory structure where appropriate
- update `AGENTS.md` and cleanup audit language so they no longer frame the runtime as blocked on legacy implementation units
- identify and remove stale references to deleted legacy units from build/config/docs where they still survive
- define the next runtime-native work categories after legacy deletion

Out of scope:
- new scheduler functionality
- new profiling schema changes
- new CLI/API behavior
- deeper refactors to `runtime-session`, `ExecutionSession`, or `ArtifactCompiler`

## Design

### 1. Tree cleanup

If `include/Runtime/Legacy/` and `lib/Runtime/Legacy/` are now empty after deleting the remaining implementation units, remove the empty directories from the tracked tree context and stop treating them as active module locations.

The cleanup should also verify there are no stale build references, source comments, or test labels that still imply the deleted units exist.

### 2. Documentation cleanup

The cleanup audits currently serve as historical records, but the active status documents should no longer read like legacy deletion is the main runtime program.

Update:
- `AGENTS.md`
- the runtime cleanup audit summary documents

So they say clearly:
- major legacy implementation units are gone
- the runtime is now runtime-native by default
- future work should be organized around runtime-native consolidation and feature evolution, not around deleting old subsystems

Historical context can remain in audits, but current TODOs should not continue to center deleted units.

### 3. Next-stage framing

Replace legacy-oriented TODO framing with runtime-native follow-up categories:
- runtime-session / C API boundary cleanup
- execution/profile contract tightening
- task-graph / scheduler evolution
- focused xvm verification maintenance

The result should be that someone opening the repo now sees the next problems as runtime architecture problems, not as leftover legacy migration problems.

## Verification

Minimum verification:
- seam search for deleted legacy unit names in code/build paths
- `git diff --check`
- a lightweight xvm runtime-focused verification pass if code paths move

If the cleanup is documentation-only plus empty-directory removal, verification can stay lightweight as long as no runtime code changes are made.

## Acceptance

This cleanup is complete when:
- no active repo status document presents `Legacy` deletion as the primary remaining runtime goal
- stale references to deleted legacy implementation units are removed from live code/build context
- next-step planning is expressed in runtime-native terms
