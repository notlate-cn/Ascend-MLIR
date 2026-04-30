# Post-Legacy Runtime Cleanup Plan

## Objective

Consolidate the repository after deletion of the major `Legacy` implementation units so the next development stage is framed around runtime-native architecture rather than legacy migration.

## Task 1: Remove stale tree-level legacy remnants

Files / paths:
- `include/Runtime/Legacy/`
- `lib/Runtime/Legacy/`
- any build/source references that still point at deleted legacy files

Steps:
1. Inspect whether the `Legacy/` directories are now empty.
2. Remove empty directories from the tracked tree context where applicable.
3. Remove any stale build or source references to deleted legacy files that survive only as dead text or comments.

Verification:
- seam search over code/build paths for deleted legacy unit names

## Task 2: Refresh active status documents

Files:
- `AGENTS.md`
- active runtime cleanup audit summary documents

Steps:
1. Remove TODO wording that still frames the work as legacy-deletion-first.
2. Rewrite progress/decision summaries so they state that runtime is now runtime-native by default.
3. Keep historical context in audit documents only where it is still useful as historical context.

Verification:
- document language consistently treats legacy deletion as completed, not ongoing

## Task 3: Reframe next-step priorities

Files:
- `AGENTS.md`
- any active cleanup summary doc that still defines next work in legacy terms

Steps:
1. Replace legacy-oriented next steps with runtime-native follow-up buckets:
   - runtime-session / C API boundary cleanup
   - execution/profile contract tightening
   - task-graph / scheduler evolution
   - xvm verification maintenance
2. Keep the next-step list short and concrete.

Verification:
- current-status docs express next work without relying on deleted legacy units as motivators

## Task 4: Final hygiene verification

Commands:
- `git diff --check`
- seam searches used in Tasks 1-3

Optional xvm verification:
- only if any runtime code/build path is changed beyond docs/tree cleanup

Acceptance:
- no active status doc still centers legacy deletion as the primary runtime goal
- stale references to deleted legacy implementation units are removed from live code/build context
- next-step planning is written in runtime-native terms
