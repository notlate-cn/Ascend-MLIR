# Legacy Compiler Deletion Plan

## Objective

Delete `Legacy/Compiler` completely, including its public shims and the final retained legacy mix compile coverage, while keeping focused runtime verification green on xvm.

## Task 1: Remove retained legacy compiler test coverage

Files:
- `test/tools/runtime/test_runtime.cpp`

Steps:
1. Remove the remaining `Compiler` include and any legacy mix compile coverage that still instantiates `Compiler` directly.
2. Keep the rest of `test_runtime.cpp` focused on runtime-native execution/runner behavior.
3. Verify there are no remaining `Compiler compiler(` references in runtime tests.

Verification:
- local grep confirms `test/tools/runtime/test_runtime.cpp` no longer references `Compiler`

## Task 2: Delete legacy compiler code and shims

Files:
- `include/Runtime/Compiler.h`
- `include/Runtime/Legacy/Compiler.h`
- `lib/Runtime/Legacy/Compiler.cpp`
- `lib/Runtime/CMakeLists.txt`

Steps:
1. Delete the top-level shim and legacy header.
2. Delete `lib/Runtime/Legacy/Compiler.cpp`.
3. Remove the file from `lib/Runtime/CMakeLists.txt`.
4. Run a seam search to confirm there are no surviving `Runtime/Compiler.h`, `Runtime/Legacy/Compiler.h`, or `Compiler compiler(` references.

Verification:
- seam grep returns no matches
- `git diff --check`

## Task 3: Sync docs and status

Files:
- `AGENTS.md`
- `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`
- `docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md`
- `docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md`

Steps:
1. Remove retained-compiler wording.
2. Update cleanup ordering to reflect that compiler deletion is complete.
3. Ensure remaining TODOs point at the next surviving legacy cleanup target.

Verification:
- doc language consistently describes `Legacy/Compiler` as deleted, not retained

## Task 4: xvm focused verification

Commands:
- `bash test/tools/runtime/run_runtime.sh`

Fallback if outer xvm build noise blocks completion:
- rerun the direct runtime-focused subset already used in earlier cleanup work
- explicitly record that the blocker is unrelated to compiler deletion

Acceptance:
- `Legacy/Compiler` files and shims are deleted
- runtime tests no longer instantiate `Compiler`
- seam search is clean
- xvm focused runtime verification stays green
