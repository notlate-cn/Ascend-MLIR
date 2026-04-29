# Legacy Compiler Deletion Design

## Goal

Delete `Legacy/Compiler` completely now that:
- `ArtifactCompiler` no longer depends on it for `vec`/`cube`
- runtime execution no longer depends on the old compile/runner stack
- the only remaining in-repo usage is retained legacy test coverage

This deletion removes the last retained `Legacy/Compiler` implementation unit and its public shims instead of preserving it as a compatibility surface.

## Scope

Delete:
- `include/Runtime/Compiler.h`
- `include/Runtime/Legacy/Compiler.h`
- `lib/Runtime/Legacy/Compiler.cpp`
- the remaining `Compiler`-based legacy mix compile coverage in `test/tools/runtime/test_runtime.cpp`

Update:
- `lib/Runtime/CMakeLists.txt`
- `AGENTS.md`
- runtime legacy cleanup audit documents

Out of scope:
- introducing any new compat shim
- rewriting deleted legacy tests into new runtime-native compile tests
- changes to `MixDirectBackend`, `VecCubeArtifactBackend`, `ArtifactCompiler`, `runtime-session`, or `autotuner` behavior

## Design

### Deletion policy

This is a hard deletion, not a boundary shrink.

The codebase will no longer expose:
- `Runtime/Compiler.h`
- `Runtime/Legacy/Compiler.h`
- `mlir::runtime::Compiler`

Any surviving include or call site should fail the build and be fixed as part of this work.

### Test handling

The remaining `Compiler` coverage in `test/tools/runtime/test_runtime.cpp` exists only to preserve retained legacy behavior. It should be removed outright.

No replacement test is required in this change because:
- runtime-native compile coverage already exists through `ArtifactCompiler`, `VecCubeArtifactBackend`, and focused runtime verification
- retaining a dedicated legacy compile test would contradict the deletion goal

### Documentation updates

`AGENTS.md` and the cleanup audits should be updated to reflect that:
- `Legacy/Compiler` is no longer a retained compatibility unit
- the runtime library no longer contains legacy compiler implementation code
- remaining legacy cleanup should move on to the next surviving unit rather than treating compiler cleanup as pending

## Verification

Minimum required verification:
- code search confirms no remaining `Runtime/Compiler.h`, `Runtime/Legacy/Compiler.h`, or `Compiler compiler(` references
- `git diff --check`
- xvm focused runtime verification via `bash test/tools/runtime/run_runtime.sh`

If full xvm verification is blocked by unrelated outer-tool build issues, use the same fallback standard already established in prior cleanup work:
- direct runtime-focused xvm verification for the affected test binaries/scripts
- explicit note that the blocker is external to this deletion

## Acceptance

The change is complete when:
- all `Legacy/Compiler` code and public shims are deleted
- the last retained legacy test coverage is removed
- runtime-focused xvm verification remains green
- audits and `AGENTS.md` no longer describe `Legacy/Compiler` as retained
