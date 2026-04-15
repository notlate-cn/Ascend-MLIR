# Runtime Dynamic Library Artifact Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make mix execution consume a complete runtime-native dynamic-library artifact contract: shared library path, launch symbol, and simulator environment are resolved before backend execution.

**Architecture:** `Artifact` / `Mix` owns compatibility with legacy manifest and metadata fields. `Execution` consumes generic artifact fields only and no longer derives mix launch details. Verification locks this boundary in `run_runtime.sh`.

**Tech Stack:** C++17, LLVM `Expected` / `Error`, shell runtime verification, xvm as authoritative test environment.

---

## Current State

- `KernelArtifact` -> has `sharedLibraryPath`, but no launch symbol field.
- `SimBackend` / `NpuBackend` -> still derive `aclrtlaunch_ + kernelName`.
- `SimBackend` -> still owns `configurePackedMixEnvironment`.
- Mix metadata -> already records `launcher_symbol`.
- Runtime verification -> already forbids `runPackedMixFile`, `PackedMixExecutionLaunch`, and `packedSharedObjectPath` in Execution/C API surfaces.

## Target State

- `KernelArtifact` -> carries `sharedLibraryPath` and `sharedLibrarySymbol`.
- `Artifact` / `Mix` -> fills `sharedLibrarySymbol` from metadata `launcher_symbol`, with fallback to `aclrtlaunch_ + kernelName` only inside artifact normalization.
- `SimBackend` / `NpuBackend` -> pass `artifact.sharedLibrarySymbol` into `DynamicLibraryExecutionLaunch`.
- `SimBackend` -> calls a support helper for dynamic-library simulator environment setup.
- `run_runtime.sh` -> rejects `aclrtlaunch_` construction in `lib/Runtime/Execution`.

## Task 1: Promote Launch Symbol Into KernelArtifact

**Files:**
- Modify: `include/Runtime/Execution/TaskGraph.h`
- Modify: `lib/Runtime/Mix/MixDirectBackend.cpp`
- Modify: `lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] Add field:

```cpp
struct KernelArtifact {
  std::string kernelName;
  KernelKind kernelKind = KernelKind::Vec;
  MixResourceType mixResourceType = MixResourceType::Unknown;
  std::string socVersion;
  std::string artifactRoot;
  std::string deviceBinaryPath;
  std::string sharedLibraryPath;
  std::string sharedLibrarySymbol;
  std::string manifestPath;
  std::string metadataPath;
};
```

- [ ] Test mix normalization:

```cpp
EXPECT(normalizedMix.sharedLibrarySymbol == "aclrtlaunch_demo_kernel",
       "normalized mix artifact stores shared library symbol");
```

- [ ] Implement in `normalizeMixArtifact`:

```cpp
normalized.sharedLibrarySymbol =
    artifact.kernel_name.empty() ? "" : "aclrtlaunch_" + artifact.kernel_name;
```

- [ ] Run RED/GREEN on xvm:

```bash
bash test/tools/runtime/run_runtime.sh
```

Expected after implementation: `test_taskgraph_runtime`, `test_capi_runtime`, `test_runtime`, SimBackend smoke, repeated mix baseline pass.

- [ ] Commit:

```bash
git add include/Runtime/Execution/TaskGraph.h lib/Runtime/Mix/MixDirectBackend.cpp test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: carry dynamic library launch symbol"
```

## Task 2: Load Launch Symbol From Artifact Metadata

**Files:**
- Modify: `lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] Add test: loading a mix artifact root with metadata `launcher_symbol` fills `KernelArtifact::sharedLibrarySymbol`.

Expected assertion:

```cpp
EXPECT(artifactOr->sharedLibrarySymbol == "aclrtlaunch_fake_kernel",
       "runtime artifact loader keeps mix metadata launcher symbol");
```

- [ ] Implement after metadata path validation:

```cpp
if (artifact.kernelKind == KernelKind::Mix && !artifact.metadataPath.empty()) {
  auto metadataOr = loadMixCompileMetadata(artifact.metadataPath);
  if (!metadataOr)
    return metadataOr.takeError();
  artifact.sharedLibrarySymbol = metadataOr->launcherSymbol;
}
if (artifact.kernelKind == KernelKind::Mix &&
    artifact.sharedLibrarySymbol.empty()) {
  artifact.sharedLibrarySymbol = "aclrtlaunch_" + artifact.kernelName;
}
```

- [ ] Run:

```bash
bash test/tools/runtime/run_runtime.sh
```

- [ ] Commit:

```bash
git add lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: load dynamic library launch symbol from metadata"
```

## Task 3: Stop Execution From Deriving Mix Launch Symbols

**Files:**
- Modify: `lib/Runtime/Execution/SimBackend.cpp`
- Modify: `lib/Runtime/Execution/NpuBackend.cpp`
- Modify: `test/tools/runtime/run_runtime.sh`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] Add verification guard:

```bash
if grep -R -n "aclrtlaunch_" lib/Runtime/Execution; then
  echo "Error: execution layer must consume artifact-provided launch symbols" >&2
  exit 1
fi
```

- [ ] Update backend validation:

```cpp
if (request.task.artifact.sharedLibrarySymbol.empty()) {
  return stageError("artifact",
                    "mix artifact is missing shared library symbol");
}
```

- [ ] Update launch assembly:

```cpp
launch.symbolName = request.task.artifact.sharedLibrarySymbol;
```

- [ ] Update missing-symbol tests to expect:

```cpp
"mix artifact is missing shared library symbol"
```

- [ ] Run RED before implementation and confirm the guard fails on current `aclrtlaunch_` construction.

- [ ] Run GREEN:

```bash
bash test/tools/runtime/run_runtime.sh
```

- [ ] Commit:

```bash
git add lib/Runtime/Execution/SimBackend.cpp lib/Runtime/Execution/NpuBackend.cpp test/tools/runtime/run_runtime.sh test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: use artifact launch symbol in execution"
```

## Task 4: Move Simulator Dynamic-Library Environment Setup Out Of SimBackend

**Files:**
- Create: `include/Runtime/Execution/DynamicLibraryArtifactEnv.h`
- Create: `lib/Runtime/Execution/DynamicLibraryArtifactEnv.cpp`
- Modify: `lib/Runtime/CMakeLists.txt`
- Modify: `lib/Runtime/Execution/SimBackend.cpp`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] Add interface:

```cpp
#pragma once

#include "Runtime/Execution/TaskGraph.h"
#include "llvm/Support/Error.h"

namespace mlir::runtime {

llvm::Error configureDynamicLibraryArtifactSimulationEnv(
    const KernelArtifact &artifact);

} // namespace mlir::runtime
```

- [ ] Move current `configurePackedMixEnvironment` logic into the new `.cpp`, rename diagnostics from packed-mix wording to dynamic-library artifact wording.

- [ ] Replace SimBackend call:

```cpp
if (auto err = configureDynamicLibraryArtifactSimulationEnv(
        request.task.artifact))
  return stageError("artifact", std::move(err));
```

- [ ] Add verification guard:

```bash
if grep -R -n "configurePackedMixEnvironment" lib/Runtime include/Runtime; then
  echo "Error: packed mix simulator env helper must not remain" >&2
  exit 1
fi
```

- [ ] Run:

```bash
bash test/tools/runtime/run_runtime.sh
```

- [ ] Commit:

```bash
git add include/Runtime/Execution/DynamicLibraryArtifactEnv.h lib/Runtime/Execution/DynamicLibraryArtifactEnv.cpp lib/Runtime/CMakeLists.txt lib/Runtime/Execution/SimBackend.cpp test/tools/runtime/run_runtime.sh test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: centralize dynamic library simulation environment"
```

## Task 5: Close Documentation And Timing Baseline

**Files:**
- Modify: `docs/superpowers/audits/2026-04-15-runtime-entrypoint-complexity-audit.md`
- Run only: `test/tools/runtime/run_mix_compile_timing_compare.sh`

- [ ] Update audit:

```markdown
| `SimBackend` / `NpuBackend` | consume `sharedLibraryPath` and `sharedLibrarySymbol` | acceptable | execution does not derive mix launch details |
```

- [ ] Run xvm default verification:

```bash
bash test/tools/runtime/run_runtime.sh
```

- [ ] Run xvm timing smoke:

```bash
RUNS=3 bash test/tools/runtime/run_mix_compile_timing_compare.sh
```

- [ ] Commit:

```bash
git add docs/superpowers/audits/2026-04-15-runtime-entrypoint-complexity-audit.md
git commit -m "docs: close dynamic library artifact contract audit"
```

## Final Acceptance

- `bash test/tools/runtime/run_runtime.sh` -> pass on xvm.
- `RUNS=3 bash test/tools/runtime/run_mix_compile_timing_compare.sh` -> reports direct-source and legacy-preprocess timing.
- `rg -n "runPackedMixFile|PackedMixExecutionLaunch" include/Runtime lib/Runtime test/tools/runtime/test_*.cpp` -> no matches.
- `rg -n "packedSharedObjectPath" include/Runtime/Execution lib/Runtime/Execution lib/CAPI/Runtime` -> no matches.
- `rg -n "aclrtlaunch_" lib/Runtime/Execution` -> no matches.
- `rg -n "configurePackedMixEnvironment" include/Runtime lib/Runtime` -> no matches.

## Execution Choice

Recommended: execute inline in this session with `executing-plans`, because the tasks are sequential and touch shared runtime files.

