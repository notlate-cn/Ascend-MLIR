# Runtime Directory Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `include/Runtime` and `lib/Runtime` into architecture-aligned subdirectories without changing runtime behavior.

**Architecture:** Move headers and sources into six mirrored subdirectories: `Artifact`, `Execution`, `Profile`, `Mix`, `Support`, and `Legacy`. Preserve existing `#include "Runtime/..."` call sites by leaving thin top-level forwarding headers at the old public include paths, then update CMake and verify the entire runtime flow on xvm.

**Tech Stack:** C++, CMake, LLVM support library, bash/xvm verification scripts

---

### Task 1: Move Canonical Headers Into Subdirectories

**Files:**
- Create: `include/Runtime/Artifact/ArtifactCompiler.h`
- Create: `include/Runtime/Artifact/RunManifest.h`
- Create: `include/Runtime/Execution/ExecutionBackend.h`
- Create: `include/Runtime/Execution/ExecutionSession.h`
- Create: `include/Runtime/Execution/NpuBackend.h`
- Create: `include/Runtime/Execution/SimBackend.h`
- Create: `include/Runtime/Execution/TaskGraph.h`
- Create: `include/Runtime/Profile/ProfileTrace.h`
- Create: `include/Runtime/Profile/ProfileUtils.h`
- Create: `include/Runtime/Mix/AscendCMixCompiler.h`
- Create: `include/Runtime/Mix/MixAbi.h`
- Create: `include/Runtime/Mix/MixAbiExtractor.h`
- Create: `include/Runtime/Mix/MixArtifact.h`
- Create: `include/Runtime/Mix/MixCommandBuilder.h`
- Create: `include/Runtime/Mix/MixDirectBackend.h`
- Create: `include/Runtime/Mix/MixSourceAnalyzer.h`
- Create: `include/Runtime/Mix/MixStubTemplate.h`
- Create: `include/Runtime/Support/NpyIO.h`
- Create: `include/Runtime/Support/PathUtils.h`
- Create: `include/Runtime/Support/TilingPack.h`
- Create: `include/Runtime/Support/TilingSchema.h`
- Create: `include/Runtime/Support/Types.h`
- Create: `include/Runtime/Legacy/CompatRuntime.h`
- Create: `include/Runtime/Legacy/Compiler.h`
- Create: `include/Runtime/Legacy/Executor.h`
- Create: `include/Runtime/Legacy/HostRunnerGen.h`
- Create: `include/Runtime/Legacy/SimValidator.h`
- Modify: `include/Runtime/*.h`

- [ ] **Step 1: Write the failing smoke check for canonical header placement**

Create a temporary shell check in `/tmp/runtime-header-layout-check.sh` on xvm:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /Users/niu/Code/Codex-Ascend-MLIR

test -f include/Runtime/Execution/ExecutionSession.h
test -f include/Runtime/Profile/ProfileUtils.h
test -f include/Runtime/Mix/MixDirectBackend.h
test -f include/Runtime/Legacy/Executor.h
```

- [ ] **Step 2: Run check to verify it fails before header moves**

Run: `bash /tmp/runtime-header-layout-check.sh`
Expected: `test -f ...` fails because the subdirectory header files do not exist yet.

- [ ] **Step 3: Move the real header contents into canonical subdirectory paths**

For each header listed above:

1. Create the target subdirectory if needed.
2. Move the current header body unchanged into the new canonical path.
3. Do not change declarations except include paths if the moved header needs them.

Representative code rule for every moved canonical header:

```cpp
#pragma once

#include "Runtime/Support/Types.h"
#include "llvm/Support/Error.h"

namespace mlir::runtime {
// existing declarations copied without behavior changes
}
```

- [ ] **Step 4: Replace each old top-level public header with a forwarding shim**

Each original top-level header should become a thin wrapper only. Example:

```cpp
#pragma once

#include "Runtime/Execution/ExecutionSession.h"
```

Apply the same pattern to every moved public header so existing includes still compile.

- [ ] **Step 5: Run the header placement smoke check again**

Run: `bash /tmp/runtime-header-layout-check.sh`
Expected: succeeds with exit code `0`.

- [ ] **Step 6: Commit**

```bash
git add include/Runtime
git commit -m "refactor: reorganize runtime public headers"
```

### Task 2: Move Runtime Source Files Into Mirrored Subdirectories

**Files:**
- Create: `lib/Runtime/Artifact/ArtifactCompiler.cpp`
- Create: `lib/Runtime/Artifact/RunManifest.cpp`
- Create: `lib/Runtime/Execution/ExecutionBackend.cpp`
- Create: `lib/Runtime/Execution/ExecutionSession.cpp`
- Create: `lib/Runtime/Execution/NpuBackend.cpp`
- Create: `lib/Runtime/Execution/SimBackend.cpp`
- Create: `lib/Runtime/Execution/TaskGraph.cpp`
- Create: `lib/Runtime/Profile/ProfileTrace.cpp`
- Create: `lib/Runtime/Profile/ProfileUtils.cpp`
- Create: `lib/Runtime/Mix/AscendCMixCompiler.cpp`
- Create: `lib/Runtime/Mix/MixAbi.cpp`
- Create: `lib/Runtime/Mix/MixAbiExtractor.cpp`
- Create: `lib/Runtime/Mix/MixCommandBuilder.cpp`
- Create: `lib/Runtime/Mix/MixDirectBackend.cpp`
- Create: `lib/Runtime/Mix/MixSourceAnalyzer.cpp`
- Create: `lib/Runtime/Mix/MixStubTemplate.cpp`
- Create: `lib/Runtime/Mix/AscendCannPaths.cmake`
- Create: `lib/Runtime/Support/NpyIO.cpp`
- Create: `lib/Runtime/Support/PathUtils.cpp`
- Create: `lib/Runtime/Support/TilingPack.cpp`
- Create: `lib/Runtime/Support/TilingSchema.cpp`
- Create: `lib/Runtime/Legacy/CompatRuntime.cpp`
- Create: `lib/Runtime/Legacy/Compiler.cpp`
- Create: `lib/Runtime/Legacy/Executor.cpp`
- Create: `lib/Runtime/Legacy/HostRunnerGen.cpp`
- Create: `lib/Runtime/Legacy/SimValidator.cpp`
- Modify: moved `.cpp` files for include path correctness only

- [ ] **Step 1: Write the failing source layout smoke check**

Create `/tmp/runtime-source-layout-check.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /Users/niu/Code/Codex-Ascend-MLIR

test -f lib/Runtime/Execution/ExecutionSession.cpp
test -f lib/Runtime/Profile/ProfileUtils.cpp
test -f lib/Runtime/Mix/MixDirectBackend.cpp
test -f lib/Runtime/Legacy/Executor.cpp
```

- [ ] **Step 2: Run check to verify it fails before source moves**

Run: `bash /tmp/runtime-source-layout-check.sh`
Expected: fails because the moved source files do not exist yet.

- [ ] **Step 3: Move implementation files into matching subdirectories**

Move each listed source file into its mirrored location under `lib/Runtime/...`.

Rules:

- preserve implementation bodies exactly
- only adjust local includes if the move breaks relative organization clarity
- do not change namespaces, symbols, or behavior

Representative include style after move:

```cpp
#include "Runtime/Execution/ExecutionSession.h"
#include "Runtime/Profile/ProfileUtils.h"
```

- [ ] **Step 4: Move `AscendCannPaths.cmake` under `lib/Runtime/Mix/`**

Update only file location in this task. The CMake reference update belongs to Task 3.

- [ ] **Step 5: Run the source layout smoke check again**

Run: `bash /tmp/runtime-source-layout-check.sh`
Expected: succeeds with exit code `0`.

- [ ] **Step 6: Commit**

```bash
git add lib/Runtime
git commit -m "refactor: reorganize runtime source layout"
```

### Task 3: Update CMake and Build References To New Layout

**Files:**
- Modify: `lib/Runtime/CMakeLists.txt`
- Modify: any CMake file or script that references `lib/Runtime/AscendCannPaths.cmake`
- Search: `CMakeLists.txt`, `cmake/**/*.cmake`, `tools/**`, `test/**`

- [ ] **Step 1: Write the failing build probe**

Run the existing focused build without CMake updates:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
cmake --build build --target runtime-session -j2
```

Expected: build fails because source file paths in CMake still point at the old flat layout.

- [ ] **Step 2: Update `lib/Runtime/CMakeLists.txt` source lists**

Rewrite every runtime source entry to use the new canonical paths. Example pattern:

```cmake
set(RUNTIME_SOURCES
  Artifact/ArtifactCompiler.cpp
  Artifact/RunManifest.cpp
  Execution/ExecutionBackend.cpp
  Execution/ExecutionSession.cpp
  Profile/ProfileTrace.cpp
  Profile/ProfileUtils.cpp
  Mix/MixDirectBackend.cpp
  Support/NpyIO.cpp
  Legacy/Executor.cpp
)
```

- [ ] **Step 3: Update any moved `AscendCannPaths.cmake` references**

Find every reference to the old path and point it to:

```cmake
${CMAKE_CURRENT_SOURCE_DIR}/Mix/AscendCannPaths.cmake
```

or the equivalent correct path from the referring file.

- [ ] **Step 4: Re-run the focused build**

Run:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
cmake --build build --target runtime-session -j2
```

Expected: build succeeds and produces `build/bin/runtime-session`.

- [ ] **Step 5: Commit**

```bash
git add lib/Runtime/CMakeLists.txt
git add . ':!docs/superpowers/plans/2026-04-10-runtime-taskgraph-mix.md'
git commit -m "build: update runtime paths for directory reorg"
```

### Task 4: Verify Forwarding Headers Preserve Existing Include Paths

**Files:**
- Modify: only if compile failures reveal missing forwarding headers or include path mistakes
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`
- Test: `test/tools/runtime/test_runtime.cpp`
- Test: `test/tools/runtime/test_capi_runtime.cpp`

- [ ] **Step 1: Write the failing compatibility probe**

Compile the focused runtime tests with the existing top-level includes unchanged:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected: if any forwarding header is missing or wrong, compilation fails in one of the runtime tests before execution begins.

- [ ] **Step 2: Fix only forwarding/header path issues revealed by the probe**

Allowed changes:

- add missing forwarding header
- correct forwarding target path
- adjust canonical header includes from old flat paths to the new grouped paths where necessary

Do not change runtime behavior in this task.

- [ ] **Step 3: Re-run the compatibility probe**

Run: `bash test/tools/runtime/run_runtime.sh`
Expected: completes successfully through all focused runtime checks.

- [ ] **Step 4: Commit**

```bash
git add include/Runtime lib/Runtime
git commit -m "fix: preserve runtime include compatibility after reorg"
```

### Task 5: xvm End-to-End Verification Of The Reorganized Layout

**Files:**
- Test only

- [ ] **Step 1: Run focused runtime verification on xvm**

Run:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected:

- `test_taskgraph_runtime` passes
- `test_capi_runtime` passes
- `test_runtime` passes
- `SimBackend` vec/mix smoke passes

- [ ] **Step 2: Run 6-example SimBackend verification on xvm**

Run:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
source examples/env.sh
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_simbackend_examples.sh
```

Expected:

- all 6 examples pass
- vec and mix summaries print retained profile paths

- [ ] **Step 3: Record the final clean state**

Run:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
git status --short
```

Expected:

- only intended tracked reorg changes remain
- `docs/superpowers/plans/2026-04-10-runtime-taskgraph-mix.md` remains untracked and untouched

- [ ] **Step 4: Commit**

```bash
git add include/Runtime lib/Runtime
git commit -m "test: verify runtime directory reorganization"
```
