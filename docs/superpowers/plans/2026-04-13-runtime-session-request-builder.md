# Runtime Session Request Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `runtime-session` input assembly into a reusable library builder while keeping CLI flags, output, and runtime behavior unchanged.

**Architecture:** Add a small builder in the Runtime library that owns artifact loading and task-graph assembly. Keep `runtime_session_main.cpp` responsible for CLI parsing, execution orchestration, and printing only.

**Tech Stack:** C++17, LLVM Support, existing Runtime artifact/taskgraph/session libraries, xvm focused runtime verification

---

## File Map

**Create:**
- `include/Runtime/Artifact/RuntimeSessionRequestBuilder.h` — library-facing request structs and builder API for artifact and graph assembly
- `include/Runtime/RuntimeSessionRequestBuilder.h` — forwarding shim for compatibility with top-level Runtime includes
- `lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp` — implementation of artifact-root loading, compile-request handling, manifest-to-graph assembly

**Modify:**
- `lib/Runtime/CMakeLists.txt` — add new builder source/header to runtime library build
- `tools/runtime-session/runtime_session_main.cpp` — replace local assembly helpers with builder calls; keep CLI/output behavior unchanged
- `test/tools/runtime/test_taskgraph_runtime.cpp` — add focused tests for builder behavior and keep existing runtime-session behavior covered

**Verification:**
- `bash test/tools/runtime/run_runtime.sh`

---

### Task 1: Add Builder API Surface

**Files:**
- Create: `include/Runtime/Artifact/RuntimeSessionRequestBuilder.h`
- Create: `include/Runtime/RuntimeSessionRequestBuilder.h`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing builder API tests**

Add focused declarations and tests in `test/tools/runtime/test_taskgraph_runtime.cpp` for the new builder entry points. Add tests for:
- artifact-root manifest -> `KernelArtifact`
- single artifact -> single task graph
- retained mix resource type survives builder path

Use tests shaped like:

```cpp
llvm::Expected<KernelArtifact>
loadRuntimeSessionArtifactFromRoot(llvm::StringRef artifactRoot);
llvm::Expected<TaskGraph>
buildRuntimeSessionSingleTaskGraph(const KernelArtifact &artifact,
                                   llvm::StringRef taskId);
```

Add one test fixture using a temporary artifact root with `out/manifest.txt` and assert:

```cpp
EXPECT(artifactOr->kernelName == "fake_kernel",
       "runtime session builder loads kernel name from manifest root");
EXPECT(artifactOr->kernelKind == KernelKind::Mix,
       "runtime session builder loads kernel kind from manifest root");
EXPECT(artifactOr->mixResourceType == MixResourceType::Mix1C1V,
       "runtime session builder loads mix resource type from manifest root");
```

Add one graph-building test:

```cpp
EXPECT(tasksOr->size() == 1,
       "runtime session builder creates a single task graph");
EXPECT(tasksOr->front().taskId == "main",
       "runtime session builder uses provided task id");
```

- [ ] **Step 2: Run focused runtime verification to confirm red**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
```

Expected:
- build/test failure due to missing builder declarations/definitions or undefined references for the new builder functions

- [ ] **Step 3: Add the public builder header**

Create `include/Runtime/Artifact/RuntimeSessionRequestBuilder.h` with a minimal surface:

```cpp
#pragma once

#include "Runtime/Artifact/ArtifactCompiler.h"
#include "Runtime/Execution/TaskGraph.h"
#include "Runtime/Support/Types.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <utility>

namespace mlir::runtime {

struct RuntimeSessionArtifactRequest {
  std::string artifactRoot;
  std::string kernelSource;
  std::string kernelName;
  std::string outputDir;
  std::string socVersion;
  std::optional<std::string> cannMlirPath;
  std::optional<std::string> npyDir;
  KernelKind kernelKind = KernelKind::Mix;
};

llvm::Expected<KernelArtifact>
loadRuntimeSessionArtifactFromRoot(llvm::StringRef artifactRoot);

llvm::Expected<KernelArtifact>
prepareRuntimeSessionArtifact(const RuntimeSessionArtifactRequest &request);

llvm::Expected<TaskGraph>
buildRuntimeSessionSingleTaskGraph(const KernelArtifact &artifact,
                                   llvm::StringRef taskId);

llvm::Expected<std::pair<ExecutionBackendKind, TaskGraph>>
prepareRuntimeSessionGraphFromManifest(llvm::StringRef runManifestPath);

} // namespace mlir::runtime
```

Create forwarding shim `include/Runtime/RuntimeSessionRequestBuilder.h`:

```cpp
#pragma once
#include "Runtime/Artifact/RuntimeSessionRequestBuilder.h"
```

- [ ] **Step 4: Commit the red-to-header step**

```bash
git add include/Runtime/Artifact/RuntimeSessionRequestBuilder.h include/Runtime/RuntimeSessionRequestBuilder.h test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "test: add runtime session request builder api coverage"
```

### Task 2: Implement Builder Logic in the Runtime Library

**Files:**
- Create: `lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp`
- Modify: `lib/Runtime/CMakeLists.txt`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Implement the minimal builder functions by moving current CLI assembly logic**

Create `lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp` and move the existing logic out of `runtime_session_main.cpp` into library functions.

The implementation should carry over the existing helper behavior for:
- manifest parsing
- required manifest field extraction
- artifact-root relative path resolution
- manifest path discovery
- `kernel_kind` parsing
- `mix_resource_type` parsing
- compile-vs-artifact-root branching
- single-task graph construction
- run-manifest graph construction

Keep helper functions file-local, e.g.:

```cpp
static llvm::Expected<KernelKind> parseBuilderKernelKind(...);
static llvm::Expected<MixResourceType> parseBuilderMixResourceType(...);
static std::map<std::string, std::string> readManifest(...);
static llvm::Expected<std::string> requireManifestValue(...);
static std::string resolveArtifactPath(...);
static llvm::Expected<std::string> locateManifestPath(...);
```

Then implement the exported functions with the same behavior currently used by the CLI.

- [ ] **Step 2: Add the builder source to the runtime library build**

Update `lib/Runtime/CMakeLists.txt` to compile the new builder source in the `Artifact` group.

Add the new `.cpp` next to the other artifact sources.

- [ ] **Step 3: Run focused runtime verification to turn green**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
```

Expected:
- `test_taskgraph_runtime` passes with the new builder tests
- runtime focused verification still passes end-to-end

- [ ] **Step 4: Commit the builder implementation**

```bash
git add lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp lib/Runtime/CMakeLists.txt test/tools/runtime/test_taskgraph_runtime.cpp

git commit -m "feat: add runtime session request builder"
```

### Task 3: Replace CLI-local Assembly With Builder Calls

**Files:**
- Modify: `tools/runtime-session/runtime_session_main.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Remove CLI-local request assembly helpers**

Delete from `runtime_session_main.cpp` the helpers that are now library concerns:
- `parseKernelKind(...)` only if no longer needed locally
- `parseManifestKernelKind(...)`
- `parseManifestMixResourceType(...)`
- `defaultKernelName(...)` only if builder now owns it
- `readManifest(...)`
- `requireManifestValue(...)`
- `resolveArtifactPath(...)`
- `locateManifestPath(...)`
- `loadArtifactFromRoot(...)`
- `prepareArtifact()`
- `buildGraph(...)`
- `prepareManifestGraph()`

Keep locally:
- CLI option declarations
- summary printers
- error printers
- retained-profile lifecycle calls
- testing-driver plumbing

- [ ] **Step 2: Replace main() assembly with builder calls**

Update `main()` so it uses:

```cpp
auto artifactOr = prepareRuntimeSessionArtifact(request);
auto graphOr = buildRuntimeSessionSingleTaskGraph(*artifact, TaskId);
auto manifestGraphOr = prepareRuntimeSessionGraphFromManifest(RunManifestPath);
```

The CLI branching should remain behaviorally identical:
- `--run-manifest` path builds `(backendKind, graph)` from builder
- otherwise `--artifact-root` or `--kernel` path builds artifact from builder, prints artifact summary, then wraps into a single-task graph through builder

- [ ] **Step 3: Run focused runtime verification again**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
```

Expected:
- same passing results as before
- no regression in CLI output or runtime behavior

- [ ] **Step 4: Commit the CLI cutover**

```bash
git add tools/runtime-session/runtime_session_main.cpp

git commit -m "refactor: move runtime session assembly into builder"
```

### Task 4: Final Cleanup and Verification

**Files:**
- Modify: `tools/runtime-session/runtime_session_main.cpp` (if minor cleanup remains)
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Remove dead includes and dead declarations**

Clean up any now-unused includes or forward declarations left behind in `runtime_session_main.cpp` and the new builder files.

Typical cleanup targets:
- `llvm/Support/MemoryBuffer.h`
- `llvm/Support/Path.h`
- `map` or `optional` includes that moved into the builder

- [ ] **Step 2: Re-run full focused verification one last time**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
```

Expected:
- `test_taskgraph_runtime`: all pass
- `test_capi_runtime`: all pass
- `test_runtime`: all pass
- sim smoke baseline passes
- repeated mix simulation baseline passes

- [ ] **Step 3: Commit cleanup**

```bash
git add tools/runtime-session/runtime_session_main.cpp lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp include/Runtime/Artifact/RuntimeSessionRequestBuilder.h include/Runtime/RuntimeSessionRequestBuilder.h lib/Runtime/CMakeLists.txt test/tools/runtime/test_taskgraph_runtime.cpp

git commit -m "chore: clean up runtime session builder cutover"
```

---

## Self-Review Checklist

Spec coverage:
- builder introduced: covered in Tasks 1-2
- artifact and graph assembly moved out of CLI: covered in Tasks 2-3
- CLI behavior unchanged: verified in Tasks 3-4
- xvm verification preserved: required in Tasks 2-4

Placeholder scan:
- No TODO/TBD markers
- All tasks include exact files and commands
- Tests and commands are explicit

Type consistency:
- Builder API names are consistent across tasks:
  - `loadRuntimeSessionArtifactFromRoot`
  - `prepareRuntimeSessionArtifact`
  - `buildRuntimeSessionSingleTaskGraph`
  - `prepareRuntimeSessionGraphFromManifest`
