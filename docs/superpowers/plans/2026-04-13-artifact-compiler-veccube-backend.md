# ArtifactCompiler Vec/Cube Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `ArtifactCompiler -> Legacy/Compiler` dependency for `vec` and `cube` source builds by introducing a runtime-native vec/cube compile backend while preserving the existing artifact/manifest contract.

**Architecture:** Add a dedicated vec/cube backend in the `Artifact` submodule and make `ArtifactCompiler` a dispatcher: vec/cube requests go to the new backend, mix requests continue to use `MixDirectBackend`. Keep emitted `KernelArtifact` and `out/manifest.txt` semantics stable so `runtime-session`, `autotuner`, examples, and C API do not need interface changes.

**Tech Stack:** C++17, LLVM Support, existing runtime artifact/session libraries, xvm verification, CMake

---

## File Map

**Create:**
- `include/Runtime/Artifact/VecCubeArtifactBackend.h` — public runtime-native vec/cube compile backend API
- `include/Runtime/VecCubeArtifactBackend.h` — forwarding shim for the new header
- `lib/Runtime/Artifact/VecCubeArtifactBackend.cpp` — vec/cube compile implementation and manifest emission

**Modify:**
- `include/Runtime/Artifact/ArtifactCompiler.h` — stop depending on legacy compiler header
- `lib/Runtime/Artifact/ArtifactCompiler.cpp` — delegate vec/cube requests to the new backend, keep mix path unchanged
- `lib/Runtime/CMakeLists.txt` — compile the new backend source and keep build graph intact
- `test/tools/runtime/test_taskgraph_runtime.cpp` — focused backend/ArtifactCompiler coverage for vec/cube path and contract preservation
- `tools/autotuner/autotuner_main.cpp` — only if verification reveals an integration regression; otherwise leave unchanged

**Reference Only:**
- `include/Runtime/Legacy/Compiler.h`
- `lib/Runtime/Legacy/Compiler.cpp`
- `include/Runtime/Mix/MixDirectBackend.h`
- `lib/Runtime/Mix/MixDirectBackend.cpp`
- `test/tools/runtime/run_runtime.sh`

**Verification:**
- `ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'`
- `ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && source examples/env.sh && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && cmake --build build --target autotuner -j2 && build/bin/autotuner --space examples/relu-broadcast-transpose/tiling_space.json --kernel examples/relu-broadcast-transpose/step8_kernel.cpp --kernel-kind vec --inputs examples/relu-broadcast-transpose/input_data0.npy,examples/relu-broadcast-transpose/input_data1.npy --expected examples/relu-broadcast-transpose/output_expected.npy --shape M=640,N=500 --output /tmp/autotuner-best.json'`
- `rg -n "Runtime/Legacy/Compiler.h|\bCompiler compiler\(" include/Runtime/Artifact/ArtifactCompiler.h lib/Runtime/Artifact/ArtifactCompiler.cpp`

---

### Task 1: Add Focused Failing Coverage For The New Vec/Cube Backend Contract

**Files:**
- Create: `include/Runtime/Artifact/VecCubeArtifactBackend.h`
- Create: `include/Runtime/VecCubeArtifactBackend.h`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
- Reference: `include/Runtime/Artifact/ArtifactCompiler.h`

- [ ] **Step 1: Declare the new backend API header with the minimal contract**

Create `include/Runtime/Artifact/VecCubeArtifactBackend.h` with a small API that accepts a normalized compile request shape and returns `KernelArtifact`:

```cpp
#pragma once

#include "Runtime/Artifact/ArtifactCompiler.h"
#include "llvm/Support/Error.h"

namespace mlir::runtime {

class VecCubeArtifactBackend {
public:
  llvm::Expected<KernelArtifact>
  compile(const ArtifactCompileRequest &req, llvm::StringRef resolvedSoc) const;
};

} // namespace mlir::runtime
```

Create the forwarding shim `include/Runtime/VecCubeArtifactBackend.h`:

```cpp
#pragma once

#include "Runtime/Artifact/VecCubeArtifactBackend.h"
```

- [ ] **Step 2: Add failing runtime tests that lock the vec/cube backend expectations**

In `test/tools/runtime/test_taskgraph_runtime.cpp`, add focused tests that compile against the new API and lock the expected output contract. Use the existing temp-dir fixture style already present in this file.

Add tests covering:
- vec request through `VecCubeArtifactBackend` yields `kernelKind == KernelKind::Vec`
- cube request through `VecCubeArtifactBackend` yields `kernelKind == KernelKind::Cube`
- both yield `mixResourceType == MixResourceType::Unknown`
- both emit a manifest under `out/manifest.txt`
- the manifest contains `kernel_kind=vec` or `kernel_kind=cube`

Use skeletons like:

```cpp
static void testVecCubeArtifactBackendWritesVecManifest() {
  const std::filesystem::path tempRoot =
      makeTempDir("vec-cube-backend-vec");
  RuntimeSessionTempRoot cleanup(tempRoot);

  ArtifactCompileRequest req;
  req.kernelSource = "/tmp/fake.cpp";
  req.kernelName = "fake_vec";
  req.kernelKind = KernelKind::Vec;
  req.outputDir = cleanup.path.string();

  VecCubeArtifactBackend backend;
  auto artifactOr = backend.compile(req, "Ascend910B1");
  EXPECT((bool)artifactOr, "vec backend compiles request");
  if (!artifactOr) {
    llvm::consumeError(artifactOr.takeError());
    return;
  }
  EXPECT(artifactOr->kernelKind == KernelKind::Vec,
         "vec backend preserves vec kind");
  EXPECT(artifactOr->mixResourceType == MixResourceType::Unknown,
         "vec backend keeps mix resource unknown");
  EXPECT(std::filesystem::exists(artifactOr->manifestPath),
         "vec backend writes manifest");
}
```

Do the same for `KernelKind::Cube`. For the first red run, it is acceptable if these fail at link time because the backend implementation does not yet exist.

- [ ] **Step 3: Wire the new tests into `main()`**

Add the new test invocations to the existing `main()` test list in `test/tools/runtime/test_taskgraph_runtime.cpp`.

- [ ] **Step 4: Run focused runtime verification to confirm the red state**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
```

Expected:
- red due to missing `VecCubeArtifactBackend` implementation or missing CMake wiring
- not due to unrelated runtime breakage

- [ ] **Step 5: Commit the failing coverage and API skeleton**

```bash
git add include/Runtime/Artifact/VecCubeArtifactBackend.h include/Runtime/VecCubeArtifactBackend.h test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "test: cover vec cube artifact backend contract"
```

### Task 2: Implement The Runtime-Native Vec/Cube Backend

**Files:**
- Create: `lib/Runtime/Artifact/VecCubeArtifactBackend.cpp`
- Modify: `lib/Runtime/CMakeLists.txt`
- Reference: `lib/Runtime/Artifact/ArtifactCompiler.cpp`
- Reference: `include/Runtime/Legacy/Compiler.h`

- [ ] **Step 1: Implement the backend with the current vec/cube artifact contract**

Create `lib/Runtime/Artifact/VecCubeArtifactBackend.cpp` with the implementation of `VecCubeArtifactBackend::compile(...)`.

Required behavior:
- validate `req.kernelSource`, `req.kernelName`, and `req.outputDir`
- derive vec/cube default arch exactly as current behavior:
  - cube -> `dav-c220-cube`
  - vec -> `dav-c220-vec`
- invoke the underlying compile machinery needed for vec/cube output generation
- normalize the result into `KernelArtifact`
- emit `out/manifest.txt`
- set `mixResourceType = MixResourceType::Unknown`

The implementation may temporarily call lower-level compile primitives if absolutely necessary, but the backend itself must be the new owner of vec/cube compilation semantics, not `ArtifactCompiler`.

Preserve the manifest content format already expected by runtime consumers:

```text
kernel_name=<name>
soc_version=<soc>
kernel_kind=<vec|cube>
device_binary_path=<relative path>
manifest_path=out/manifest.txt
```

- [ ] **Step 2: Add the new source file to the runtime build**

Update `lib/Runtime/CMakeLists.txt` to compile `Artifact/VecCubeArtifactBackend.cpp` as part of the runtime library.

- [ ] **Step 3: Run focused runtime verification to turn Task 1 green**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
```

Expected:
- `test_taskgraph_runtime` now passes the new vec/cube backend tests
- no regression in existing smoke baselines

- [ ] **Step 4: Commit the backend implementation**

```bash
git add lib/Runtime/Artifact/VecCubeArtifactBackend.cpp lib/Runtime/CMakeLists.txt
git commit -m "feat: add vec cube artifact backend"
```

### Task 3: Cut ArtifactCompiler Off From Legacy/Compiler

**Files:**
- Modify: `include/Runtime/Artifact/ArtifactCompiler.h`
- Modify: `lib/Runtime/Artifact/ArtifactCompiler.cpp`
- Reference: `include/Runtime/Artifact/VecCubeArtifactBackend.h`
- Reference: `include/Runtime/Mix/MixDirectBackend.h`

- [ ] **Step 1: Remove the legacy compiler header dependency from the public ArtifactCompiler header**

In `include/Runtime/Artifact/ArtifactCompiler.h`, remove:

```cpp
#include "Runtime/Legacy/Compiler.h"
```

Keep only the includes actually needed for `ArtifactCompileRequest`, `KernelArtifact`, and error types.

- [ ] **Step 2: Make ArtifactCompiler a dispatcher**

In `lib/Runtime/Artifact/ArtifactCompiler.cpp`:
- remove `#include "Runtime/Compiler.h"`
- include the new vec/cube backend header
- keep the mix branch on `MixDirectBackend`
- route vec/cube requests to `VecCubeArtifactBackend::compile(req, resolvedSoc)`

The resulting branching shape should be:

```cpp
if (req.kernelKind == KernelKind::Mix) {
  ... existing mix path ...
}

VecCubeArtifactBackend backend;
return backend.compile(req, resolvedSoc);
```

Do not reintroduce runner compatibility outputs.

- [ ] **Step 3: Verify ArtifactCompiler no longer depends on Legacy/Compiler**

Run:

```bash
rg -n "Runtime/Legacy/Compiler.h|\bCompiler compiler\(" include/Runtime/Artifact/ArtifactCompiler.h lib/Runtime/Artifact/ArtifactCompiler.cpp
```

Expected:
- no matches in either file

- [ ] **Step 4: Run focused runtime verification again**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
```

Expected:
- runtime verification passes fully
- vec and mix smoke still pass
- repeated mix simulation baseline still passes

- [ ] **Step 5: Commit the dispatcher cutover**

```bash
git add include/Runtime/Artifact/ArtifactCompiler.h lib/Runtime/Artifact/ArtifactCompiler.cpp
git commit -m "refactor: route artifact compiler through vec cube backend"
```

### Task 4: Verify Consumer Compatibility

**Files:**
- Verify: `tools/autotuner/autotuner_main.cpp`
- Verify: `lib/CAPI/Runtime/Runtime.cpp`
- Verify: vec examples via existing scripts
- Modify only if verification finds a real regression

- [ ] **Step 1: Run autotuner vec smoke on xvm**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && source examples/env.sh && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && cmake --build build --target autotuner -j2 && build/bin/autotuner --space examples/relu-broadcast-transpose/tiling_space.json --kernel examples/relu-broadcast-transpose/step8_kernel.cpp --kernel-kind vec --inputs examples/relu-broadcast-transpose/input_data0.npy,examples/relu-broadcast-transpose/input_data1.npy --expected examples/relu-broadcast-transpose/output_expected.npy --shape M=640,N=500 --output /tmp/autotuner-best.json'`
```

Expected:
- autotuner completes successfully
- output JSON exists
- `cycle_count` and `score` remain non-zero

- [ ] **Step 2: If practical, confirm a cube compile path remains valid**

If there is an existing stable cube artifact path or test fixture, run it. If not, document that limitation explicitly in task notes and rely on the focused vec/cube backend tests added in Task 1.

- [ ] **Step 3: If verification finds no regression, do not change consumer code**

If all consumer-facing verification passes, leave `autotuner`, C API, and examples untouched.

If a small compatibility fix is required, keep it narrowly scoped and commit it separately.

- [ ] **Step 4: Commit only if a verification-driven compatibility fix was required**

```bash
git add <only the compatibility fix files>
git commit -m "fix: preserve artifact compiler consumer compatibility"
```

## Self-Review

- Spec coverage:
  - remove `ArtifactCompiler -> Legacy/Compiler` for vec/cube: covered by Task 3
  - add runtime-native vec/cube backend: covered by Task 2
  - keep mix path unchanged: preserved in Task 3
  - preserve artifact/manifest contract: covered by Tasks 1 and 2
  - no runner compatibility outputs: explicitly preserved in Tasks 2 and 3
  - xvm runtime + autotuner verification: covered by Tasks 2, 3, and 4
- Placeholder scan:
  - no `TODO` / `TBD` / “similar to” placeholders remain
- Type consistency:
  - new backend API consistently uses `VecCubeArtifactBackend::compile(const ArtifactCompileRequest &, llvm::StringRef)`
  - artifact output type consistently remains `KernelArtifact`
