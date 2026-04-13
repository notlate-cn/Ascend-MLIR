# Autotuner Artifact Root Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize `autotuner --artifact-root` so it uses the same canonical runtime artifact-loading semantics as `runtime-session` and preserves mix metadata such as `mix_resource_type`.

**Architecture:** Replace autotuner-local artifact-root manifest loading with the shared runtime loader `loadRuntimeSessionArtifactFromRoot(...)`. Keep autotuner-specific source-build, candidate execution, profiling, and output logic unchanged.

**Tech Stack:** C++17, LLVM Support, Runtime artifact/taskgraph/session libraries, xvm autotuner smoke verification

---

## File Map

**Modify:**
- `tools/autotuner/autotuner_main.cpp` — remove local artifact-root manifest loading path and route `--artifact-root` through the canonical runtime loader
- `test/tools/runtime/test_taskgraph_runtime.cpp` — add focused coverage for canonical artifact-root loading preserving mix resource metadata when used by autotuner-facing paths

**Reference Only:**
- `include/Runtime/Artifact/RuntimeSessionRequestBuilder.h`
- `lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp`
- `test/tools/runtime/run_runtime.sh`
- `tools/runtime-session/runtime_session_main.cpp`

**Verification:**
- `ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'`
- `ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && source examples/env.sh && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && cmake --build build --target autotuner -j2 && build/bin/autotuner --space examples/relu-broadcast-transpose/tiling_space.json --kernel examples/relu-broadcast-transpose/step8_kernel.cpp --kernel-kind vec --inputs examples/relu-broadcast-transpose/input_data0.npy,examples/relu-broadcast-transpose/input_data1.npy --expected examples/relu-broadcast-transpose/output_expected.npy --shape M=640,N=500 --output /tmp/autotuner-best.json'`

---

### Task 1: Add Focused Coverage For Canonical Artifact Loading

**Files:**
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
- Reference: `include/Runtime/Artifact/RuntimeSessionRequestBuilder.h`

- [ ] **Step 1: Write the failing tests for artifact-root normalization assumptions**

Add focused tests in `test/tools/runtime/test_taskgraph_runtime.cpp` that use the canonical artifact loader API and explicitly lock the mix metadata contract expected by autotuner.

Add tests for:
- mix artifact-root loads preserve `MixResourceType::Mix1C1V`
- vec artifact-root loads preserve `KernelKind::Vec` and keep `MixResourceType::Unknown`

Use the existing temporary manifest fixture style already present in this file. Add a vec fixture parallel to the existing mix fixture:

```cpp
static std::filesystem::path makeRuntimeSessionVecArtifactRoot(
    const std::string &stem) {
  std::filesystem::path root = makeTempDir(stem);
  std::filesystem::create_directories(root / "out");
  std::ofstream manifest(root / "out" / "manifest.txt");
  manifest << "kernel_name=fake_vec\n";
  manifest << "soc_version=Ascend910B1\n";
  manifest << "kernel_kind=vec\n";
  manifest << "device_binary_path=fake.bin\n";
  std::ofstream binary(root / "fake.bin", std::ios::binary);
  binary.put('\0');
  return root;
}
```

And add a test like:

```cpp
static void testRuntimeSessionRequestBuilderLoadsVecArtifactFromRoot() {
  const std::filesystem::path rootPath =
      makeRuntimeSessionVecArtifactRoot("runtime-session-builder-vec");
  RuntimeSessionTempRoot cleanup(rootPath);
  auto artifactOr = loadRuntimeSessionArtifactFromRoot(cleanup.path.string());
  EXPECT((bool)artifactOr, "runtime session builder loads vec artifact root");
  if (!artifactOr) {
    llvm::consumeError(artifactOr.takeError());
    return;
  }
  EXPECT(artifactOr->kernelKind == KernelKind::Vec,
         "runtime session builder preserves vec kernel kind");
  EXPECT(artifactOr->mixResourceType == MixResourceType::Unknown,
         "runtime session builder keeps vec mix resource type unknown");
}
```

Also ensure the existing mix test remains in `main()` as the positive metadata-preservation case.

- [ ] **Step 2: Run focused runtime verification to confirm red if needed**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
```

Expected:
- If the new test references are wired correctly but implementation is not yet updated, the runtime test suite should still build, and any failing assertion should reveal the current gap.
- If the current code already satisfies the contract, note that the test is “red by coverage gap” rather than behavioral failure and proceed.

- [ ] **Step 3: Commit the focused coverage update**

```bash
git add test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "test: cover autotuner artifact-root normalization"
```

### Task 2: Remove Autotuner-Local Artifact Root Loading

**Files:**
- Modify: `tools/autotuner/autotuner_main.cpp`
- Reference: `include/Runtime/Artifact/RuntimeSessionRequestBuilder.h`

- [ ] **Step 1: Locate and remove the local artifact-root loader helpers**

In `tools/autotuner/autotuner_main.cpp`, identify the local helpers used only for artifact-root manifest parsing, including the equivalents of:

```cpp
static llvm::Expected<std::string> locateManifestPath(...);
static llvm::Expected<KernelArtifact> loadArtifactFromRoot(...);
```

Delete those helpers once their only caller is switched to the shared runtime loader.

- [ ] **Step 2: Route --artifact-root through the canonical runtime loader**

Update `prepareArtifact(const TilingSpace &space)` so the `ArtifactRoot` branch becomes:

```cpp
if (!ArtifactRoot.empty())
  return loadRuntimeSessionArtifactFromRoot(ArtifactRoot);
```

Add the required include:

```cpp
#include "Runtime/RuntimeSessionRequestBuilder.h"
```

Preserve the rest of `prepareArtifact(...)` unchanged:
- source-build path still uses `ArtifactCompiler`
- `--kernel-kind` handling remains the same
- `--kernel` / `kernel_file` resolution remains the same

- [ ] **Step 3: Verify the file no longer owns duplicated manifest semantics**

Run:

```bash
rg -n "locateManifestPath|loadArtifactFromRoot|mixResourceType = MixResourceType::Unknown" tools/autotuner/autotuner_main.cpp
```

Expected:
- the old local manifest loader symbols are gone from `autotuner_main.cpp`
- no artifact-root path is still hardcoding `MixResourceType::Unknown`

- [ ] **Step 4: Commit the autotuner loader normalization**

```bash
git add tools/autotuner/autotuner_main.cpp
git commit -m "refactor: use canonical artifact loader in autotuner"
```

### Task 3: Verify Behavior And Smoke Paths

**Files:**
- Modify if needed: `test/tools/runtime/test_taskgraph_runtime.cpp`
- Verify: `tools/autotuner/autotuner_main.cpp`

- [ ] **Step 1: Run focused runtime verification on xvm**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
```

Expected:
- `test_taskgraph_runtime` passes
- `test_capi_runtime` passes
- `test_runtime` passes
- SimBackend smoke passes
- repeated mix simulation baseline passes

- [ ] **Step 2: Run autotuner vec smoke on xvm**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && source examples/env.sh && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && cmake --build build --target autotuner -j2 && build/bin/autotuner --space examples/relu-broadcast-transpose/tiling_space.json --kernel examples/relu-broadcast-transpose/step8_kernel.cpp --kernel-kind vec --inputs examples/relu-broadcast-transpose/input_data0.npy,examples/relu-broadcast-transpose/input_data1.npy --expected examples/relu-broadcast-transpose/output_expected.npy --shape M=640,N=500 --output /tmp/autotuner-best.json'`
```

Expected:
- autotuner completes successfully
- `/tmp/autotuner-best.json` exists
- output still contains non-zero `cycle_count` / `score`

- [ ] **Step 3: If practical, run a focused mix artifact-root check**

If there is an existing mix artifact root available from the repo’s smoke flows, run a focused command or helper path to prove the normalized loader preserves `mix_resource_type` and does not fail immediately in scheduling.

If no stable CLI-level mix artifact-root smoke exists yet, document that limitation explicitly in the task notes and rely on the focused runtime test coverage added in Task 1.

- [ ] **Step 4: Commit any final verification-driven test adjustment**

If verification required no further edits, do not create an extra commit. If a small test-only adjustment was needed, commit it with:

```bash
git add test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "test: tighten autotuner artifact-root verification"
```

## Self-Review

- Spec coverage:
  - canonical artifact-root loading via runtime loader: covered by Task 2
  - mix metadata preservation: covered by Task 1 and Task 3
  - no scope expansion into ArtifactCompiler/backends: preserved across all tasks
- Placeholder scan:
  - no `TODO`, `TBD`, or undefined “similar to” steps remain
- Type/term consistency:
  - canonical loader is consistently named `loadRuntimeSessionArtifactFromRoot(...)`
  - the metadata term is consistently `mix_resource_type` / `MixResourceType`
