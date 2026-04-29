# Runtime Mix Compile Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a runtime-native, general `MixCompileMetadata` contract for mix artifacts while preserving all existing runtime-session, C API, backend, and test caller interfaces during the migration.

**Architecture:** Add a new JSON metadata object emitted by `MixDirectBackend` in parallel with the current text manifest, then teach artifact-loading and execution code to optionally consume it without breaking the legacy manifest contract. Keep legacy probe/config/stub generation in place during Phase 1 so the change is metadata-first, compatibility-first, and reversible.

**Tech Stack:** C++17, LLVM support utilities, existing Runtime mix/artifact/execution code, focused runtime xvm verification scripts.

---

### Task 1: Add General Mix Metadata Types

**Files:**
- Create: `include/Runtime/Mix/MixCompileMetadata.h`
- Create: `lib/Runtime/Mix/MixCompileMetadata.cpp`
- Modify: `lib/Runtime/CMakeLists.txt`
- Test: `test/tools/runtime/test_runtime.cpp`

- [ ] **Step 1: Add a failing metadata round-trip test**

Add a focused unit test in `test/tools/runtime/test_runtime.cpp` that exercises:
- parse JSON containing required top-level fields
- reject missing required fields
- preserve unknown optional fields by ignoring them

Use a test shape like:

```c++
TEST(RuntimeMixCompileMetadataTest, ParsesRequiredSchema) {
  const char *json = R"json(
{
  "schema_version": 1,
  "kernel_kind": "mix",
  "kernel_name": "k",
  "runtime_kernel_name": "k",
  "soc_version": "Ascend910B1",
  "mix_kernel_type": "mix_aic_1_2",
  "launcher_symbol": "aclrtlaunch_k",
  "entries": { "aic": "k_0_mix_aic", "aiv": "k_0_mix_aiv" },
  "generated": { "source_path": "work/generated/auto_gen_k.cpp" },
  "device_compile": {
    "aic_arch": "dav-c220-cube",
    "aiv_arch": "dav-c220-vec",
    "aic_definitions": [],
    "aiv_definitions": []
  },
  "artifacts": {
    "device_object_path": "out/device.o",
    "packed_shared_object_path": "out/libk_packed.so",
    "tiling_file_path": "out/tiling.bin",
    "launch_info_file_path": "out/launch_info.txt"
  },
  "abi": {
    "workspace_mode": "fixed",
    "workspace_bytes": 16777216,
    "tiling_mode": "generated_file",
    "tiling_source": "out/tiling.bin",
    "inputs": [],
    "outputs": []
  },
  "host_launch": {
    "mode": "helper",
    "helper_kind": "mix-tiling-helper",
    "helper_inputs": {}
  }
}
)json";
  auto metadataOr = parseMixCompileMetadataJson(json);
  ASSERT_TRUE(static_cast<bool>(metadataOr));
  EXPECT_EQ(metadataOr->kernelName, "k");
  EXPECT_EQ(metadataOr->runtimeKernelName, "k");
  EXPECT_EQ(metadataOr->mixKernelType, "mix_aic_1_2");
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cmake --build build --target test_runtime -j8
ctest --test-dir build --output-on-failure -R test_runtime
```

Expected: compile failure because `MixCompileMetadata` parser/types do not exist yet.

- [ ] **Step 3: Add the metadata schema types**

Create `include/Runtime/Mix/MixCompileMetadata.h` with focused structs:

```c++
#pragma once

#include "Runtime/Support/Types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::runtime {

struct MixCompileMetadataEntries {
  std::string aic;
  std::string aiv;
};

struct MixCompileMetadataGenerated {
  std::string sourcePath;
};

struct MixCompileMetadataDeviceCompile {
  std::string aicArch;
  std::string aivArch;
  std::vector<std::string> aicDefinitions;
  std::vector<std::string> aivDefinitions;
};

struct MixCompileMetadataArtifacts {
  std::string deviceObjectPath;
  std::string packedSharedObjectPath;
  std::string tilingFilePath;
  std::string launchInfoFilePath;
};

struct MixCompileMetadataTensorDesc {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::string runtimeFile;
};

struct MixCompileMetadataAbi {
  std::string workspaceMode;
  uint64_t workspaceBytes = 0;
  std::string tilingMode;
  std::string tilingSource;
  std::vector<MixCompileMetadataTensorDesc> inputs;
  std::vector<MixCompileMetadataTensorDesc> outputs;
};

struct MixCompileMetadataHostLaunch {
  std::string mode;
  std::string helperKind;
  std::string helperInputsJson;
};

struct MixCompileMetadata {
  uint64_t schemaVersion = 0;
  std::string kernelKind;
  std::string kernelName;
  std::string runtimeKernelName;
  std::string socVersion;
  std::string mixKernelType;
  std::string launcherSymbol;
  MixCompileMetadataEntries entries;
  MixCompileMetadataGenerated generated;
  MixCompileMetadataDeviceCompile deviceCompile;
  MixCompileMetadataArtifacts artifacts;
  MixCompileMetadataAbi abi;
  MixCompileMetadataHostLaunch hostLaunch;
};

llvm::Expected<MixCompileMetadata>
parseMixCompileMetadataJson(llvm::StringRef jsonText);

llvm::Expected<std::string>
serializeMixCompileMetadataJson(const MixCompileMetadata &metadata);

} // namespace mlir::runtime
```

- [ ] **Step 4: Implement JSON parse/serialize**

Create `lib/Runtime/Mix/MixCompileMetadata.cpp` using `llvm::json` helpers. Enforce:
- required top-level fields
- `kernel_kind == "mix"`
- arrays parse cleanly
- unknown fields ignored

Keep `helperInputsJson` as an opaque JSON string in Phase 1 instead of introducing subtype-specific structs.

- [ ] **Step 5: Wire the new file into the runtime library**

Modify `lib/Runtime/CMakeLists.txt` to compile `MixCompileMetadata.cpp` into the runtime library target that already contains the mix backend.

- [ ] **Step 6: Run the test and verify it passes**

Run:

```bash
cmake --build build --target test_runtime -j8
ctest --test-dir build --output-on-failure -R test_runtime
```

Expected: new metadata parse tests pass.

- [ ] **Step 7: Commit**

```bash
git add include/Runtime/Mix/MixCompileMetadata.h lib/Runtime/Mix/MixCompileMetadata.cpp lib/Runtime/CMakeLists.txt test/tools/runtime/test_runtime.cpp
git commit -m "runtime: add mix compile metadata schema"
```

### Task 2: Emit Metadata from MixDirectBackend Without Breaking Existing Outputs

**Files:**
- Modify: `lib/Runtime/Mix/MixDirectBackend.cpp`
- Modify: `include/Runtime/Mix/MixArtifact.h`
- Test: `test/tools/runtime/test_runtime.cpp`

- [ ] **Step 1: Add a failing compile-output test**

Extend `test/tools/runtime/test_runtime.cpp` with a focused check that a mix artifact root now contains:
- legacy `out/manifest.txt`
- new `out/mix_metadata.json`
- legacy manifest line `metadata_path=...`

If an end-to-end compile fixture already exists, extend it; otherwise add a narrow helper that inspects an existing checked-in sample artifact root.

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cmake --build build --target test_runtime -j8
ctest --test-dir build --output-on-failure -R test_runtime
```

Expected: failure because `mix_metadata.json` is not emitted yet.

- [ ] **Step 3: Extend `MixArtifact` with metadata path**

Modify `include/Runtime/Mix/MixArtifact.h`:

```c++
  std::string metadata_path;
```

Keep all existing fields unchanged.

- [ ] **Step 4: Build metadata from current compile outputs**

In `MixDirectBackend::compile()` build `MixCompileMetadata` from existing values already computed in the function:
- analyzer output
- parsed generated config
- ABI data
- artifact paths
- helper/legacy runner mode

Do not derive any field from matmul-specific assumptions except the current helper payload, which stays opaque under `host_launch.helper_inputs_json`.

- [ ] **Step 5: Write `out/mix_metadata.json` and append `metadata_path` to legacy manifest**

In `MixDirectBackend.cpp`:
- write `out/mix_metadata.json`
- keep `out/manifest.txt` format
- append `metadata_path=out/mix_metadata.json`

Do not remove or rename any existing manifest field in this task.

- [ ] **Step 6: Run the tests and verify they pass**

Run:

```bash
cmake --build build --target test_runtime -j8
ctest --test-dir build --output-on-failure -R test_runtime
```

Expected: metadata file exists and legacy manifest still loads.

- [ ] **Step 7: Commit**

```bash
git add include/Runtime/Mix/MixArtifact.h lib/Runtime/Mix/MixDirectBackend.cpp test/tools/runtime/test_runtime.cpp
git commit -m "runtime: emit mix compile metadata"
```

### Task 3: Keep Artifact Loading Backward Compatible

**Files:**
- Modify: `lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp`
- Modify: `include/Runtime/Execution/TaskGraph.h`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Add a failing artifact-loader compatibility test**

Add focused tests in `test/tools/runtime/test_taskgraph_runtime.cpp` covering:
- artifact root with legacy manifest only
- artifact root with legacy manifest plus `metadata_path`
- missing `mix_metadata.json` while `metadata_path` is present should fail loudly

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j8
ctest --test-dir build --output-on-failure -R test_taskgraph_runtime
```

Expected: failure because `metadata_path` is not yet loaded/validated.

- [ ] **Step 3: Extend artifact loading to capture metadata opportunistically**

Modify `RuntimeSessionRequestBuilder.cpp`:
- parse `metadata_path` if present
- resolve it relative to artifact root
- validate that the JSON file exists
- do not require it for non-mix kernels
- do not require it for legacy mix artifacts that do not advertise it

If `KernelArtifact` already has an extensible field for metadata path, use it; otherwise add a narrow optional field in `TaskGraph.h`-owned artifact structs only if needed.

- [ ] **Step 4: Keep all existing manifest resolution rules**

Do not change:
- accepted manifest locations
- `kernel_so_path` behavior
- `device_binary_path` fallback rules
- default `mix_resource_type` rules

This task is additive only.

- [ ] **Step 5: Run tests and verify compatibility**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j8
ctest --test-dir build --output-on-failure -R test_taskgraph_runtime
```

Expected: legacy artifact loading still passes; advertised metadata path is validated.

- [ ] **Step 6: Commit**

```bash
git add lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp include/Runtime/Execution/TaskGraph.h test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: preserve mix artifact loader compatibility"
```

### Task 4: Lock the Caller Compatibility Matrix

**Files:**
- Create: `test/tools/runtime/test_mix_metadata_contract.cpp`
- Modify: `test/tools/runtime/CMakeLists.txt`
- Test: `test/tools/runtime/run_runtime.sh`

- [ ] **Step 1: Add a failing focused compatibility test target**

Create `test/tools/runtime/test_mix_metadata_contract.cpp` that verifies:
- `runtime-session` can still consume a mix artifact root built before metadata adoption
- `runtime-session` can consume a mix artifact root with metadata enabled
- `SimBackend` and `NpuBackend`-facing manifest fields expected by `KernelArtifact` remain unchanged

- [ ] **Step 2: Run the focused test target and verify it fails**

Run:

```bash
cmake --build build --target test_mix_metadata_contract -j8
ctest --test-dir build --output-on-failure -R test_mix_metadata_contract
```

Expected: failure because the test target and checks do not exist yet.

- [ ] **Step 3: Register the new test target**

Modify `test/tools/runtime/CMakeLists.txt` to build and register `test_mix_metadata_contract`.

- [ ] **Step 4: Implement compatibility assertions**

In `test_mix_metadata_contract.cpp`, assert:
- old fields still exist in manifest
- new metadata path is optional for legacy artifacts
- new metadata path is stable for newly compiled mix artifacts

Do not add assertions that require callers to switch to JSON yet.

- [ ] **Step 5: Run focused runtime verification**

Run:

```bash
cmake --build build --target test_mix_metadata_contract test_runtime test_taskgraph_runtime runtime-session -j8
ctest --test-dir build --output-on-failure -R 'test_mix_metadata_contract|test_runtime|test_taskgraph_runtime'
bash test/tools/runtime/run_runtime.sh
```

Expected: all focused runtime checks pass.

- [ ] **Step 6: Commit**

```bash
git add test/tools/runtime/CMakeLists.txt test/tools/runtime/test_mix_metadata_contract.cpp
git commit -m "test: lock mix metadata compatibility contract"
```

### Task 5: Prepare Phase 2 Consumer Cutover Without Enabling It

**Files:**
- Create: `docs/superpowers/specs/2026-04-14-runtime-mix-metadata-consumer-cutover-design.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Write the Phase 2 cutover spec**

Create `docs/superpowers/specs/2026-04-14-runtime-mix-metadata-consumer-cutover-design.md` documenting:
- which legacy fields become redundant after metadata adoption
- how `runtime-session`, `SimBackend`, and `NpuBackend` will prefer JSON metadata
- what fallback behavior remains during transition

- [ ] **Step 2: Record compatibility rule in AGENTS**

Append a short decision to `AGENTS.md` stating:
- `mix` metadata migration must preserve current caller-facing CLI and artifact-root behavior until explicit cutover is approved

- [ ] **Step 3: Review plan coverage**

Verify this plan covers:
- general metadata schema
- backward-compatible artifact emission
- backward-compatible artifact loading
- focused tests for caller compatibility

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-04-14-runtime-mix-metadata-consumer-cutover-design.md AGENTS.md
git commit -m "docs: define phase 2 mix metadata cutover"
```

### Verification Summary

- [ ] `test_runtime` passes with metadata parse coverage
- [ ] `test_taskgraph_runtime` passes with artifact-loader compatibility coverage
- [ ] `test_mix_metadata_contract` passes with caller compatibility coverage
- [ ] `bash test/tools/runtime/run_runtime.sh` passes on xvm after Phase 1
- [ ] new mix artifacts contain both `out/manifest.txt` and `out/mix_metadata.json`
- [ ] legacy mix artifact roots remain runnable without requiring JSON metadata
