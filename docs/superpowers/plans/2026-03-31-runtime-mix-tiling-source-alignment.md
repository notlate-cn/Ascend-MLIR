# RuntimeMix Tiling Source Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `RuntimeMix` direct packed validation consume the same tiling result as the working runner path, so baremix direct packed execution stops diverging from runner execution.

**Architecture:** First turn the runner’s real tiling output into an explicit artifact and manifest contract, then update direct packed validation to load that exact tiling artifact instead of synthesizing a separate hardcoded blob. Validate on xvm that runner-backed and direct-packed paths both pass with the same artifact inputs.

**Tech Stack:** C++17, LLVM Support, RuntimeMix direct backend, manifest text format, xvm simulator, baremix acceptance flow.

---

## File Structure

- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`
  Purpose: emit a real tiling artifact from the same generation path the runner uses today, and record it in the manifest.

- Modify: `include/RuntimeMix/MixArtifact.h`
  Purpose: only if needed to expose the real tiling artifact path at the artifact boundary.

- Modify: `tools/mix-validator/mix_validator_main.cpp`
  Purpose: stop synthesizing `buildBaremixTiling()` for the direct packed path and instead load the artifact-backed tiling file referenced by manifest metadata.

- Modify: `examples/baremix-test/README.md`
  Purpose: document that runner and direct packed now consume the same artifact tiling.

- Test/Verify: `examples/baremix-test/run.sh`
  Purpose: normal acceptance must remain passing on xvm.

## Task 1: Emit Real Tiling as an Artifact

**Files:**
- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`
- Modify: `include/RuntimeMix/MixArtifact.h` only if needed

- [ ] **Step 1: Find the exact point where runner tiling is currently generated**

Confirm in `lib/RuntimeMix/MixDirectBackend.cpp` that runner tiling comes from the generated `GenerateTiling(...)` source and not from a manifest blob. Use these references:

```bash
rg -n "GenerateTiling|tiling_source|abi_tiling_mode|abi_tiling_source" \
  lib/RuntimeMix/MixDirectBackend.cpp
```

Expected:
- `GenerateTiling(...)` source generation is identified
- current manifest still says `abi_tiling_mode=fixed_bytes`
- current manifest still says `abi_tiling_source=baremix_fixed_blob`

- [ ] **Step 2: Add a concrete tiling artifact output path under the build artifact**

In `lib/RuntimeMix/MixDirectBackend.cpp`, define a stable artifact location for the generated tiling bytes, for example:

```cpp
const std::string tilingArtifactPath =
    joinPath(outDir.str(), "tiling.bin");
```

or another clearly named file under the final artifact tree.

The file must live in the produced artifact, not under a temporary-only work directory.

- [ ] **Step 3: Generate the tiling bytes through the same backend path the runner uses**

Still in `lib/RuntimeMix/MixDirectBackend.cpp`, add the minimal logic needed to materialize the real tiling bytes into that artifact file. The source of truth must remain the same `GenerateTiling(...)` logic currently used by the runner path.

It is acceptable to implement this by:
- generating the tiling source as today
- building a small helper executable or using an existing generated binary path
- writing the resulting bytes to `tiling.bin`

It is **not** acceptable to:
- copy the current `buildBaremixTiling()` hardcoded blob into the backend
- keep `tiling.bin` as a second independent implementation

- [ ] **Step 4: Update manifest metadata to describe the real tiling artifact**

Replace the old manifest semantics:

```text
abi_tiling_mode=fixed_bytes
abi_tiling_source=baremix_fixed_blob
```

with artifact-backed semantics such as:

```text
abi_tiling_mode=generated_file
abi_tiling_source=out/tiling.bin
```

Use the actual chosen path, and keep it relative to the artifact root if that is already the manifest convention.

- [ ] **Step 5: If needed, expose the tiling artifact through `MixArtifact`**

Only if consumers truly need a dedicated field, extend `include/RuntimeMix/MixArtifact.h` with something like:

```cpp
std::string tiling_artifact_path;
```

If the manifest path remains the only required source of truth, keep `MixArtifact` unchanged and document that decision in the task handoff.

- [ ] **Step 6: Verify artifact emission on xvm**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  rm -f build/runtime-mix-bootstrap/bin/mix-compiler &&
  bash examples/baremix-test/run.sh >/tmp/runtime-mix-tiling-task1.log &&
  ls -l build/runtime-mix-baremix/out &&
  grep -n "^abi_tiling_" build/runtime-mix-baremix/out/manifest.txt
'
```

Expected:
- `tiling.bin` (or chosen file) exists in the artifact
- manifest no longer claims `baremix_fixed_blob`

- [ ] **Step 7: Commit the tiling artifact emission**

```bash
git add lib/RuntimeMix/MixDirectBackend.cpp include/RuntimeMix/MixArtifact.h
git commit -m "feat: emit artifact-backed mix tiling"
```

## Task 2: Make Direct Packed Validation Load the Artifact Tiling

**Files:**
- Modify: `tools/mix-validator/mix_validator_main.cpp`

- [ ] **Step 1: Remove the manifest-to-hardcoded-blob coupling**

In `tools/mix-validator/mix_validator_main.cpp`, replace the current logic:

```cpp
if (abi.tilingMode == "fixed_bytes" &&
    abi.tilingSource == "baremix_fixed_blob")
  return buildBaremixTiling();
```

with logic that reads the real tiling artifact referenced by `abi_tiling_source`.

- [ ] **Step 2: Add a narrow helper to load tiling bytes from the artifact**

Implement a helper shaped like:

```cpp
static llvm::Expected<std::vector<uint8_t>>
loadTilingFromAbi(const AbiMetadata &abi, const std::string &artifactRoot) {
  if (abi.tilingMode != "generated_file")
    return llvm::createStringError(...);
  const std::string path = resolvePath(artifactRoot, abi.tilingSource);
  return readBinary(path);
}
```

Keep it narrow. Do not invent a generic tiling schema system here.

- [ ] **Step 3: Thread artifact root into direct packed tiling construction**

Update the direct packed path so `args.tiling` is built from the artifact-backed helper instead of `buildBaremixTiling()`. The call site should look like:

```cpp
auto tilingOr = loadTilingFromAbi(abi, artifactRoot);
if (!tilingOr)
  return tilingOr.takeError();
args.tiling = std::move(*tilingOr);
```

Use the actual `artifactRoot` already available in the validator flow.

- [ ] **Step 4: Verify the old hardcoded tiling blob path is gone**

Run:

```bash
rg -n "buildBaremixTiling|baremix_fixed_blob|fixed_bytes" \
  tools/mix-validator/mix_validator_main.cpp
```

Expected:
- either no matches
- or only legacy comments that are clearly no longer on the execution path

- [ ] **Step 5: Commit the direct packed tiling alignment**

```bash
git add tools/mix-validator/mix_validator_main.cpp
git commit -m "fix: load direct packed tiling from artifact"
```

## Task 3: Prove Direct Packed and Runner Now Match on xvm

**Files:**
- Test only

- [ ] **Step 1: Re-run the normal acceptance path**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  rm -f build/runtime-mix-bootstrap/bin/mix-compiler build/runtime-mix-bootstrap/bin/mix-validator &&
  bash examples/baremix-test/run.sh
'
```

Expected:
- `PASS`
- `test pass`

- [ ] **Step 2: Re-run forced direct packed validation**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  timeout 25 ./build/runtime-mix-bootstrap/bin/mix-validator \
    --artifact-root build/runtime-mix-baremix \
    --input-dir build/runtime-mix-baremix/testdata/input \
    --golden build/runtime-mix-baremix/testdata/output/golden.bin \
    --output-file build/runtime-mix-baremix/testdata/output/direct-actual.bin \
    --soc Ascend910B1 \
    --force-direct-packed; \
  echo EXIT:$?
'
```

Expected:
- no timeout
- `PASS`
- `EXIT:0`

- [ ] **Step 3: Verify precision parity**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  python3 examples/baremix-test/scripts/verify_result.py \
    build/runtime-mix-baremix/testdata/output/direct-actual.bin \
    build/runtime-mix-baremix/testdata/output/golden.bin &&
  md5sum \
    build/runtime-mix-baremix/testdata/output/golden.bin \
    build/runtime-mix-baremix/testdata/output/direct-actual.bin
'
```

Expected:
- `test pass`
- identical md5 values

- [ ] **Step 4: If direct packed still fails, record the new narrowed failure**

If the direct packed path still fails after tiling alignment, the task handoff must explicitly state:

```text
Tiling source mismatch has been eliminated, so the remaining root cause is no longer explained by tiling divergence.
```

Only then is it valid to reopen `Executor` execution-model investigation.

- [ ] **Step 5: Commit the verified xvm alignment state**

```bash
git add -A
git commit -m "test: verify artifact-backed tiling alignment on xvm"
```

## Task 4: Update Documentation

**Files:**
- Modify: `examples/baremix-test/README.md`

- [ ] **Step 1: Update README to describe the new tiling contract**

Replace the old wording:

```md
- `tiling`: 当前路径使用固定字节 blob，不做动态 tiling 协商
```

with wording like:

```md
- `tiling`: 当前路径使用 artifact 中真实生成的 tiling 结果；runner 和 direct packed 读取同一份 tiling 来源
```

- [ ] **Step 2: Mention the direct packed debug path explicitly**

Add a note that `--force-direct-packed` is the way to validate that the runtime path consumes the same tiling artifact as the runner-backed path.

- [ ] **Step 3: Commit the README update**

```bash
git add examples/baremix-test/README.md
git commit -m "docs: describe shared tiling artifact for baremix"
```

## Self-Review

- Spec coverage:
  - real tiling artifact emission is covered by Task 1
  - direct packed tiling consumption is covered by Task 2
  - xvm proof for runner/direct parity is covered by Task 3
  - README update is covered by Task 4

- Placeholder scan:
  - No `TBD` / `TODO`
  - Every task includes concrete files and commands

- Type consistency:
  - The plan consistently uses `abi_tiling_mode`, `abi_tiling_source`, `tiling.bin`, `--force-direct-packed`, and artifact-root based tiling loading terminology
