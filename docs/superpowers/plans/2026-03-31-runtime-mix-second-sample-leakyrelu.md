# RuntimeMix Second Sample LeakyRelu Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate that RuntimeMix’s current artifact-backed tiling and direct-packed execution flow also works for a second, structurally different mix sample based on `fc_leakyrelu_mix.cpp`.

**Architecture:** Add one focused `RuntimeMix` validation entry for the leakyrelu mix sample, reuse the existing compile/artifact/manifest/direct-packed/runner mechanisms, and prove on xvm that both execution paths consume the same artifact-backed tiling and pass accuracy checks. Keep the current effort explicitly sample-scoped; do not generalize all mix kernels in this step.

**Tech Stack:** C++17, LLVM Support, RuntimeMix direct backend, xvm simulator, Python data scripts, bash validation glue.

---

## File Structure

- Create: `examples/leakyrelu-mix-test/README.md`
  Purpose: document the second RuntimeMix sample entrypoint and current accepted ABI/inputs.

- Create: `examples/leakyrelu-mix-test/run.sh`
  Purpose: build RuntimeMix tools, compile the leakyrelu kernel, generate sample data, validate via runner and direct-packed paths.

- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`
  Purpose: ensure the backend can describe the leakyrelu sample’s ABI, filenames, and real tiling artifact without reintroducing hardcoded baremix-only assumptions.

- Modify: `tools/mix-validator/mix_validator_main.cpp`
  Purpose: ensure direct-packed validation can consume the leakyrelu sample’s manifest-described ABI using the existing artifact-backed tiling path, without a second hardcoded sample table.

- Reuse: `examples/matmul-add-relu-sum/fc_leakyrelu_mix.cpp`
  Purpose: the actual second mix kernel under test.

- Reuse/Glue: `examples/matmul-add-relu-sum/gen_data_leakyrelu.py`
  Purpose: sample data generation source of truth, either used directly or wrapped minimally.

## Task 1: Define the LeakyRelu Sample Entry

**Files:**
- Create: `examples/leakyrelu-mix-test/README.md`
- Create: `examples/leakyrelu-mix-test/run.sh`

- [ ] **Step 1: Inspect the current leakyrelu sample assets before writing glue**

Run:

```bash
rg -n "fc_leakyrelu_mix|gen_data_leakyrelu|run_leakyrelu|tiling" \
  examples/matmul-add-relu-sum
```

Expected:
- identify the kernel source file
- identify the data generation script
- identify any existing run/tiling helper scripts worth reusing minimally

- [ ] **Step 2: Write a minimal README for the new sample**

Create `examples/leakyrelu-mix-test/README.md` with concise content shaped like:

```md
# leakyrelu-mix-test

`leakyrelu-mix-test` 是 `RuntimeMix` 的第二个 mix 验证样例，目标是验证当前 artifact-backed tiling 机制不只适用于 baremix。

内核来源：
- `examples/matmul-add-relu-sum/fc_leakyrelu_mix.cpp`

运行方式（仅 xvm）：
```bash
cd /home/niu/code/Ascend-MLIR
bash examples/leakyrelu-mix-test/run.sh
```
```

- [ ] **Step 3: Create a minimal run script skeleton**

Create `examples/leakyrelu-mix-test/run.sh` with the same overall shape as `examples/baremix-test/run.sh`, but targeting `fc_leakyrelu_mix.cpp`. The first draft should include:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SOC_VERSION=${SOC_VERSION:-Ascend910B1}
BOOTSTRAP_BUILD_DIR="${REPO_ROOT}/build/runtime-mix-bootstrap"
ARTIFACT_DIR="${REPO_ROOT}/build/runtime-mix-leakyrelu"
```

Do not wire all commands yet in this step; just establish the entrypoint and naming.

- [ ] **Step 4: Commit the sample entry skeleton**

```bash
git add examples/leakyrelu-mix-test/README.md examples/leakyrelu-mix-test/run.sh
git commit -m "feat: add leakyrelu RuntimeMix sample entry"
```

## Task 2: Adapt RuntimeMix Metadata for the LeakyRelu Sample

**Files:**
- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`

- [ ] **Step 1: Identify which ABI fields are currently still baremix-shaped**

Run:

```bash
rg -n "buildCurrentBaremixAbi|x1_gm.bin|x2_gm.bin|bias.bin|output.bin|generated_file" \
  lib/RuntimeMix/MixDirectBackend.cpp
```

Expected:
- locate the current sample-specific ABI construction point

- [ ] **Step 2: Generalize the current sample-specific metadata just enough for a second sample**

In `lib/RuntimeMix/MixDirectBackend.cpp`, replace the single baremix-only metadata helper with a narrow sample-aware helper. The shape should be:

```cpp
static llvm::Expected<SampleAbiMetadata>
buildCurrentSampleAbi(llvm::StringRef kernelName,
                      llvm::StringRef tilingMode,
                      llvm::StringRef tilingSource);
```

It must support at least:
- `baremix_custom`
- `fc_leakyrelu_mix` (or the exact selected kernel symbol name)

Do not invent a plugin system or generic schema registry here.

- [ ] **Step 3: Encode the leakyrelu sample’s real filenames, dtypes, shapes, workspace, and tiling metadata**

The leakyrelu sample metadata must be explicit and artifact-driven, just like baremix. Populate:
- input names/files
- output name/file
- dtype/shape
- workspace bytes
- `abi_tiling_mode=generated_file`
- `abi_tiling_source=out/tiling.bin`

Use the actual leakyrelu sample data conventions from its generation script and existing files; do not guess names if the sample already defines them.

- [ ] **Step 4: Ensure runner tiling artifact emission still works for the new sample**

The same backend path that now emits `out/tiling.bin` for baremix must also do so for the leakyrelu sample. No second tiling emission mechanism is allowed.

- [ ] **Step 5: Commit the sample-aware metadata update**

```bash
git add lib/RuntimeMix/MixDirectBackend.cpp
git commit -m "feat: add leakyrelu sample metadata to RuntimeMix backend"
```

## Task 3: Wire the LeakyRelu Run Script End-to-End

**Files:**
- Modify: `examples/leakyrelu-mix-test/run.sh`

- [ ] **Step 1: Build RuntimeMix tools the same way as baremix**

Update `examples/leakyrelu-mix-test/run.sh` so it bootstraps or reuses:
- `mix-compiler`
- `mix-validator`

Use the same reuse-first pattern already established in `examples/baremix-test/run.sh`.

- [ ] **Step 2: Compile the leakyrelu kernel through RuntimeMix**

In `examples/leakyrelu-mix-test/run.sh`, add the compile step:

```bash
"${BOOTSTRAP_BUILD_DIR}/bin/mix-compiler" \
  --kernel "${REPO_ROOT}/examples/matmul-add-relu-sum/fc_leakyrelu_mix.cpp" \
  --name fc_leakyrelu_mix \
  --output "${ARTIFACT_DIR}" \
  --soc "${SOC_VERSION}"
```

If the actual exported kernel name differs, use the real one discovered from the sample source.

- [ ] **Step 3: Generate input/golden data using the existing leakyrelu script**

Wire the script to call the existing data generator, for example:

```bash
python3 "${REPO_ROOT}/examples/matmul-add-relu-sum/gen_data_leakyrelu.py"
```

If that script writes to a location incompatible with the new artifact layout, add only the smallest glue needed to place inputs/golden under:

```bash
${ARTIFACT_DIR}/testdata/input
${ARTIFACT_DIR}/testdata/output
```

- [ ] **Step 4: Run the validator through the normal path**

Add the normal validation step:

```bash
"${BOOTSTRAP_BUILD_DIR}/bin/mix-validator" \
  --artifact-root "${ARTIFACT_DIR}" \
  --input-dir "${ARTIFACT_DIR}/testdata/input" \
  --golden "${ARTIFACT_DIR}/testdata/output/golden.bin" \
  --output-file "${ARTIFACT_DIR}/testdata/output/actual.bin" \
  --soc "${SOC_VERSION}"
```

- [ ] **Step 5: Add the forced direct-packed comparison step**

In the same script, add:

```bash
"${BOOTSTRAP_BUILD_DIR}/bin/mix-validator" \
  --artifact-root "${ARTIFACT_DIR}" \
  --input-dir "${ARTIFACT_DIR}/testdata/input" \
  --golden "${ARTIFACT_DIR}/testdata/output/golden.bin" \
  --output-file "${ARTIFACT_DIR}/testdata/output/direct-actual.bin" \
  --soc "${SOC_VERSION}" \
  --force-direct-packed
```

- [ ] **Step 6: Commit the leakyrelu sample runner**

```bash
git add examples/leakyrelu-mix-test/run.sh
git commit -m "feat: wire leakyrelu RuntimeMix validation flow"
```

## Task 4: Verify the LeakyRelu Sample on xvm

**Files:**
- Test only

- [ ] **Step 1: Run the new sample end-to-end on xvm**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  bash examples/leakyrelu-mix-test/run.sh
'
```

Expected:
- compile succeeds
- runner-backed validation succeeds
- direct-packed validation succeeds

- [ ] **Step 2: Confirm the artifact-backed tiling is present**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  ls -l build/runtime-mix-leakyrelu/out/tiling.bin &&
  grep -n "^abi_tiling_" build/runtime-mix-leakyrelu/out/manifest.txt
'
```

Expected:
- `tiling.bin` exists
- manifest says `generated_file`

- [ ] **Step 3: Compare md5 for runner and direct-packed outputs**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  md5sum \
    build/runtime-mix-leakyrelu/testdata/output/golden.bin \
    build/runtime-mix-leakyrelu/testdata/output/actual.bin \
    build/runtime-mix-leakyrelu/testdata/output/direct-actual.bin
'
```

Expected:
- all md5 values match

- [ ] **Step 4: If the sample fails, capture where it failed**

If the leakyrelu sample does not pass, the handoff must explicitly classify the failure as one of:
- metadata/ABI mismatch
- tiling artifact mismatch
- packed mix execution mismatch

Do not stop at “the second sample failed”.

- [ ] **Step 5: Commit the verified second-sample state**

```bash
git add -A
git commit -m "test: validate leakyrelu mix sample on RuntimeMix"
```

## Self-Review

- Spec coverage:
  - second sample entry is covered by Task 1
  - sample-specific metadata adaptation is covered by Task 2
  - end-to-end runtime flow is covered by Task 3
  - xvm validation of runner + direct packed + artifact-backed tiling is covered by Task 4

- Placeholder scan:
  - No `TBD` / `TODO`
  - All tasks contain exact files and commands

- Type consistency:
  - The plan consistently refers to `fc_leakyrelu_mix.cpp`, `tiling.bin`, `mix-compiler`, `mix-validator`, and `--force-direct-packed`
