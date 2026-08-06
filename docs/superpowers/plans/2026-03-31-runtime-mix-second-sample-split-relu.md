# RuntimeMix Second Sample Split ReLU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate that RuntimeMix’s current artifact-backed tiling and direct-packed execution flow also works for a second, structurally different mix sample based on `fc_relu_split_mix.cpp`.

**Architecture:** Add one focused `RuntimeMix` validation entry for the split-relu mix sample, reuse the existing compile/artifact/manifest/direct-packed/runner mechanisms, and prove on xvm that both execution paths consume the same artifact-backed tiling and pass accuracy checks. Keep the current effort explicitly sample-scoped; do not generalize all mix kernels in this step.

**Tech Stack:** C++17, LLVM Support, RuntimeMix direct backend, xvm simulator, Python data scripts, bash validation glue.

---

## File Structure

- Create: `examples/relu-split-mix-test/README.md`
  Purpose: document the second RuntimeMix sample entrypoint and accepted ABI/runtime-kernel naming.

- Create: `examples/relu-split-mix-test/run.sh`
  Purpose: build RuntimeMix tools, compile the split-relu kernel, generate sample data, validate via runner and direct-packed paths.

- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`
  Purpose: ensure the backend can describe the split-relu sample’s ABI, filenames, runtime kernel name, and real tiling artifact without reintroducing baremix-only assumptions.

- Modify: `tools/mix-validator/mix_validator_main.cpp`
  Purpose: ensure runner/direct-packed validation continues to consume manifest-described ABI for the new sample without a second hardcoded validator table.

- Reuse: `examples/matmul-add-relu-sum/fc_relu_split_mix.cpp`
  Purpose: the actual second mix kernel under test.

- Reuse/Glue: `examples/matmul-add-relu-sum/gen_data_fp16ab_biasn_relu.py`
  Purpose: source of truth for split-relu sample data generation.

- Reuse/Glue: `examples/matmul-add-relu-sum/gen_tiling_leakyrelu.py`
  Purpose: temporary tiling field reference if the split-relu sample still matches the same 128x256x128 / 1 AIC + 2 AIV contract.

## Task 1: Define the Split ReLU Sample Entry

**Files:**
- Create: `examples/relu-split-mix-test/README.md`
- Create: `examples/relu-split-mix-test/run.sh`

- [ ] **Step 1: Inspect the current split-relu sample assets before writing glue**

Run:

```bash
rg -n "fc_relu_split_mix|gen_data_fp16ab_biasn_relu|gen_cube_tiling|tiling" \
  examples/matmul-add-relu-sum
```

Expected:
- identify the kernel source file
- identify the closest data generation script
- identify the closest tiling helper script worth reusing minimally

- [ ] **Step 2: Write a minimal README for the new sample**

Create `examples/relu-split-mix-test/README.md` with concise content shaped like:

```md
# relu-split-mix-test

`relu-split-mix-test` 是 `RuntimeMix` 的第二个 mix 验证样例，目标是验证当前 artifact-backed tiling 机制不只适用于 baremix。

内核来源：
- `examples/matmul-add-relu-sum/fc_relu_split_mix.cpp`

运行方式（仅 xvm）：
```bash
cd /home/niu/code/Ascend-MLIR
bash examples/relu-split-mix-test/run.sh
```

当前运行时 kernel 名固定为：
- `fc_relu_split`
```

- [ ] **Step 3: Create a minimal run script skeleton**

Create `examples/relu-split-mix-test/run.sh` with the same overall shape as `examples/baremix-test/run.sh`, but targeting `fc_relu_split_mix.cpp`. The first draft should include:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
BOOTSTRAP_BUILD_DIR="${BOOTSTRAP_BUILD_DIR:-${REPO_ROOT}/build/runtime-mix-bootstrap}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/build/runtime-mix-relu-split}"
DATA_DIR="${DATA_DIR:-${ARTIFACT_DIR}/testdata}"
KERNEL_SRC="${REPO_ROOT}/examples/matmul-add-relu-sum/fc_relu_split_mix.cpp"
KERNEL_NAME="${KERNEL_NAME:-fc_relu_split}"
```

Do not wire all commands yet in this step; just establish the entrypoint and naming.

- [ ] **Step 4: Commit the sample entry skeleton**

```bash
git add examples/relu-split-mix-test/README.md examples/relu-split-mix-test/run.sh
git commit -m "feat: add split relu RuntimeMix sample entry"
```

## Task 2: Adapt RuntimeMix Metadata for the Split ReLU Sample

**Files:**
- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`

- [ ] **Step 1: Identify which ABI fields and runtime-kernel naming are still sample-specific**

Run:

```bash
rg -n "buildCurrentSampleAbi|baremix_custom|fc_leakyrelu|generated_file|kernel_name=" \
  lib/RuntimeMix/MixDirectBackend.cpp
```

Expected:
- locate the current sample-specific ABI construction point
- locate where runtime kernel name differs from source file name

- [ ] **Step 2: Add the split-relu sample to the narrow sample-aware metadata helper**

In `lib/RuntimeMix/MixDirectBackend.cpp`, extend the current helper so it supports:
- `fc_relu_split`
- `fc_relu_split_mix`
- `auto_gen_fc_relu_split_kernel`

Do not invent a registry system or generic plugin layer here.

- [ ] **Step 3: Encode the split-relu sample’s explicit ABI metadata**

Populate:
- input 0: file `fc_relu_split_input_a.bin`, dtype `f16`, shape `128,256`
- input 1: file `fc_relu_split_input_b.bin`, dtype `f16`, shape `256,128`
- input 2: file `fc_relu_split_input_bias.bin`, dtype `f32`, shape `128`
- output 0: file `fc_relu_split_output.bin`, dtype `f32`, shape `128,128`
- workspace bytes `16777216`
- `abi_tiling_mode=generated_file`
- `abi_tiling_source=out/tiling.bin`

Use the existing split-relu kernel contract and the reused data script as the source of truth.

- [ ] **Step 4: Keep runtime kernel naming explicit**

In the same backend file, ensure the artifact/manifest path exposes:
- `kernel_name=fc_relu_split`
- `requested_kernel_name=<user requested sample name>`

This must be explicit; do not allow one code path to use `fc_relu_split_mix` while another uses `fc_relu_split`.

- [ ] **Step 5: Commit the sample-aware metadata update**

```bash
git add lib/RuntimeMix/MixDirectBackend.cpp
git commit -m "feat: add split relu sample metadata to RuntimeMix backend"
```

## Task 3: Wire the Split ReLU Run Script End-to-End

**Files:**
- Modify: `examples/relu-split-mix-test/run.sh`

- [ ] **Step 1: Build RuntimeMix tools the same way as baremix**

Update `examples/relu-split-mix-test/run.sh` so it bootstraps or reuses:
- `mix-compiler`
- `mix-validator`

Use the same reuse-first pattern already established in `examples/baremix-test/run.sh`.

- [ ] **Step 2: Compile the split-relu kernel through RuntimeMix**

In `examples/relu-split-mix-test/run.sh`, add the compile step:

```bash
"${BOOTSTRAP_BUILD_DIR}/bin/mix-compiler" \
  --kernel "${REPO_ROOT}/examples/matmul-add-relu-sum/fc_relu_split_mix.cpp" \
  --name fc_relu_split \
  --output "${ARTIFACT_DIR}" \
  --soc "${SOC_VERSION}"
```

The runtime kernel name must stay `fc_relu_split`, not `fc_relu_split_mix`.

- [ ] **Step 3: Generate input/golden data using the existing ReLU data script**

Wire the script to call the existing data generator:

```bash
python3 "${REPO_ROOT}/examples/matmul-add-relu-sum/gen_data_fp16ab_biasn_relu.py" \
  "${DATA_DIR}/npy"
```

If that script takes no CLI args, use the smallest wrapper needed so outputs land under:

```bash
${ARTIFACT_DIR}/testdata/npy
${ARTIFACT_DIR}/testdata/input
${ARTIFACT_DIR}/testdata/output
```

- [ ] **Step 4: Convert `.npy` files into the ABI-expected `.bin` layout**

Add a minimal inline Python conversion step shaped like:

```python
mapping = [
    ("input_a.npy", "fc_relu_split_input_a.bin"),
    ("input_b.npy", "fc_relu_split_input_b.bin"),
    ("input_bias.npy", "fc_relu_split_input_bias.bin"),
]
```

Also write:
- `fc_relu_split_output.bin`
- `golden.bin`

under `testdata/output/`.

- [ ] **Step 5: Run the validator through the normal path**

Add the normal validation step:

```bash
"${BOOTSTRAP_BUILD_DIR}/bin/mix-validator" \
  --artifact-root "${ARTIFACT_DIR}" \
  --input-dir "${ARTIFACT_DIR}/testdata/input" \
  --golden "${ARTIFACT_DIR}/testdata/output/golden.bin" \
  --output-file "${ARTIFACT_DIR}/testdata/output/actual.bin" \
  --soc "${SOC_VERSION}"
```

- [ ] **Step 6: Add the forced direct-packed comparison step**

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

- [ ] **Step 7: Add binary comparison and md5 reporting**

Add a final inline Python verification step shaped like:

```python
golden = np.fromfile(golden_path, dtype=np.float32)
actual = np.fromfile(actual_path, dtype=np.float32)
direct = np.fromfile(direct_path, dtype=np.float32)
```

Print:
- `runner_max_abs_diff`
- `runner_mean_abs_diff`
- `direct_max_abs_diff`
- `direct_mean_abs_diff`
- all three md5 values
- final `test pass`

- [ ] **Step 8: Commit the end-to-end run script**

```bash
git add examples/relu-split-mix-test/run.sh
git commit -m "feat: add split relu RuntimeMix validation flow"
```

## Task 4: Validate on xvm and Update Docs

**Files:**
- Modify: `examples/relu-split-mix-test/README.md`

- [ ] **Step 1: Rebuild tool binaries on xvm so the latest RuntimeMix code is used**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  rm -f build/runtime-mix-bootstrap/bin/mix-compiler \
        build/runtime-mix-bootstrap/bin/mix-validator
'
```

Expected:
- old bootstrap binaries removed successfully

- [ ] **Step 2: Run the new split-relu sample entry on xvm**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  bash examples/relu-split-mix-test/run.sh
'
```

Expected:
- compile succeeds
- `mix-validator` prints `PASS`
- final script prints `test pass`

- [ ] **Step 3: Confirm artifact-backed tiling and outputs exist**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  ls -l build/runtime-mix-relu-split/out/tiling.bin \
        build/runtime-mix-relu-split/testdata/output/golden.bin \
        build/runtime-mix-relu-split/testdata/output/actual.bin \
        build/runtime-mix-relu-split/testdata/output/direct-actual.bin
'
```

Expected:
- all four files exist

- [ ] **Step 4: Compare md5 for runner and direct-packed outputs**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  md5sum \
    build/runtime-mix-relu-split/testdata/output/golden.bin \
    build/runtime-mix-relu-split/testdata/output/actual.bin \
    build/runtime-mix-relu-split/testdata/output/direct-actual.bin
'
```

Expected:
- all md5 values match

- [ ] **Step 5: Update README with the verified runtime kernel name and output layout**

Make sure `examples/relu-split-mix-test/README.md` explicitly documents:
- source file path
- runtime kernel name `fc_relu_split`
- output directory `build/runtime-mix-relu-split`
- runner/direct-packed dual validation

- [ ] **Step 6: Commit docs and validation entry**

```bash
git add examples/relu-split-mix-test/README.md
git commit -m "docs: document split relu RuntimeMix sample"
```
