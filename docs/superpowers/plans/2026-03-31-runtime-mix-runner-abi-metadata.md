# RuntimeMix Runner ABI Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the current baremix runner ABI from scattered hardcoded constants into explicit artifact/manifest metadata so both the generated runner and the validator fallback consume the same ABI description.

**Architecture:** Keep the current baremix acceptance surface unchanged, but introduce a small ABI metadata model at the `RuntimeMix` library layer. `MixDirectBackend` becomes the single producer of that metadata, writes it into the artifact/manifest, and injects it into runner generation. `mix-validator` then reads the same metadata instead of duplicating baremix file/shape/dtype/workspace/tiling assumptions.

**Tech Stack:** C++17, LLVM Support, RuntimeMix direct backend, manifest text format, xvm simulator verification.

---

## File Map

- Modify: `include/RuntimeMix/MixArtifact.h`
  Purpose: carry explicit ABI metadata fields or references to them at the artifact boundary.

- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`
  Purpose: define the current baremix ABI metadata in one place, write it to the manifest, and generate the runner from metadata rather than scattered constants.

- Modify: `tools/mix-validator/mix_validator_main.cpp`
  Purpose: parse the same ABI metadata from the manifest and use it in the direct fallback path instead of duplicating baremix constants.

- Modify: `examples/baremix-test/README.md`
  Purpose: document that the current baremix path is now metadata-driven rather than template-hardcoded.

- Verify: `examples/baremix-test/run.sh`
  Purpose: xvm acceptance entrypoint only.

- Create: `docs/superpowers/specs/2026-03-31-runtime-mix-runner-abi-metadata-design.md`
  Purpose: already written spec for this phase.

- Create: `docs/superpowers/plans/2026-03-31-runtime-mix-runner-abi-metadata.md`
  Purpose: this implementation plan.

## Task 1: Define and Emit Explicit ABI Metadata

**Files:**
- Modify: `include/RuntimeMix/MixArtifact.h`
- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`
- Test: xvm compile/artifact generation plus manifest inspection

- [ ] **Step 1: Add explicit ABI metadata fields or storage handles to `MixArtifact`**

Extend `include/RuntimeMix/MixArtifact.h` so the artifact boundary can represent the current baremix ABI beyond coarse counts. Keep the model small and focused. For example:

```cpp
struct MixArtifact {
  ...
  std::string abi_metadata_path;
};
```

or, if you keep everything in the manifest only, add comments that `manifest_path` now contains the authoritative detailed ABI metadata and no separate baremix constants may exist elsewhere.

The key requirement is that the artifact boundary clearly identifies where the ABI description lives.

- [ ] **Step 2: Add a single baremix ABI metadata builder in `MixDirectBackend.cpp`**

In `lib/RuntimeMix/MixDirectBackend.cpp`, introduce one focused helper that defines the current baremix ABI in one place. For example:

```cpp
struct BaremixAbiMetadata {
  struct TensorDesc {
    std::string name;
    std::string file;
    std::string dtype;
    std::vector<int64_t> shape;
  };

  std::vector<TensorDesc> inputs;
  std::vector<TensorDesc> outputs;
  size_t workspaceBytes;
  std::string workspaceMode;
  std::string tilingMode;
  std::string tilingSource;
};

static BaremixAbiMetadata buildCurrentBaremixAbi();
```

Requirements:
- define current inputs explicitly (`x1_gm.bin`, `x2_gm.bin`, `bias.bin`)
- define current output explicitly
- define current workspace bytes and mode
- define current tiling mode/source
- do not scatter the same shape/file/dtype constants elsewhere after this helper exists

- [ ] **Step 3: Emit detailed ABI metadata into the manifest**

Still in `lib/RuntimeMix/MixDirectBackend.cpp`, expand manifest emission so the current ABI details are explicit and parseable. Use a stable flat key format, for example:

```cpp
manifest += "abi_input_count=3\n";
manifest += "abi_input0_name=x1\n";
manifest += "abi_input0_file=x1_gm.bin\n";
manifest += "abi_input0_dtype=f16\n";
manifest += "abi_input0_shape=128,256\n";
...
manifest += "abi_output0_name=y\n";
manifest += "abi_output0_file=output.bin\n";
manifest += "abi_output0_dtype=f32\n";
manifest += "abi_output0_shape=128,128\n";
manifest += "abi_workspace_bytes=16777216\n";
manifest += "abi_workspace_mode=fixed\n";
manifest += "abi_tiling_mode=fixed_bytes\n";
manifest += "abi_tiling_source=baremix_fixed_blob\n";
```

Keep the existing coarse fields (`abi_inputs`, `abi_outputs`, etc.) if useful, but the new detailed fields become the source of truth.

- [ ] **Step 4: Verify manifest contents on xvm**

Run:

```bash
sleep 2 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  rm -f build/runtime-mix-bootstrap/bin/mix-compiler &&
  bash examples/baremix-test/run.sh &&
  rg -n "^abi_" build/runtime-mix-baremix/out/manifest.txt
'
```

Expected:
- `run.sh` still passes
- manifest contains detailed `abi_input*`, `abi_output*`, `abi_workspace_*`, and `abi_tiling_*` keys

- [ ] **Step 5: Commit**

```bash
git add include/RuntimeMix/MixArtifact.h \
        lib/RuntimeMix/MixDirectBackend.cpp
git commit -m "refactor: emit detailed RuntimeMix runner ABI metadata"
```

## Task 2: Generate Runner From ABI Metadata

**Files:**
- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`
- Test: xvm full baremix acceptance

- [ ] **Step 1: Replace scattered runner constants with metadata-driven generation**

In `lib/RuntimeMix/MixDirectBackend.cpp`, update runner source generation so it consumes the centralized ABI metadata instead of hardcoding:

- input file names
- input/output shapes
- input/output dtypes
- workspace bytes
- tiling source/mode

This does not require a generic runtime parser. It is acceptable to inject the metadata into the generated runner source as literal values produced from the centralized metadata builder.

Concretely, remove direct duplication such as:

```cpp
size_t aFileSize = 32768 * sizeof(int16_t);
...
if (!readHostToDevice(inputDir + "/x1_gm.bin", ...))
```

and replace it with code generated from the metadata source of truth.

- [ ] **Step 2: Replace hardcoded tiling/workspace generation with metadata-derived injection**

Still in `lib/RuntimeMix/MixDirectBackend.cpp`, make the generated runner’s workspace size and tiling setup come from the centralized metadata instead of separate baremix constants.

For this phase, the tiling implementation may still use the current fixed-bytes baremix path, but the runner must be told that through metadata emission/injection rather than a separate hidden assumption.

- [ ] **Step 3: Verify xvm runner path still passes**

Run:

```bash
sleep 2 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  rm -f build/runtime-mix-bootstrap/bin/mix-compiler build/runtime-mix-bootstrap/bin/mix-validator &&
  bash examples/baremix-test/run.sh
'
```

Expected:
- `PASS`
- `test pass`
- no regression in runner-first behavior

- [ ] **Step 4: Commit**

```bash
git add lib/RuntimeMix/MixDirectBackend.cpp
git commit -m "refactor: drive RuntimeMix runner generation from ABI metadata"
```

## Task 3: Make Validator Fallback Consume The Same ABI Metadata

**Files:**
- Modify: `tools/mix-validator/mix_validator_main.cpp`
- Modify: `examples/baremix-test/README.md`
- Test: xvm full acceptance plus manifest/README inspection

- [ ] **Step 1: Add a small manifest ABI parser in `mix-validator_main.cpp`**

In `tools/mix-validator/mix_validator_main.cpp`, add a narrow parser for the new manifest ABI keys. It only needs to support the current baremix metadata shape, for example:

```cpp
struct AbiTensorDesc {
  std::string name;
  std::string file;
  std::string dtype;
  std::vector<int64_t> shape;
};

struct AbiMetadata {
  std::vector<AbiTensorDesc> inputs;
  std::vector<AbiTensorDesc> outputs;
  size_t workspaceBytes = 0;
  std::string workspaceMode;
  std::string tilingMode;
  std::string tilingSource;
};
```

Keep the parser intentionally narrow and manifest-specific; do not build a generic schema system here.

- [ ] **Step 2: Replace fallback baremix constants with parsed ABI metadata**

Use the parsed metadata in the direct fallback path instead of hardcoding:

- shapes
- dtypes
- filenames
- workspace size
- tiling mode/source dispatch

It is acceptable if the fallback still errors out for anything outside the current baremix acceptance shape, but it must do so after reading manifest metadata, not by relying on duplicated constants.

- [ ] **Step 3: Update README to describe metadata-driven ABI**

Update `examples/baremix-test/README.md` to reflect the new state, for example:

```md
## Current ABI

The current baremix path is still limited to the existing 3-input/1-output acceptance case, but the runner and validator now consume explicit ABI metadata emitted in the artifact manifest instead of duplicating baremix constants in multiple places.
```

- [ ] **Step 4: Run final xvm acceptance**

Run:

```bash
sleep 2 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  rm -f build/runtime-mix-bootstrap/bin/mix-compiler build/runtime-mix-bootstrap/bin/mix-validator &&
  bash examples/baremix-test/run.sh &&
  md5sum build/runtime-mix-baremix/testdata/output/golden.bin \
         build/runtime-mix-baremix/testdata/output/actual.bin
'
```

Expected:
- `PASS`
- `test pass`
- identical md5 values

- [ ] **Step 5: Commit**

```bash
git add tools/mix-validator/mix_validator_main.cpp \
        examples/baremix-test/README.md
git commit -m "refactor: consume RuntimeMix ABI metadata in validator fallback"
```

## Self-Review

- Spec coverage:
  - explicit ABI metadata model is covered by Task 1
  - runner generation from metadata is covered by Task 2
  - validator fallback alignment is covered by Task 3

- Placeholder scan:
  - No `TODO`, `TBD`, or vague “handle appropriately” placeholders remain.
  - Every task includes exact file paths and xvm verification commands.

- Type consistency:
  - The plan keeps the current baremix acceptance surface and does not drift into generic ABI inference.
  - `MixDirectBackend` remains the single producer of ABI metadata.
