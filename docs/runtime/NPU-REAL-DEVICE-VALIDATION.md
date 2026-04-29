# NPU Real-Device Validation

## Goal

Provide the first real-device bring-up checklist for the current runtime-native
NPU path.

This document does not claim NPU completion. It defines the smallest command
sequence and expected observations once hardware is available.

## Preconditions

- a machine with usable Ascend NPU hardware
- Ascend toolkit/runtime environment configured
- `LLVM_BUILD_DIR` set correctly
- repository build succeeds through:
  - `./scripts/build.sh --build-project --llvm-build-dir "$LLVM_BUILD_DIR"`

## What To Validate First

Validate in this order:

1. runtime binary/toolchain availability
2. single-task NPU launch
3. expected-output validation path
4. profile/summary retention
5. only then multi-task scheduler behavior

Do not start with concurrency or performance tuning.

## Suggested Command Sequence

1. Build the project:

```bash
export LLVM_BUILD_DIR=/path/to/llvm/build
./scripts/build.sh --build-project --llvm-build-dir "$LLVM_BUILD_DIR"
```

2. Run the focused runtime baseline first:

```bash
bash test/tools/runtime/run_runtime.sh
```

3. Run a minimal NPU smoke through `runtime-session` with a known-good artifact
   root or kernel setup.

Expectation:

- `session.backend=npu`
- no `[npu:bindings]` or `[npu:artifact]` failure for a valid request
- if device initialization still fails, the failure should be clearly staged as
  `[npu:executor_initialize]` or `[npu:kernel_launch]`

4. Repeat with expected outputs enabled.

Expectation:

- output files are written
- validation either passes or returns a clear `[npu:validate]` failure

5. Only after single-task success, run a multi-task NPU case through the global
   scheduler path.

Expectation:

- `scheduler_scope=global`
- session summary/profile artifacts are retained
- scheduler counters remain coherent after session release

## What To Record

For each real-device run, record:

- exact command
- `session.profile.summary` path
- `session.runtime.attribute.*`
- `session.runtime.counter.*`
- first failing stage if the run does not succeed

## Repository-Owned Smoke Assets

The repository now includes a manifest helper for two minimal NPU smoke cases:

- `vec` smoke based on `examples/relu-broadcast-transpose`
- `mix` smoke based on `examples/matmul-add-leakyrelu`

Helper:

```bash
test/tools/runtime/prepare_npu_smoke_manifests.sh
```

### Vec Smoke

1. Prepare or reuse the example artifact root:

```bash
bash examples/relu-broadcast-transpose/run.sh
```

2. Generate the NPU smoke manifest:

```bash
MANIFEST=/tmp/runtime-npu-vec-smoke.json
ARTIFACT_ROOT=$PWD/examples/relu-broadcast-transpose/build_e2e/artifact
OUTPUT=/tmp/runtime-npu-vec-output.npy

bash test/tools/runtime/prepare_npu_smoke_manifests.sh vec \
  "$MANIFEST" \
  "$ARTIFACT_ROOT" \
  "$PWD/examples/relu-broadcast-transpose/input_data0.npy" \
  "$PWD/examples/relu-broadcast-transpose/input_data1.npy" \
  "$OUTPUT" \
  "$PWD/examples/relu-broadcast-transpose/output_expected.npy"
```

3. Run it:

```bash
build/bin/runtime-session --run-manifest "$MANIFEST" --run
```

### Mix Smoke

1. Prepare or reuse the example artifact/data roots:

```bash
bash examples/matmul-add-leakyrelu/run.sh
```

2. Generate the NPU smoke manifest:

```bash
MANIFEST=/tmp/runtime-npu-mix-smoke.json
ARTIFACT_ROOT=$PWD/build/runtime-mix-matmul-add-leakyrelu
DATA_DIR=$PWD/build/runtime-mix-matmul-add-leakyrelu-data/npy
OUTPUT=/tmp/runtime-npu-mix-output.npy

bash test/tools/runtime/prepare_npu_smoke_manifests.sh mix \
  "$MANIFEST" \
  "$ARTIFACT_ROOT" \
  "$DATA_DIR/input_a.npy" \
  "$DATA_DIR/input_b.npy" \
  "$DATA_DIR/input_bias.npy" \
  "$OUTPUT" \
  "$DATA_DIR/output.npy"
```

3. Run it:

```bash
build/bin/runtime-session --run-manifest "$MANIFEST" --run
```

## Failure Triage

Use this split when the first real-device failures appear:

- `[npu:bindings]`
  - runtime request contract error
- `[npu:artifact]`
  - missing binary/shared-object/kernel metadata
- `[npu:executor_initialize]`
  - device runtime/driver initialization failure
- `[npu:kernel_launch]`
  - real launch failure after initialization
- `[npu:validate]`
  - launch completed but outputs mismatched
- `[npu:write_outputs]`
  - runtime could not persist produced outputs

The first investigation target should be the earliest failing stage, not later
effects.
