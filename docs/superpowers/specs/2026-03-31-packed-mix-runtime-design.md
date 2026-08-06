# Packed Mix Runtime Support Design

## Goal

Enable `lib/runtime` to execute matmul mix kernels using our own runtime path, while preserving all existing pure-vector and cube example flows.

The immediate target is mix kernels in `examples/matmul-add-relu-sum/` that currently fail under our raw runtime path but succeed under the sample/toolkit packed execution path.

## Current Situation

### What already works

- Existing `lib/runtime` path works for pure-vector examples.
- Sample/toolkit packed path works for mix kernels.
- A sample-derived target graph has already been validated in simulator with:
  - fp16 x fp16 -> fp32
  - `bias[M,N]`
  - ReLU
  - sample execution model
  - toolkit auto-packed artifact

### What fails

Our current runtime mix path is still raw-launch based:

- `Compiler` emits a plain linked `.bin`
- `Executor` uses:
  - `rtDevBinaryRegister`
  - `rtFunctionRegister`
  - `rtKernelLaunch`
- `validator` and `HostRunnerGen` assume this same raw launch contract

This path does not successfully run matmul mix kernels.

## Root Cause Summary

The failure is not the graph itself. The graph has already been validated under the sample/toolkit path.

The missing capability is a **packed mix runtime contract**.

The sample success path depends on:

- separate AIC/AIV device objects
- merged `device.o`
- packed host-side shared library (`libascendc_kernels_sim.so`)
- host stub entrypoints (`aclrtlaunch_*`)
- host-stub-managed hidden launch ABI

Our runtime does not currently support this contract.

## Recommended Approach

## Decision

Add a **mix-only packed execution branch**.

- `vec` / `cube`: keep current runtime path unchanged
- `mix`: add a new path that consumes a packed host-side shared library and launches through `aclrtlaunch_*`

This is intentionally a split runtime path. It minimizes risk and preserves working behavior.

## Why this approach

1. It matches the only validated successful mix contract.
2. It avoids destabilizing working pure-vector flows.
3. It keeps the new behavior isolated to `kernel_type == mix`.
4. It avoids re-implementing the full sample host-stub ABI manually inside our runtime.

## Non-goal

Do **not** unify mix and vector under one raw launch mechanism in this step.

That can be revisited later only if needed, after mix support is stable.

## Design

## 1. Compiler output contract

### Existing behavior

For mix kernels, `Compiler` currently:

- compiles AIC object
- compiles AIV object
- links both directly into one plain `.bin`

### New behavior for `kernel_type == mix`

`Compiler` should instead produce a packed host-side shared library using the toolkit flow.

### Required artifact flow

1. compile AIC object
2. compile AIV object
3. merge AIC/AIV device objects into merged `device.o`
4. generate/update host stub
5. pack merged `device.o` into host stub object
6. link packed host shared library

### Minimum final artifact

For runtime consumption, the only required final artifact is:

- packed shared library

Recommended path shape:

- `output_dir/lib<kernel_name>_packed.so`

Intermediate files may remain in output dir for debugging, but runtime should only depend on the final shared library path.

## 2. Executor mix runtime contract

### Existing behavior

`Executor` currently assumes a raw ELF launch contract.

### New mix behavior

For `kernel_type == mix`, `Executor` should not use:

- `rtDevBinaryRegister`
- `rtFunctionRegister`
- `rtKernelLaunch`

Instead it should:

1. `dlopen(packed_so)`
2. `dlsym("aclrtlaunch_<kernel>")`
3. allocate device memory for inputs / outputs / workspace / tiling
4. upload input tensors and tiling
5. call packed launch wrapper
6. synchronize stream
7. copy outputs back to host

### Important note

The host stub already encapsulates:

- `RegisterAscendBinary`
- `LaunchAscendKernel`
- hidden args such as `ffts_addr` and overflow buffer

So `Executor` should **reuse** that wrapper contract, not reimplement it.

## 3. Validator changes

`validator` should branch on `kernel_type`:

- `vec` / `cube`: existing flow unchanged
- `mix`: use packed artifact path and packed launch API

Validator should still own:

- input loading
n- tiling packing
- output buffer allocation
- comparison / tolerance logic

Only the final launch method changes.

## 4. HostRunnerGen changes

`HostRunnerGen` should also branch on `kernel_type`:

- `vec` / `cube`: keep current generated runner
- `mix`: generate a runner that loads the packed shared library and calls `aclrtlaunch_*`

Do not duplicate the raw mix launch path anymore.

## 5. Compatibility boundary

The safety boundary is strict:

- no behavior changes for non-mix kernels
- new code path guarded by `kernel_type == mix`

This is the key mechanism that keeps existing examples safe.

## Implementation Outline

## Step 1: Compiler

Update `lib/Runtime/Compiler.cpp` so mix mode emits packed shared library output.

Likely changes:

- keep current AIC/AIV two-pass compilation
- replace final direct `.bin` link with toolkit merge + pack sequence
- return packed `.so` path for mix

## Step 2: Executor

Add a mix launch helper in:

- `include/Runtime/Executor.h`
- `lib/Runtime/Executor.cpp`

Suggested interface shape:

- `RunPackedMixFile(...)`

This should be internal to runtime and only used by mix callers.

## Step 3: validator

Update:

- `tools/validator/validator_main.cpp`

Behavior:

- if `kernel_type != mix`: use existing path
- if `kernel_type == mix`: call packed mix path

## Step 4: HostRunnerGen

Update:

- `lib/Runtime/HostRunnerGen.cpp`

Behavior:

- generated mix runner uses packed shared library + `aclrtlaunch_*`
- generated vector/cube runner unchanged

## Verification Plan

## Phase A: artifact verification

For a mix kernel, verify compiler outputs:

- packed shared library exists
- library contains:
  - `.ascend.meta.*`
  - `.ascend.kernel.*`
  - `aclrtlaunch_*`

## Phase B: runtime verification

Use `validator` mix path on a known successful sample-derived kernel first.

Expected result:

- simulator completes
- outputs are produced
- accuracy passes

## Phase C: regression verification

Re-run at least one existing pure-vector example through current runtime path.

Expected result:

- unchanged behavior
- no regressions

## Risks

### 1. Toolkit artifact path assumptions

The toolkit auto-pack flow may assume file names or directory layout. Keep the mix compiler layout simple and explicit.

### 2. Dynamic linking environment

Packed mix shared libraries may require specific runtime search paths. Runtime should report clear errors when `dlopen` fails.

### 3. HostRunnerGen drift

If `HostRunnerGen` keeps old raw mix assumptions, user-visible inconsistency will remain. Its mix branch should be updated after validator is proven.

## Recommended file changes

### Must change

- `lib/Runtime/Compiler.cpp`
- `include/Runtime/Compiler.h`
- `lib/Runtime/Executor.cpp`
- `include/Runtime/Executor.h`
- `tools/validator/validator_main.cpp`

### Should change next

- `lib/Runtime/HostRunnerGen.cpp`

## Expected outcome

After this change:

- our own `lib/runtime` can execute mix kernels through the correct packed contract
- pure-vector examples still use the old runtime path unchanged
- the matmul mix example can move from sample/toolkit-only success to runtime-owned success
