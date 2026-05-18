# AGENTS

## Goal

- Keep `lib/Runtime` as the single runtime center for:
  - AscendC kernel compilation
  - CPU simulation execution with profiling
  - NPU execution path wiring
  - task-graph-based execution and future multi-task scheduling
- Keep `runtime-session` as the only general runtime CLI entry point.
- Keep xvm CPU-simulation verification green while finishing real NPU validation.
- Treat the original runtime task as architecturally complete except for real-device validation and follow-up fixes exposed by real hardware.

## Current Development Mode

- Edit locally in this workspace, then verify on xvm under `/home/niu/code/Codex-Ascend-MLIR`.
- xvm is the authoritative development verification environment.
- For xvm build/test/debug workflow, follow `examples/dev-env.md`.
- xvm currently uses CANN 9.1:
  - `/home/niu/Ascend/latest -> /home/niu/Ascend/cann-9.1.0`
  - Set `ASCEND_HOME_PATH=/home/niu/Ascend/latest` or source `/home/niu/Ascend/latest/set_env.sh` before verification.
- Do not treat xvm CPU simulation as real NPU completion.
- Before any candidate fix, generated-kernel variant, or new runtime artifact is
  advanced to 910C real-device validation, run it on xvm with Ascend910B1
  simulation first and require `session.backend=sim`, `session.result=success`,
  and `session.validation=pass`.
- Keep the xvm result for the original demo separate from the xvm result for a
  candidate fix. An original-demo xvm pass does not authorize taking an
  unverified candidate fix to the real NPU.
- Early-return checkpoints and other intentionally incomplete diagnostic kernels
  may be run on the real NPU only for localization. They are not proof that a
  fix is ready for board validation.
- The remote real-NPU host is run-only for this project. It does not have the full LLVM development stack.
- For real-device NPU debugging and operation, follow `examples/real-npu.md`.
- For remote NPU validation:
  1. Generate artifacts, data, expected outputs, and run manifests on xvm.
  2. Build a run-manifest-only `runtime-session` on xvm with `ASCEND_RUNTIME_SESSION_RUN_ONLY=ON`.
  3. Package only the runner, runtime artifacts, data, run manifest, and required runtime libraries.
  4. Run on the remote host after sourcing the remote CANN/driver environment.
- Use `ASCEND_DEVICE_ID=7` for the shared real-NPU host unless the user explicitly changes the device.

## Current Progress

- Runtime architecture has been reorganized around:
  - `Artifact/`
  - `Execution/`
  - `Profile/`
  - `Mix/`
  - `Support/`
- `runtime-session` keeps CLI parsing, summary printing, and execution orchestration only.
- Runtime request assembly lives in:
  - `include/Runtime/Artifact/RuntimeSessionRequestBuilder.h`
  - `lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp`
- Shared runtime frontend behavior lives in:
  - `include/Runtime/Execution/RuntimeFrontendCore.h`
  - `lib/Runtime/Execution/RuntimeFrontendCore.cpp`
- `runtime-session` and the C API use the shared frontend core for compile request assembly, single-task run preparation, normalized execution, and summary interpretation.
- `DefaultExecutionRunner` delegates to `NativeExecutionRunner`.
- The old active `Legacy/` runtime implementation path has been removed from the current runtime execution path.
- `NpuBackend` is wired through the runtime-native execution path and honors `ASCEND_DEVICE_ID`.
- `NativeExecutionRunner` now reports `rtStreamSynchronize` failures instead of treating failed real-device execution as success.
- A run-manifest-only `runtime-session` build path exists so the remote host can execute prebuilt artifacts without loading CANN compiler/simulator dependencies at process startup.
- Remote host CANN 9.1 toolkit and A3 ops are installed under `/data/nyh/Ascend`.
- xvm CANN 9.1 toolkit and A3 ops are installed under `/home/niu/Ascend`, with top-level `latest` switched to 9.1.
- Current xvm 9.1 verification passes:
  - `bash test/tools/runtime/run_runtime.sh`
    - `RC=0`
    - runtime tests report `115 passed, 0 failed`
    - SimBackend vec/mix baseline passes
    - repeated mix simulation baseline passes
  - `bash test/tools/runtime/run_simbackend_examples.sh`
    - `RC=0`
    - 8 SimBackend examples pass
  - `bash test/tools/examples/example_pipelines.sh`
    - `RC=0`
    - 10 example pipelines pass
    - cross-session runtime-session smoke passes
- Real NPU validation current state:
  - minimal `const640` run-only case passes on device 7 with `session.result=success` and `session.validation=pass`
  - `examples/relu-broadcast-transpose` now passes on device 7 with `TB_N=16`, run-only packaging, `session.result=success`, and `session.validation=pass`
  - `examples/add-broadcast-concat` now passes on device 7 with `TB_N=16`, run-only packaging, `session.result=success`, and `session.validation=pass`
  - `examples/split-relu-brc-add-mul` now passes on device 7 after removing dead queue-backed TBuf initializers, with `session.result=success` and `session.validation=pass`
  - The previous `rtStreamSynchronize failed: rc=507035` / `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION` was localized to generated-kernel UB usage:
    - launch ABI, H2D/D2H, GM pointer alignment, workspace, and tiling words were correct
    - real plog reported VEC UB out-of-bounds / scalar GM address over 48 bits, not a pure host dependency failure
    - the fix keeps VECOUT outputs allocated from the VECOUT queue, frees temporary VECIN tensors, resolves `affine.min`/`memref.dim(subview)` tail alloc sizes to the loop-step upper bound, and hoists loop-invariant `InitBuffer`/`InitQueue` out of the inner tile loop

## Decisions

- Keep request-building logic in the runtime library, not in CLI `main.cpp`.
- Preserve CLI behavior while refactoring internals; do not accept silent semantic drift.
- Use xvm for all normal build/test verification.
- For real-NPU debugging status, do not present xvm simulation success as a
  project result. Treat xvm success only as the required gate before a
  candidate is allowed onto the real device.
- Use the real NPU host only for run-only validation and hardware-specific debugging.
- Do not run a proposed fix on the real NPU until that exact candidate has
  passed xvm Ascend910B1 simulation, except for explicitly labeled diagnostic
  checkpoint runs.
- Treat any new real-NPU failure as a kernel/ABI/tiling investigation until evidence proves otherwise.
- Do not reopen old `Legacy/` cleanup work as an active blocker unless a concrete current dependency reappears.
- Treat these as the current CPU-simulation regression baselines:
  - `test/tools/runtime/run_runtime.sh`
  - `test/tools/runtime/run_simbackend_examples.sh`
  - `test/tools/examples/example_pipelines.sh`

## Real NPU Debug Playbook

Real-device failures must be debugged by narrowing the failing layer, not by guessing.

1. Add NPU launch tracing near `NativeExecutionRunner` launch assembly:
   - kernel name
   - binary path
   - block dim
   - input and output counts
   - each input/output byte size
   - GM pointer values and alignment
   - workspace size
   - tiling byte count and first 64-bit words
   - final launch argument count and byte size
2. Build real-NPU microcases and run them through the same xvm-to-remote run-only packaging path:
   - `const640`: write-only output baseline
   - `copy640`: read input and write output
   - `relu-only`: GM to local compute to GM
   - `broadcast-add-only`
   - `transpose-only`
   - full `relu-broadcast-transpose`
3. Bisect shape and scheduling sensitivity:
   - `block_dim=1` vs current multi-block launch
   - aligned shapes such as 64 or 128
   - tail shapes such as 65, 127, 500, and 640
4. If the full generated kernel is still the first failing case, insert early-return checkpoints into the generated kernel:
   - immediately after entry
   - after buffer/queue init
   - after first `DataCopy`
   - after compute primitive
   - before and after final writeback
5. Collect remote CANN logs and `npu-smi` state for each failure, but use microcase and checkpoint results as the primary localization signal.
6. Fix at the layer identified by evidence:
   - runtime ABI/launch packing if trace data disagrees with the generated CANN signature
   - tiling/schema packing if the kernel receives wrong tiling values
   - lowering/scheduling/codegen if only a specific primitive, tail path, or block partition fails

### Real NPU Failure Log Triage

- For any `rtStreamSynchronize failed` on the real device, collect the real
  plog `errorStr` before guessing at the fix. Start with:
  ```shell
  grep -R "errorStr" -n \
    /root/ascend/log/debug/plog \
    /var/log/npu/slog \
    /var/log/npu/plog \
    /root/ascend/log 2>/dev/null | tail -n 50
  ```
- `507035` / `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION` is a symptom class, not the
  root cause. Interpret it together with `errorStr` and launch trace data.
- If `errorStr` says `ADDR_MISALIGN` or `UB address ... is not aligned`, check
  API-specific alignment constraints and print/check UB offsets around the
  failing primitive.
- If `errorStr` says `VEC instruction to read/write UB is out of bounds`, treat
  it as generated-kernel UB lifetime, queue, tile-size, or primitive-shape
  misuse until evidence proves otherwise.
- If `errorStr` also says `GM address accessed by scalar exceeds 48 bits`, do
  not immediately blame host GM allocation. If launch trace shows H2D/D2H,
  GM pointer alignment, workspace, argument count, and tiling words are sane,
  this can be a downstream effect of UB corruption in the kernel.
- Recompiling the same generated `step8_kernel.cpp` on the remote CANN host is
  useful to rule out xvm compile artifact mismatch, but if the same error
  remains, continue debugging kernel ABI/tiling/lowering rather than host
  dependency setup.
- Status reports during real-NPU work should emphasize real-device facts only:
  case name, remote directory, device id, pass/fail, error code, key `errorStr`,
  and whether launch trace ruled out ABI/H2D/GM/tiling issues.

### `relu-broadcast-transpose` 507035 Lessons

- The earlier `examples/relu-broadcast-transpose` real-device failure was not
  caused by launch ABI: H2D roundtrip, GM pointer 512B alignment, workspace,
  argument count, and tiling words were correct.
- The failing plog reported VEC UB out-of-bounds / scalar GM address over 48
  bits. The accepted fix was in generated kernel lowering and tile sizing.
- VECOUT outputs must be allocated from the VECOUT queue before enqueue.
  Enqueueing a VECCALC tensor into a VECOUT queue may pass simulator checks but
  fail on real hardware with UB/MTE faults.
- Temporary GM-to-VECIN tensors created through
  `AllocTensor -> DataCopy -> EnQue -> DeQue` must be released with
  `FreeTensor` after their final use.
- For all-parallel tail-tiled kernels, buffer allocation should use the
  enclosing loop-step upper bound while DataCopy/compute still use the actual
  tail element count. This allows one max-sized queue/tbuf to be reused safely
  across tail iterations.
- Resolve `affine.min(remaining, step)` and `memref.dim(subview)` tail sizes to
  the loop-step upper bound when computing all-parallel VECOUT/VECIN/VECCALC
  buffer byte sizes.
- Hoist loop-invariant `InitBuffer` / `InitQueue` out of the inner tile loop.
  Repeated per-iteration initialization can consume real UB differently from
  the simulator and produce `507035`.
- Do not apply the same max-size substitution blindly to reduction outputs.
  Rank-1 VECOUT reduction outputs may need exact tail size because
  `ReduceSum2DL2` codegen derives rows/cols from source and destination tensor
  sizes.
- Reducing `TB_N` from 64 to 16 lowered the current demo's UB live set and is
  part of the accepted real-device configuration, but tile changes alone are
  not a root-cause fix unless the generated kernel's queue and buffer lifetime
  are also correct.

### `add-broadcast-concat` 507035 Lessons

- The earlier `examples/add-broadcast-concat` real-device failure was a tiling
  configuration issue, not a launch ABI issue. Launch trace showed sane
  argument count, H2D/D2H, 512B-aligned GM pointers, workspace, and tiling words.
- In this generated kernel, `TB_N` currently behaves as the inner M tile size;
  `N=500` stays full-width inside each tile. `TB_N=192` overcommits the UB live
  set and failed on 910C with `rtStreamSynchronize rc=507035`.
- The accepted configuration is `TB_M=64, TB_N=16`, with xvm Ascend910B1
  simulation first and real 910C validation afterwards.

### `split-relu-brc-add-mul` 507035 Lessons

- Matching the run manifest tiling fields to the generated CANN `TilingData`
  signature is good hygiene, but it was not the root cause: an 8-field tiling
  probe still failed with the same `507035` before the UB cleanup.
- The root cause was generated-kernel UB pressure from queue-backed memref allocs
  that also retained standalone TBuf initializers. Those TBufs had no users
  except `TPipe.InitBuffer`, but still consumed real UB after hoisting.
- Delete TBufs whose only users are `TPipe.InitBuffer` after data-move/compute
  conversion. For this demo, generated `InitBuffer` calls dropped from 20 to 14,
  xvm simulation stayed green, and the real 910C run passed.
- Keep `examples/split-relu-brc-add-mul` run manifests aligned with the current
  CANN signature: `TB_M`, `TB_N`, `dim_arg0_1`, `dim_arg1_0`, `dim_arg0_0`,
  `dim_arg3_0`, `dim_arg2_0`, and `dim_arg4_0`.

## Notes

- `examples/dev-env.md` is the operational guide for xvm build/test/debug workflow.
- `examples/real-npu.md` is the operational guide for remote real-device NPU debugging, run-only packaging, and current real-NPU findings.
- If xvm reports stale or inconsistent build state, prefer a clean reconfigure before interpreting failures as CANN 9.1 regressions.
- Do not include unrelated dirty files in commits or reviews.
