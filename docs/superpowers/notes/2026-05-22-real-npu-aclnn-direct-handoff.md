# Real-NPU handoff: aclnn direct-call ops (Matmul / LayerNorm / Transpose / FA)

**Date:** 2026-05-22 · **Branch:** `dev-network` (worktree `encoder-robustness`)
**Author session role:** code + unit tests. **Runner session role:** execute on 910C.

## TL;DR for the runner session

The encoder and tiny-BERT networks pass on the **CPU simulator** today
(max_diff 7e-7 / 1.4e-5). On sim, every aclnn op runs a **host-mode CPU
reference**; the AscendC kernels run on camodel. **Nothing on the aclnn side has
ever executed on real hardware** — including FlashAttention, whose device branch
was written earlier but never validated (confirmed by the user).

This session added the pieces needed to *try* a real-device run. They compile
and the CPU references are unit-tested, but the **device path is unverified**.
Your job: run it on a 910C and report where it breaks.

## What I implemented this session (all on `dev-network`)

1. **`--backend npu` ported into the runner** (from the dangling WIP `64314d5`):
   - `python/network_runner.py`: `--backend {sim,npu}` (default sim). For npu,
     phase-5 sets `NETWORK_RUNNER_BACKEND=npu` and strips `/simulator/` +
     `/devlib/` from `LD_LIBRARY_PATH` (those shadow the real driver HAL →
     `rtSetDevice` 107001).
   - `python/runner_utils/build_host.py`: `backend="npu"` drops the
     simulator-only link libs (`runtime_camodel/npu_drv/stars/model_top`) and
     the simulator `-L` dir; real `libruntime.so` is dlopen'd via `ASCEND_HOME`.
   - `lib/Runtime/Execution/HostLaunchHelper.cpp`: `NETWORK_RUNNER_BACKEND=npu`
     selects `ExecutionBackendKind::Npu` (else Simulation).
   - Phases 1–4 always use sim; only the phase-5 final run+verify goes to device.

2. **Real-device branches for the 3 host-only aclnn ops** in
   `lib/Runtime/AclnnOps.cpp` (each: `if (g_host_mode) cpu_ref; else aclnn`):
   - `run_Matmul` → `aclnnMatmul` (rank 2) / `aclnnBatchMatMul` (rank > 2),
     `cubeMathType=1` (ALLOW_FP32_DOWN_PRECISION).
   - `run_LayerNorm` → `aclnnLayerNorm`, normalized over the last dim, eps 1e-5,
     allocates the mean/rstd aux outputs.
   - `run_Transpose` → `aclnnPermute` with `aclCreateIntArray(perm)`.
   - `run_FlashAttentionScore` already had a device branch (untested).

3. **Unit test** `test/tools/runtime/test_aclnn_ops.cpp` — 7 cases pinning the
   CPU references (the sim source-of-truth) with hand-computed expecteds.

## Verified locally (no NPU)

- Host CPU-reference unit tests: **7/7 pass**
  ```
  c++ -std=c++17 -I include \
      test/tools/runtime/test_aclnn_ops.cpp lib/Runtime/AclnnOps.cpp \
      -o /tmp/test_aclnn_ops && /tmp/test_aclnn_ops
  ```
- `AclnnOps.cpp` compiles clean against **real CANN headers** (device-branch
  signatures correct):
  ```
  c++ -std=c++17 -c -I include \
      -I $ASCEND_HOME/x86_64-linux/include lib/Runtime/AclnnOps.cpp -o /tmp/x.o
  ```
- `libAscendCRuntime.a` rebuilds; sim path unchanged (npu is opt-in).

## NOT verified — your work on the 910C

### A. The aclnn ops themselves (numerics + that they even launch)
None of `aclnnMatmul / aclnnBatchMatMul / aclnnLayerNorm / aclnnPermute /
aclnnFlashAttentionScore` has run on hardware. Start with the **unit test on
device** before the full networks — it isolates each op:

- Build with CANN libs and toggle off host mode. Easiest: add `aclInit/aclrtSetDevice`
  + `setHostMode(false)` in a device `main` (or guard via env), feed device
  buffers (`aclrtMalloc` + `aclrtMemcpy` H2D for inputs; D2H the output before
  `checkClose`). The expected values are backend-agnostic; use a looser tol for
  f16 cube accumulation.
- Watch for: `*GetWorkspaceSize rc != 0` (printed to stderr), shape/format
  rejections, `cubeMathType` accuracy (flip to `0` KEEP_DTYPE if fp32 matmul is
  off).

### B. Mixed device-memory orchestration — CONFIRMED BROKEN then FIXED (Approach B)

**Confirmed broken (2026-05-22, device 7):** full encoder `--backend npu` ran
phases 1-4 clean (11 AscendC kernels compiled + autotuned on-device, each
standalone max_diff=0) but the integrated phase-5 failed. Root cause exactly as
predicted: the generated `network_host.cpp` orchestrates every tensor in HOST
memory (`::operator new`); the AscendC launch path (`hostLaunchAscendCKernel`)
is host-in/host-out (stages its own H2D/D2H); but the aclnn `run_*` device
branches used DEVICE pointers (aclrtMalloc out, host ptr wrapped as device in).
A tensor crossing aclnn↔AscendC got a device ptr read as host (group2
`input[1].host` decoded to a literal CANN `version.info` → aicore fault
`rtStreamSynchronize rc=507034`) or a host ptr read as device (group14/21
all-zero). Evidence: `/tmp/npu-real-logs/20260522-encoder-e2e/`.

**Fixed (Approach B, commit `085ad7a2` on dev-network):** the aclnn `run_*`
device branches are now **host-in/host-out**, matching the AscendC convention.
Each stages its inputs H2D into temp device buffers (`stageToDevice`), runs the
aclnn op into a temp device output (`stageDeviceOut`), then copies the result D2H
into a fresh host buffer (`stageToHost`); temps freed via `freePool`. All in
`AclnnOps.cpp` — no host-gen / HostLaunchHelper changes.

**✅ HW-re-validated end-to-end 2026-05-23 — via BERT.** BERT tiny full e2e
`--backend npu` on device 7 PASSES: `network.output[0]: max_diff=1.407e-05 PASS`,
`out0.npy` produced, matching BERT sim. The mixed-memory bug is fixed on a
complete network. (Verified locally too: unit 7/7, encoder+BERT sim no
regression, compiles vs real CANN headers.) Evidence: `/tmp/npu-real-logs/20260523-bert-fix/`.

**Encoder full e2e still fails — but for two SEPARATE, encoder-specific reasons,
not the mixed-mem fix** (which BERT proves correct). BERT's graph doesn't exercise
either; encoder does because it has a chain of Transposes feeding the first
AscendC kernels. Earlier "encoder advanced to group20, only one wall left" was
over-optimistic — the front-end is already corrupt before any tiling matters:

  (a) ~~front-end host-gen wiring bug~~ MISDIAGNOSED — NOT a bug. group3/14/21
      are `linalg.fill 0` zeroing a DPS-dest tensor.empty (init not read), feeding
      aclnn-Matmul inits that run_Matmul ignores → dead. "version.info input" =
      pre-fill init dump; all-zero output = correct. host-gen faithful; sim 7e-7
      confirms. Encoder's only blocker is (b).
  (b) **tiling-related real-HW AscendC kernel faults (camodel-invisible).** Default
      v0 tiling faults group2 (507034); the autotuner-selected v1 faults group20
      (507035, VEC UB out-of-bounds). Changing tiling only moves which kernel dies
      first — same family as the regression-sweep gather/combo UB. Kernel-codegen
      session's domain. NETWORK_RUNNER_SKIP_AUTOTUNE only swaps which (b)-kernel
      dies; it does NOT dodge (a).

Evidence: `/tmp/npu-real-logs/20260523-encoder-handoff/` (README with a debug
index). Encoder inputs staged remotely at `/data/gser/enc_ref/`.

[PERF FUTURE — Approach A] These per-op H2D/D2H round-trips are redundant once
both domains agree. Future optimization: unify on the DEVICE domain — host-gen
`aclrtMalloc`s all intermediates and `hostLaunchAscendCKernel` becomes
device-pointer-aware (skips its npy/H2D/D2H staging). Bigger change (AclnnBackend
+ HostLaunchHelper + ExecutionSession); deferred behind correctness.

### C. Link / driver environment
`64314d5`'s commit message flagged `ld: cannot find -lnpu_drv/-lstars/-lmodel_top`
— that's the simulator lib drop, already handled in `build_host.py`. If link
still fails on device, capture the exact `g++` line (printed with `+`).

## How to run the full networks on device

```bash
# inputs already staged: /tmp/enc_ref/{encoder.mlir,input0.npy,expected0.npy}
#                        /tmp/bert_e2e/tiny_fp32/{step0_linalg.mlir,input_0.npy,expected_0.npy}
source /home/gser/Ascend/cann/set_env.sh
export ASCEND_DEVICE_ID=0              # pick a free device
export NETWORK_RUNNER_SKIP_AUTOTUNE=1 # accuracy-only: skip phase-4 autotune,
                                      # use phase-3 default tilings (much faster)
PYTHONPATH=python python3 python/network_runner.py \
  --input-linalg /tmp/enc_ref/encoder.mlir \
  --inputs /tmp/enc_ref/input0.npy --expected /tmp/enc_ref/expected0.npy \
  --workdir /tmp/enc_npu --soc Ascend910B1 --atol 1e-2 --rtol 1e-2 \
  --backend npu
```
Phases 1–3 run on sim (build + default-tiling dump); phase-4 autotune is skipped
when `NETWORK_RUNNER_SKIP_AUTOTUNE` is set (recommended while validating numerics
— default tilings are correct, just untuned); phase-5 builds the device host
binary and runs on the NPU. Same form for BERT with `tiny_fp32/step0_linalg.mlir`.
Drop the env var once accuracy is confirmed to also exercise the on-device tuner.

## Op inventory per network (what the aclnn path must serve)

- **encoder**: 1 FlashAttention + 3 Matmul + 2 LayerNorm + 13 Transpose (aclnn),
  rest AscendC (elementwise/bcast on camodel→device).
- **BERT (tiny, fp32)**: Matmul + LayerNorm via aclnn; GELU (erf) stays fused
  **on-chip AscendC** (not aclnn) — so the device run also exercises the AscendC
  divf/erf vector codegen, not just aclnn.

## Process reminders
Follow the 8-point real-NPU checklist (record/kill exact container+PID, timeout
+monitor, pull logs repo-external, sync hygiene, rc semantics, clean orphans,
stage-aware diagnosis). Confirm before any kill/rm. Report stage where it breaks
(GetWorkspaceSize / launch / numeric) — that localizes A vs B vs C above.

## 2026-05-25 — encoder group20 wild-address diagnosis (broadcast Vector kernel)

plog confirmed: group20 v1 = "GM address accessed by scalar exceeds 48 bits"
(subErrType 4, fixp_error0=0x8b57905); group2 = trap. NOT AF-tail-alignment.
Source-read bounds are clamped; v1 tiles the leading broadcast axis (extent 2,
XBLOCK=XBLOCK_SUB=2). Suspect = tail offset `v41 = extent - XBLOCK_SUB` in
uint32 (kernel_group20.cpp:193) → underflows to ~4e9 when XBLOCK_SUB≥extent →
SetGlobalBuffer base+huge = wild addr (camodel doesn't check; same class for
group2). Offsets are uint32 (vs AF's int64). Not magnitude here (small shapes),
so int64 alone won't fix — needs the exact faulting address. To decide v37(slab)
vs v41(tail underflow): capture fixp_error0 + the 3 GM base ptrs (v1/v3/out)
printed post-aclrtMalloc; subtract. Minimal-change candidates: clamp tail
extent-XBLOCK_SUB underflow + clamp broadcast-axis tile offset to extent.
Reverted an ineffective TilePlanGen broadcast-exclude attempt; tree clean.
