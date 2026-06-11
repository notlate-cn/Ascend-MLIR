# Encoder real-NPU rerun + plog classification — handoff (2026-05-25)

Runner session re-ran the single-layer transformer encoder on real 910C (device 7) and corrected the prior fault classification by pulling the already-collected plog. Branch `dev-network` (worktree `encoder-robustness`) @ `746ba742`.

## TL;DR
- **plog was never lost.** The EXIT trap in `run-real-npu-job.sh` runs `collect-plog.sh` before the `--rm` container dies → `…/logs/plog/plog-errorStr.txt` persists in the job dir on mounted `/data`. The previous "no plog captured" was a wrong assumption; the runner had only copied `custom-cmd.log`. **No re-run was needed to get the sub-error codes — they were sitting on the remote.**
- **Corrected fault classification (was guessed "VEC UB" — wrong):**
  - 085ad7a2 autotuned, group20__v1 (XBLOCK=2): rc=507035 → **"The GM address accessed by scalar exceeds 48 bits"** subErrType:4 — wild scalar GM addr, not UB tile overrun.
  - 1926675e no-auto: group2: rc=507034 → **"timeout or trap error"**.
  - Neither is the f16 unaligned-tail problem the AF AlignmentStrategy port solves → that big port is the WRONG lever.
- **Today's rerun (746ba742, skip-autotune): group2/20/26 now PASS** (the leading-broadcast faults are gone on newer dev-network). New wall = **kernel_group18 HANGS** on device 7 (no fault, no plog). Killed clean, device 7 freed.

## Today's run order on device 7
`group3 → 14 → 21 → 2 → 20 → 26 → 5 → 16 → 18[HANG]`. network_test stuck ~9 min on group18__v0, killed by PID. plog empty = hang, no aicore trap.

## kernel_group18 = dual-output elementwise (add + fill0), NOT reduce/broadcast
`groups/kernel_group18.mlir`: add(arg0,arg1)→out0 AND fill(0.0)→out1, shape 2x8x64=1024, two results, kind=Vector. Default picker XBLOCK=256 → block_dim=ceil(1024/256)=4. block_dim=4 multicore on 1024 elems deadlocks — same family as sweep's block_dim=2 sim hangs, but on real silicon + dual-output. Tiling space XBLOCK/XBLOCK_SUB∈{16,32,64,128,256}.

## To get a plog code for group18: force a fault
hang → no errorStr. Force smaller XBLOCK (16 → block_dim=64) or single block → likely fault/pass; or enable autotune to find a passing variant. Then pull `…/logs/plog/plog-errorStr.txt`.

## How to run (gser, dev-network worktree)
sync needs symlinked externals: untracked `scripts/sync-and-submit-gser.sh` = stock + `tar -h --exclude=externals/llvm-project`. `source examples/env_gser.sh` first.
```
bash scripts/sync-and-submit-gser.sh --cmd \
 'NETWORK_RUNNER_SKIP_AUTOTUNE=1 PYTHONPATH=python python3 python/network_runner.py \
   --input-linalg /data/gser/enc_ref/encoder.mlir --inputs /data/gser/enc_ref/input0.npy \
   --expected /data/gser/enc_ref/expected0.npy --workdir /data/gser/enc_npu_rerun3 \
   --soc Ascend910B1 --atol 1e-2 --rtol 1e-2 --backend npu'
```
Evidence: `/tmp/npu-real-logs/20260525-encoder-rerun/` (custom-cmd.log, group18/, group20/). See [[project_real_npu_aclnn_direct]], [[project_real_npu_test_checklist]].
