# Real-NPU regression sweep — failure handoff (2026-05-21)

Runner session ran the regression-focus e2e set on the real 910C (device 7) from the x86 dev box via `scripts/sync-and-submit.sh`. **10/14 PASS, device healthy.** This doc hands the 4 non-passing cases to the fix session.

- Worktree: `/home/gser/code/Ascend-MLIR` (shared between runner + fix sessions), branch `develop`, ref `b264c917-local`.
- Copied raw logs (repo-external, not synced): **`/tmp/npu-real-logs/20260521-sweep/<case>/`** (logs/ + run_manifest.npu.json; per-case `logs/plog/plog-errorStr.txt` has the aicore detail).
- PASS (no action): multi-r, combo-elewise-reduce, reduce-sum-3d (+f16/+tail), leading-reduce, reduce-big-r, broadcast-add-reduce, relu-broadcast-transpose, two-elewise.

---

## 1. gather-elementwise-fusion — REAL-NPU UB out-of-bounds — ✅ RESOLVED 2026-05-22

> **FIXED & real-NPU verified PASS** on ref `23d06fb3` (`session.result=success / validation=pass`). Likely fixed by `23d06fb3` "map result dim to source dim for rank-reduced subview sizes" — index_select's rank-reduced subview dim-map bug was producing the out-of-bounds UB address. Original report kept below for record.



- **Symptom:** `session.error_stage=kernel_launch`, `rtStreamSynchronize failed: rc=507035`. **Reproducible** (2 consecutive runs identical).
- **Sim vs real:** SIM **PASSES** (`session.validation=pass`, sim STAGE 10). Fails **only on real hardware** → camodel does not model the fault.
- **Root cause (from plog):**
  ```
  [device_error_core_proc.cc] errorStr: The address for the VEC instruction to read/write UB
  is out of bounds. errcode:(0x4000000000000000,0,0) subErrType:4
  ```
  A VEC instruction in the kernel reads/writes Unified Buffer out of bounds → aicore exception.
- **Kernel:** `relu_index_select_add`, `block_dim=1`, args_count=11, inputs=3 (20480 / 1024 / 256 B), output 4096 B, `workspace=16MB`, tiling.words = `16,16,16,128,640,128`. All 3 inputs `h2d_roundtrip match=yes` (inputs land correctly; the bug is in-kernel UB addressing).
- **Notes:** memory says gather PASSED earlier this session → **regression introduced by a change now in the synced worktree**. Likely the index_select/gather UB offset or the elementwise tail handling computes an out-of-range UB address for this shape/tiling.
- **Fix session:** audit `relu_index_select_add` codegen UB addressing (gather offset + relu/add epilogue) against tiling `16,16,16,128,640,128`; the sim won't catch it — verify on real NPU (ask runner) or add a UB-bounds assert.
- Logs: `/tmp/npu-real-logs/20260521-sweep/gather-elementwise-fusion/`

## 2. relu-e2e — f16 tolerance, not a functional bug

- **Symptom:** `session.error_stage=validate`, `npu output mismatch: max_abs_diff=0.04 mean_abs_diff=0.00`. Sim passes.
- **Root cause:** `run_manifest.npu.json` has `atol=1e-05, rtol=1e-05` — unrealistically tight for f16 (f16 rel. error ~1e-3; on values ~20–30 an abs diff of 0.0x is normal rounding). Inputs `h2d_roundtrip match=yes`.
- **Decision needed:** either relax atol/rtol for f16 e2e cases, or investigate whether the f16 relu (`arith.maximumf` path) near-zero handling adds a tiny bias (negative inputs showed small non-zero outputs ≤0.04 instead of exact 0 — possibly just f16 rounding). Low severity.
- Logs: `/tmp/npu-real-logs/20260521-sweep/relu-e2e/`

## 3. multi-r-noncontig-e2e — camodel SIM hang (multicore) — ✅ RESOLVED 2026-05-22

> **FIXED & real-NPU verified PASS** on ref `23d06fb3` (commits `c6d44c27` "free per-trip VECIN tile tensors in reduce loops" + `23d06fb3` "map result dim to source dim for rank-reduced subview sizes"). Re-run on device 7: `session.result=success / validation=pass`. Root cause matched the hypothesis below (VECIN tile tensor leak in reduce loops → camodel resource spin). Original report kept below for record.



- **Symptom:** **hangs in the camodel simulator** at sim `STAGE 3 Simulator Run + Verify` (never reaches real NPU). runtime-session spins ~261% CPU indefinitely (observed 34 min). In the batch sweep this surfaced as `rc=255` (ssh idle-timeout drop) and orphaned a hung container.
- **Context:** `XBLOCK=8, XBLOCK_SUB=8, block_dim=2`, x[8,16,32]→out[16], "peeled outer R" template. Compile OK; hang is at sim run.
- **Hypothesis:** multicore (`block_dim=2`) sync/peeled-outer-R interaction deadlocks the simulator. Note the contiguous `multi-r-e2e` PASSES — difference is the non-contiguous peeled-outer-R path.
- **Fix session:** repro locally with the camodel runtime-session on this case; suspect the SyncAll/peel-outer-R sync. Add a sim watchdog/timeout so it can't wedge the harness.
- Logs: `/tmp/npu-real-logs/20260521-sweep/multi-r-noncontig-e2e/` (sim.log frozen at STAGE 3).

## 4. full-reduce-e2e — camodel SIM hang (multicore) — ✅ RESOLVED 2026-05-22

> **FIXED & real-NPU verified PASS** on ref `23d06fb3` (same commits as #3, primarily `c6d44c27` reduce-loop VECIN tile free). Re-run on device 7: `session.result=success / validation=pass`. Original report kept below for record.



- **Symptom:** same shape of failure as #3 — **hangs in camodel sim at STAGE 3** (timed out by runner's 8-min wrapper, rc=124). Never reaches real NPU.
- **Context:** `XBLOCK=128, RBLOCK_0=64, block_dim=2`, x[256]→scalar, "RCore + SyncAll<false> soft sync" template. Compile OK.
- **Hypothesis:** RCore + SyncAll soft-sync at `block_dim=2` deadlocks the simulator (same multicore-sync family as #3).
- **Fix session:** repro with camodel runtime-session; inspect the SyncAll<false> soft-sync emission for full-reduce; add sim watchdog.
- Logs: `/tmp/npu-real-logs/20260521-sweep/full-reduce-e2e/` (sim.log frozen at STAGE 3).

---

### Cross-cutting
- #1 is real-HW-only (sim blind to UB OOB) → real-NPU verification required after fix.
- #3 + #4 are camodel **sim hangs** that block before real NPU; both at `block_dim=2` multicore sync — likely one shared root cause in the multicore sync path. A sim-side watchdog/timeout would stop them wedging the CI harness (and prevent orphaned hung containers).
- Re-verify via runner session: `BACKEND/--case <case>` through `scripts/sync-and-submit.sh` on device 7.
