# BERT tiny real-NPU integrated run — group20 fault — handoff (2026-05-26)

Runner session ran tiny BERT (HF BertLayer, hidden=64/heads=1/seq=8, fp32) on real 910C
device 7 via `network_runner.py --backend npu`. Branch `dev-network` @ `99957e9a` (current
HEAD, includes the 2026-05-25 "fall back inner-tile params to extent" fix). Cross-ref:
[[project_bert_bringup]] (which records BERT tiny PASS max_diff=1.4e-5 on a prior commit
state), [[project_real_npu_aclnn_direct]], [[project_real_npu_test_checklist]].

## TL;DR
- Phases 1–4 clean on real HW. **All 15 vector groups pass autotune standalone** with
  max_diff=0 (incl. group20: v0 XBLOCK=256 cycles=923455, v1 XBLOCK=8 cycles=1028674).
- **Phase-5 integrated run fails at `kernel_group20__v0`**: `rtStreamSynchronize rc=507035`.
  Plog `errorStr: The GM address accessed by scalar exceeds 48 bits` subErrType:4,
  fixp_error0=0xfa3f211, blk:0 sublk:0 thread:0 (block_dim=1 picked by best.json).
- Same fault class as encoder group20 on dev-network (see [[project_real_npu_aclnn_direct]]
  point 3) — **wild scalar GM addr / stride**, NOT VEC UB tile overrun, NOT timeout/trap,
  NOT the f16 unaligned-tail thing the AF AlignmentStrategy port addresses.
- Standalone PASS + integrated FAIL points at **front-of-graph buffer-wiring carrying
  garbage forward**, the same `(a)` class the encoder handoff calls out.
- Regression vs the 2026-05-22 BERT tiny PASS recorded in [[project_bert_bringup]]: that
  was on the `bert-bringup` branch state after `43211908`+`c70d82b8`; this 99957e9a path
  goes farther (gets to group20 in phase5) but doesn't complete.

## 2026-05-26 diagnostic rerun update (corrects two earlier guesses)

A follow-up SKIP_AUTOTUNE+TRACE_LAUNCH rerun (workdir `w_npu_diag`, job
`20260526-060132-99957e9a-local-cmd`, evidence at
`/tmp/npu-real-logs/20260526-bert-diag/logs/custom-cmd.log`) sharpens the diagnosis:

1. **The fault reproduces with SKIP_AUTOTUNE=1** — same `kernel_group20__v0` rc=507035
   "GM address accessed by scalar exceeds 48 bits". Autotuner ruled out.
2. **Same run's phase-3 reference (in-pipeline default-tiling test) successfully runs
   `kernel_group20__v0`** on its isolated test buffers (tiling=256,256, output_after_d2h
   shows valid f32). So the kernel binary + tiling combo is correct; only the phase-5
   integrated path explodes.
3. **CORRECTION to the earlier 'tiling pack 错位' guess (it was wrong).** group20 was
   always launched with `tiling.word[0]=256, word[1]=256` in BOTH phase-3 and phase-5,
   matching `best.json` XBLOCK=XBLOCK_SUB=256. The `tiling.word=64,64` lines I attributed
   to group20 in the prior handoff belong to group16/group24 (which also launched and
   succeeded). No packTiling index bug here — please disregard the 9834c1a9-analog lead.
4. **Smoking gun for the buffer-wiring class.** Phase-5 group20 launch dumps:
   ```
   [npu-launch] input[2].host bytes=8192 sample_bytes=64 sample_hex=
     90221faaffff0000 90221faaffff0000 d0186fb5aaaa0000 d0186fb5aaaa0000
     0000000000000000 0000000000000000 0000000000000000 0000000000000000
   [npu-launch] input[2].h2d_roundtrip match=yes
   ```
   That `0x0000ffff aa1f2290` / `0x0000aaaa b56f18d0` byte pattern is the typical
   host-pointer encoding (low 4B = low ptr, high 4B = 0x0000ffff sign extension).
   The host buffer staged into device-side `input[2]` for group20 contains **leaked
   host pointers / freed-allocator metadata, not f32 data**, and that pattern is then
   faithfully H2D'd to device. group2/9/16 also show similar `90211faaffff0000` /
   `e0211faaffff0000` patterns in their `input[2]` — same class, but only group20's
   addressing path turns the bad bytes into a wild scalar GM address.
5. **ASK 3's "in_2 has NaN/0.84-zeros" is a red herring.** `kernel_group20.mlir` body
   `^bb0(%in, %in_2, %out)` only uses `%in` and `%in_2`; `%out` (the DPS init = arg2)
   is not read by the body. The intermediates_default NPY for in_2 is garbage in both
   the passing phase-3 run and the failing phase-5 run — so the NaN content does not
   functionally matter. The bug is **not** "in_2's values"; it is "arg2's host buffer
   in phase-5 is pre-initialized with pointer metadata that the codegen happens to
   load/address-compute against, blowing the 48-bit GM scalar address".
6. **What's likely fed into arg2 in phase-5:** `arg2` is the DPS-init for the
   `linalg.generic` Bias-Add (typed `tensor<8x256xf32>`, 8 KB). In the phase-5
   integrated graph this almost certainly comes from an upstream tensor.empty / 
   alloc that `network_host.cpp` orchestrates. If `network_host` allocates a host
   buffer with `::operator new` (no zero-init) and stages it host→device without
   first clearing it, that buffer holds whatever the host allocator's free-list had
   — exactly the `0x0000ffff aa…` pointer pattern we see. (Cross-ref: [[project_real_npu_aclnn_direct]]
   `085ad7a2` already added `stageToHost/stageDeviceOut` for aclnn — the same audit
   is needed for the AscendC DPS-init path, in particular for kernels whose body
   doesn't write every element via `outs`.)

## Repro (~2 min build + ~2 min run on shared 910C device 7)

Stage inputs once, then submit:
```bash
# from x86 dev box, repo root, after `source examples/env_gser.sh`
SSHPASS=$ASCEND_MLIR_CI_SSH_PASSWORD sshpass -e scp -P 141 \
  /tmp/bert_e2e/tiny/{step0_linalg.mlir,input_0.npy,expected_0.npy} \
  root@113.46.10.114:/data/gser/bert_e2e/tiny/

bash scripts/sync-and-submit.sh --cmd \
 'PYTHONPATH=python timeout 900 python3 python/network_runner.py \
   --input-linalg /data/gser/bert_e2e/tiny/step0_linalg.mlir \
   --inputs /data/gser/bert_e2e/tiny/input_0.npy \
   --expected /data/gser/bert_e2e/tiny/expected_0.npy \
   --workdir /data/gser/bert_e2e/tiny/w_npu \
   --soc Ascend910B1 --atol 1e-2 --rtol 1e-2 --backend npu'
```
Inputs locally generated by `bash examples/bert-e2e/run.sh 1 /tmp/bert_e2e/tiny`
(phase-1 only; that script's later phases can be ignored — we drive phase-5 via the
direct `--cmd` above).

## kernel_group20 = Bias-Add + GELU(erf form)

`/data/gser/bert_e2e/tiny/w_npu/groups/kernel_group20.mlir` (= local
`/tmp/npu-real-logs/20260526-bert-tiny/group20/kernel_group20.mlir`):
```
func @kernel_group20(arg0: 8x256xf32, arg1: 256xf32, arg2: 8x256xf32) -> 8x256xf32 {
  %0 = linalg.generic ... %0[i,j] = addf(arg0[i,j], arg1[j])         // bias add
  %2 = linalg.generic ... %2[i,j] = %0 * 0.5 * (1 + erf(%0/1.4142))  // GELU
}
```
Shape: 8×256=2048 f32 = 8 KB inputs/output. arg2 is the DPS init for the bias-add
linalg.generic, so the runtime sees 3 inputs + 1 output. Mirrors the BERT FFN
intermediate (intermediate_size=256). GELU body is the path enabled by `c70d82b8`
(ComputeConversion divf+erf).

## Evidence

### Phase 5 launch + fault (from `custom-cmd.log`)
```
[npu-launch] kernel=kernel_group20__v0 binary=/data/gser/bert_e2e/tiny/w_npu/artifacts/kernel_group20__v0/kernel_group20__v0.bin
             device_id=7 block_dim=1 args_count=7 args_size=56
[npu-launch] inputs=3 outputs=1 workspace_size=16777216 tiling_bytes=16 tiling_words=2
[npu-launch] input[0] bytes=8192 ptr=0x12c0c0015000 align512=yes
[npu-launch] input[1] bytes=1024 ptr=0x12c0c0018000 align512=yes
[npu-launch] input[2] bytes=8192 ptr=0x12c0c0019000 align512=yes
[npu-launch] output[0] bytes=8192 ptr=0x12c0c001c000 align512=yes
[npu-launch] workspace bytes=16777216 ptr=0x12c081200000 align512=yes
[npu-launch] tiling.word[0]=64    # NOTE: tiling words ≠ XBLOCK=256 from best.json
[npu-launch] tiling.word[1]=64
hostLaunchAscendCKernel(kernel_group20__v0) session.run failed:
  [npu:kernel_launch] rtStreamSynchronize failed: rc=507035
```
**Tiling mismatch worth a look:** best.json says XBLOCK=XBLOCK_SUB=256 but the launch
prints `tiling.word[0]=64 tiling.word[1]=64`. Could be a separate index/order bug in
how phase-5 packs tilings vs how the kernel reads them; if word[0/1] are interpreted
as XBLOCK/SUB the kernel runs a 64-tile pattern over 256-wide data and the inner-most
loop's GM address can wrap.

### plog
`logs/plog/plog-errorStr.txt`:
```
errorStr: The GM address accessed by scalar exceeds 48 bits.
fixp_error0 info: 0xfa3f211, fixp_error1 info: 0xf1,
fsmId:1 tslot:0 thread:0 ctxid:0 blk:0 sublk:0 subErrType:4
```

### Autotune standalone results (`kernel_group20_best.json`)
```
v0: XBLOCK=256 XBLOCK_SUB=256  cycles=923455  passed=true  max_diff=0
v1: XBLOCK=8   XBLOCK_SUB=8    cycles=1028674 passed=true  max_diff=0
best: v0 block_dim=1
```

### Run order to fault
Phase 5 reached `kernel_group16__v0` (Vector, args=8 — likely the up-projection feeding
GELU) then `kernel_group20__v0` immediately; group20 is the SECOND host-launch in phase-5
integrated. Plenty of phase-1 aclnn (Matmul/LayerNorm/FA/Transpose) ran before; those
would have written into the device buffers that feed group20.

### Artifacts staged for the fix session
`/tmp/npu-real-logs/20260526-bert-tiny/` (repo-external, per checklist #3):
- `logs/custom-cmd.log` (377 KB), `logs/build-*.log`, `logs/plog/{plog-errorStr.txt,npu-smi.txt}`
- `group20/kernel_group20.mlir`, `kernel_group20.cpp` (18 KB), `kernel_group20_lowered.mlir`
  (32 KB), `kernel_group20_{space,__v0_space,__v1_space,best,family}.json`
- `submit.log` (local wrapper).

Remote (untouched, available for re-pull): `/data/gser/bert_e2e/tiny/w_npu/` —
groups/ + per-kernel cpp/lowered + autotune profiles + `intermediates_default/`
(reference NPY for each kernel's inputs/outputs from default-tilings run; useful for
diffing what phase-5 actually feeds group20 vs. what autotuner fed it).

## Suggested fix directions (for code-fix session — runner does not implement)

Updated after the 2026-05-26 diagnostic (most-likely first):

1. **DPS-init host buffer not zero-initialized in `network_host.cpp` (or never
   D2D-copied) for `arg2`.** Concrete asks:
   - Find where `network_host.cpp` allocates the host buffer that becomes group20's
     `input[2]` (= DPS init for the bias-add `linalg.generic`'s `outs`). Likely a
     `::operator new` / `new float[2048]` without `std::memset(buf, 0, …)`.
   - Either zero-init it OR — if upstream there is a producer tensor that should
     supply this buffer — wire that producer's output (aclnn or AscendC) into
     group20's `arg2` instead of letting `network_host` create a fresh garbage one.
   - Verify with the dumped hex: a fix should change `input[2].host` for group20
     in phase-5 from the `90221faaffff0000 …` pointer pattern to either all-zeros
     or actual upstream tensor data.
2. **Audit AscendC DPS-init staging the same way `085ad7a2` did for aclnn.** The
   2026-05-23 mixed-mem fix added `stageToHost/stageDeviceOut` for aclnn `run_*`
   device branches; the AscendC `hostLaunchAscendCKernel` host-in/host-out path may
   need the symmetric audit for DPS init buffers (kernels whose body doesn't write
   every output element via the linalg.generic body — but in this kernel the body
   DOES write every element via `linalg.yield %3`, so the DPS init content
   shouldn't be observable. Suggests the kernel codegen path is, despite that,
   loading or addressing against arg2's GM region before/while writing it — worth
   looking at `kernel_group20.cpp` and `kernel_group20_lowered.mlir` for whether
   the bias-add op produces a load+add+store sequence that touches arg2 GM with
   an offset derived from the loaded value).
3. **Where the pointer-bytes come from:** the leading `90221faaffff…` matches the
   sample also seen in earlier `input[2]` dumps for group2/9/16 (4 KB / 2 KB
   buffers). The recurrence across kernels strongly suggests the same allocator
   region is being reused across DPS-init slots without zero. Search `network_host.cpp`
   generation logic (likely `lib/Conversion/.../AclnnBackend.*`) for how DPS-init
   slots are emitted vs producer outputs.

Less likely (now): tiling pack mismatch (verified equal phase-3 vs phase-5);
multicore (block_dim=1); f16 alignment (fp32); GELU codegen (standalone passed).

## 2026-05-26 zero-init fix verify — RED HERRING confirmed

dev session pushed `e4fc1b73 fix(aclnn-backend): zero-init host buffers for tensor.empty
+ kernel output bindings` and asked for a verify. Job `20260526-062411-e4fc1b73-local-cmd`,
evidence at `/tmp/npu-real-logs/20260526-bert-fix/`.

- **Fix is applied and visible:** `network_host.cpp` contains 21 `std::memset` calls.
  All earlier kernels' `input[2].host` dumps switched from `90221faaffff0000 …` pointer
  bytes to `0000000000000000 …` (h2d_roundtrip match=yes).
- **group20 STILL fails identically:** `rtStreamSynchronize rc=507035`, errorStr
  unchanged ("GM address accessed by scalar exceeds 48 bits" subErrType:4).
- **But fixp_error codes shifted:** was `fixp_error0=0xfa3f211 fixp_error1=0xf1`, now
  `fixp_error0=0xd1 fixp_error1=0x1`. Different micro-state in the same fault. This
  is **strong evidence the kernel does read arg2 GM** and the value affects the
  scalar-address computation — BUT both content values (pointer bytes AND zeros)
  trigger the fault. So buffer contents are not the determining factor.
- **Same-run phase-3 standalone for group20 PASSED with the SAME zero input[2]**
  (device_id=0, harness.cpp launcher). Phase-5 integrated (device_id=7, network_host.cpp
  launcher) FAILED on the SAME kernel binary with the SAME tiling (256,256) on the
  SAME input bytes. **Therefore the differential is in the launcher path, not the
  kernel codegen and not the buffer content.**
- **What's different between phase-3 harness vs phase-5 network_host:**
  - device_id=0 (phase-3) vs device_id=7 (phase-5) — same card type, different slot
  - launcher = `runtime-session`/harness.cpp (phase-3) vs aclnn-backend-generated
    `network_host.cpp` (phase-5)
  - Workspace base, tiling-struct layout, and aux ptr-arg ordering are all candidates
    for what `network_host.cpp` packs differently than the autotuner/harness path.
  - `args_count=7 args_size=56` reported identical in both, but the ACTUAL POINTERS
    that those 7 slots carry are determined by the launcher — and the kernel's
    scalar-address fault implies one of them is decoded as a length/stride rather
    than a base pointer (or vice versa) in network_host's call site.

## Sharpened fix directions after zero-init verify
1. **Diff `network_host.cpp` vs harness.cpp for how group20's 7 args are populated.**
   The 7 slots are: 3 input ptrs + 1 output ptr + workspace ptr + tiling ptr + (one
   more — maybe block_dim or aux). The order or one of those slots is being mis-set
   by `network_host`.
2. **Force phase-5 onto device_id=0** (the slot phase-3 used) to rule out device-slot
   peculiarities. If it still fails → 100% launcher path; if it now passes → unlikely
   but would shift to device-slot.
3. **Read `kernel_group20.cpp`'s parameter list and compare to the host-side call site
   `network_host.cpp` for group20.** A single shifted argument index would explain
   the scalar address fault perfectly.

## 2026-05-26 🎯 BREAKTHROUGH — bug 锁定 kernel codegen,与 launcher 无关

v7 用 `build-runtime-session-run-only/bin/runtime-session` 在真机 device 7 跑
standalone(完全绕开 network_host.cpp / aclnn-backend):

| Kernel | Body | 真机 standalone | Tiling | 备注 |
|---|---|---|---|---|
| **kernel_group16** | broadcast addf 4-in | ✅ **PASS rc=0** | 64,64 | 真机 ptr 0x12c0c... PASS |
| **kernel_group20** | bias-add + GELU(Erf) 3-in | ❌ **FAIL rc=507035** | 256,256 | 同 device 同 ptr 范围 FAIL |

Job `20260526-074352-e4fc1b73-local-cmd`, evidence
`/tmp/npu-real-logs/20260526-bert-real-standalone-v7/logs/custom-cmd.log`.

排除项(逐项 falsified):
- 不是 launcher 路径(standalone runtime-session 已绕过 network_host)
- 不是 buffer 内容(input[2] 都是 zero_init,group16 也是 zero,但 group16 PASS)
- 不是 LD_LIBRARY_PATH / sim stub(已 strip 且用 run-only binary)
- 不是真机高位指针地址(group16 同样 0x12c0c... 范围 PASS)
- 不是 tiling pack 错位(group16 tiling 64,64 也用了同样 packTiling 机制)
- 不是 workspace size(8192 vs 16777216 都试过,都炸)

剩余唯一变量 = **group20 的 Erf-GELU body** (`arith.divf(x, √2) + math.erf(...) + addf + mulf + mulf`)。
group16 是纯 broadcast addf,没有 divf 或 erf。

回顾 [[project_bert_bringup]]:`c70d82b8 ComputeConversion 支持 divf(DivL2Op)+erf
(AscendC::Erf verbatim)`——**这俩 op 的真机 codegen 极可能就是 bug 所在**,
camodel sim 兜过去了。

## 推荐下一步(给 code-fix session)
1. **diff `kernel_group20_lowered.mlir` vs `kernel_group16_lowered.mlir`** 的
   compute body —— group20 应该有 `DivL2Op` + `AscendC::Erf`(或类似)的
   生成代码,group16 完全没有。
2. **审 `ComputeConversion.cpp` 里 c70d82b8 加的 divf/erf 处理逻辑**,特别是
   生成的标量地址算术、立即数加载,看有没有"sim 接受/真机拒绝"的子模式。
3. **写一个 minimal 真机单测** —— 一个只含 `arith.divf` 或 `math.erf` 的
   linalg.generic,看是单独 divf 炸,还是 erf 炸,还是组合才炸。
4. workaround(若短期需要 BERT 跑通):把 erf-GELU 走 aclnn(`run_GELU`?
   `aclnnGeluV2` 存在),不走片上融合 —— 牺牲 transformer 唯一可融 epilogue
   但能 unblock。

## Why only group20 dies even though group2/9/16 also see pointer-byte input[2]

The body of group20 contains `arith.divf` + `math.erf` + `addf` + `mulf` on every
element. If any element decodes the pointer-byte pattern as a denormal/inf/NaN,
later codegen-emitted intrinsics (Duplicate? broadcast load?) may compute a scalar
offset from a vector reduction or status flag — only group20 has divf+erf in the
op list, and erf in particular is implemented as a polynomial that can chain
through several scalar ops before final SetMaskNorm. The fix should still be at
the buffer-init level (item 1), not in the kernel.

## Verify after fix (runner)
Same repro above; PASS = rc=0, `network.output[0]: max_diff <= 1e-2` in `custom-cmd.log`.
Memory baseline (2026-05-22 PASS) was max_diff=1.4e-5; expect comparable.
