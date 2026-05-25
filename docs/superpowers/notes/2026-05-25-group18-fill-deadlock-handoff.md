# encoder group18 single-core hang — root cause (fill→GM-output empty VECOUT)

group18 = dual-output: %0=add(arg0,arg1)→out0, %1=fill0→out1 (2x8x64). Hangs
even block_dim=1 (~9min, no aicore plog) → kernel-codegen deadlock, not tiling.

Root cause: ALL 8 queues balanced, no SyncAll, so not imbalance. The fill0
output path emits an EMPTY VECOUT pipeline — alloc→EnQue→DeQue→DataCopy with NO
Duplicate writing it (kernel_group18.cpp:79-86 / 117-124): VECOUT buf enqueued
never produced → on real HW the que sync blocks forever (camodel tolerates, so
sim/encoder e2e pass). The fill PRE-pass at lib/Conversion/LinalgToAscendC/
ComputeConversion.cpp:499-530 only lowers linalg.fill writing a VECCALC (ms==11)
to DuplicateL2; a fill whose out is a kernel GM output is NOT converted → empty
VECOUT. Dual-output (one result = standalone fill) triggers it; BERT/front had none.

Fix: extend the fill lowering to a fill whose output is a kernel output buffer —
emit DuplicateL2(0) into the VECOUT tile before the store. Verify: sim no
regression; out1 = zeros; rebuild afir-opt, regen group18, confirm Duplicate in
.cpp. Reverify encoder --backend npu past group18. group2/20/26 PASS — don't touch.

## CORRECTION: not the fill PRE-pass — dual-output empty store
Extending the fill PRE-pass (ms 0/11/12, even all fills) did NOT change group18
.cpp — still 0 Duplicate, empty `EnQue(v44)`. So out1's empty VECOUT is NOT a
FillOp (canonicalized away pre-codegen): it's the dual-output store reusing
add's result, 2nd output buffer enqueued with NO producer. Real root = dual-result
codegen emits a VECOUT for the 2nd output with no compute. Fix belongs in the
multi-output store path (DataMove/IsolateKernelOutputs), not fill lowering. Encoder
sim stays 7.15e-7 regardless (camodel tolerates). Reverted; tree clean.
