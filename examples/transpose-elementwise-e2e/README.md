# transpose-elementwise-e2e — tail-axis transpose, "eliminate" template

Computation: `out = relu(transpose(x, [1,0]))`, `x : [16,32] f16 → out : [32,16] f16`.

`--linalg-fuse-elementwise-ops` absorbs the named `linalg.transpose` into the
relu generic's operand indexing map (`(d0,d1)->(d1,d0)`), so by codegen there
is no transpose op — this is the "eliminate" template (vs. a "preserve"
template that keeps a separate transpose op).  On-chip, the transposed operand
tile is a row-strided subview of `x`, loaded with a per-row `DataCopy` into a
packed VECIN tile, then rearranged with `AscendC::Transpose` into the output
layout before the relu.

`AscendC::Transpose` (the basic 16×16 form) operates on 16-bit data, so this
example uses f16 and a square 16×16 inner tile (`XBLOCK_SUB == XBLOCK_SUB_0 ==
16`); the tiling-space generator must honour that constraint for transposed
operands.  Wider / non-square / non-16-multiple inner tiles need a different
lowering (`TransDataTo5HD`, or a "preserve" template) — future work.

## Status

The generated kernel is **verified bit-exact correct** against `relu(x.T)` on
the AscendC simulator (run the kernel, dump the output buffer, compare —
`max_abs_diff == 0`).

However `run.sh`'s `session.validation=pass` assertion currently **fails**:
`runtime-session --run` reports a spurious `max_abs_diff ≈ 1.98` for this
kernel (the sim output it actually writes is perfect) and then segfaults — the
same family as the pre-existing `relu-e2e` sim segfault.  This kernel differs
from the passing e2e gates in two ways that may trip the validator: a
`memref<…, strided<[?,1], offset:?>>` output out-param (bufferization wraps the
tile-fuse loop result as a fresh strided arg; the passing examples reuse a
plain passed-in `%init` memref), and an `emitasc.verbatim` (the per-row
DataCopy loop).

So this example is **not in the e2e gate** for now.  When the runtime-session
validation bug is fixed, `run.sh` should pass as-is.

The codegen path is regression-tested by
`test/Conversion/Collapse/tile-fuse-vector-transpose.mlir` (absorption + tiled
IR shape).
