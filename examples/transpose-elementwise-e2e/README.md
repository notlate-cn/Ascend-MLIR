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

`run.sh` runs the full `--auto-fuse-codegen` pipeline + the simulator and
asserts `session.validation=pass`.  The codegen path is also regression-tested
by `test/Conversion/Collapse/tile-fuse-vector-transpose.mlir` (absorption +
tiled IR shape).
