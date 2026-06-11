// Tail-axis transpose absorbed into an elementwise consumer — end-to-end.
//
// Computation:
//   t = transpose(x, [1,0])          x : [16,32] f16  ->  t : [32,16] f16
//   out[d0,d1] = max(t[d0,d1], 0)    out : [32,16] f16   (relu)
//
// --linalg-fuse-elementwise-ops absorbs the named linalg.transpose into the
// relu generic (operand map (d0,d1)->(d1,d0)), so by codegen there is no
// transpose op left — the "eliminate" template.  On-chip the transposed
// operand tile is loaded from GM (a row-strided subview of x → per-row
// DataCopy into a packed VECIN tile) and rearranged with AscendC::Transpose
// into the output layout before the relu.
//
// f16: AscendC::Transpose (the basic 16x16 form) operates on 16-bit data.
// Tiling: d0 (=32) block-split (XBLOCK), walked XBLOCK_SUB at a time; d1 (=16)
// is fully loaded (non-ub parallel axis → whole-dim slice, no loop — §3.4).
// AscendC::Transpose needs a square 16x16 inner tile, so XBLOCK_SUB == 16;
// run.sh pins it.  block_dim = ceil(32 / XBLOCK).

#id2 = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @transpose_relu(%x: tensor<16x32xf16>) -> tensor<32x16xf16> {
    %zero = arith.constant 0.0 : f16
    %t_init = tensor.empty() : tensor<32x16xf16>
    %t = linalg.transpose ins(%x : tensor<16x32xf16>)
                          outs(%t_init : tensor<32x16xf16>)
                          permutation = [1, 0]
    %r_init = tensor.empty() : tensor<32x16xf16>
    %out = linalg.generic {
        indexing_maps = [#id2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%t : tensor<32x16xf16>)
        outs(%r_init : tensor<32x16xf16>) {
    ^bb0(%in: f16, %o: f16):
      %v = arith.maximumf %in, %zero : f16
      linalg.yield %v : f16
    } -> tensor<32x16xf16>
    return %out : tensor<32x16xf16>
  }
}
