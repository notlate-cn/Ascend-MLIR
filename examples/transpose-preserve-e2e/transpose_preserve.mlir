// Multi-use transpose — the "preserve" template — end-to-end.
//
// Computation:
//   t = transpose(x, [1,0])           x : [32,32] f16  ->  t : [32,32] f16
//   out_a[i,j] = max(t[i,j], 0)        out_a : [32,32] f16   (relu)
//   out_b[i,j] = t[i,j] * 2            out_b : [32,32] f16
//
// `t` has two consumers, so --linalg-fuse-elementwise-ops cannot absorb the
// transpose into a single downstream indexing map — it survives to tile-fuse
// as a standalone op (= the "preserve" template, vs. the "eliminate" template
// in examples/transpose-elementwise-e2e where the single consumer absorbs it).
// classifyAxes (≈ AF GenTransposeTilingGroup) puts the input-side divergent
// axis (d0 = out's outer dim) in X (own inner tunable XBLOCK_X_0, never the
// block axis), the output-side axis (d1) in Y (the block axis).
//
// On-chip per 16x16 tile: the transpose loads its row-strided x slice into
// VECIN with a per-row DataCopy, AscendC::Transpose rearranges it into the
// output layout in a VECCALC TBuf, both consumers read that same TBuf (it
// stays on-chip — not accumulated into a full-shape tensor that would
// bufferize to a GM out-param), and the consumers write their row-strided
// out[d0_range, d1_range] tile back to GM with a per-row DataCopy.  f16 +
// square 16x16 inner tile (AscendC::Transpose basic form is 16-bit / 16x16).
// block_dim = ceil(M / XBLOCK); run.sh pins XBLOCK = XBLOCK_SUB = XBLOCK_X_0 =
// 16, M = 32 → block_dim = 2 (multi-core).

#id2 = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @transpose_preserve(%x: tensor<32x32xf16>)
      -> (tensor<32x32xf16>, tensor<32x32xf16>) {
    %zero = arith.constant 0.0 : f16
    %two = arith.constant 2.0 : f16
    %t_init = tensor.empty() : tensor<32x32xf16>
    %t = linalg.transpose ins(%x : tensor<32x32xf16>)
                          outs(%t_init : tensor<32x32xf16>) permutation = [1, 0]
    %a_init = tensor.empty() : tensor<32x32xf16>
    %a = linalg.generic {
        indexing_maps = [#id2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%t : tensor<32x32xf16>) outs(%a_init : tensor<32x32xf16>) {
    ^bb0(%v: f16, %o: f16):
      %m = arith.maximumf %v, %zero : f16
      linalg.yield %m : f16
    } -> tensor<32x32xf16>
    %b_init = tensor.empty() : tensor<32x32xf16>
    %b = linalg.generic {
        indexing_maps = [#id2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%t : tensor<32x32xf16>) outs(%b_init : tensor<32x32xf16>) {
    ^bb0(%v: f16, %o: f16):
      %m = arith.mulf %v, %two : f16
      linalg.yield %m : f16
    } -> tensor<32x32xf16>
    return %a, %b : tensor<32x32xf16>, tensor<32x32xf16>
  }
}
