// Trailing-axis broadcast end-to-end example.
//
// Computation:
//   out[a,b,c] = x[a,b,c] + y[a,b]      (y broadcast over the trailing axis c)
//
// y's indexing map (a,b,c)->(a,b) projects away the trailing iteration dim c.
// After collapse([0,1],[2]) the iteration space is [ab(=32, parallel, block
// axis), c(=32, broadcast)]; c is an ordinary whole-dim parallel axis (no
// BCAST tunable) — the lowering replicates y on-chip via ascendc.broadcast_l2.

func.func @bcast_trailing(%x: tensor<8x4x32xf32>, %y: tensor<8x4xf32>) -> tensor<8x4x32xf32> {
  %o = tensor.empty() : tensor<8x4x32xf32>
  %r = linalg.generic {
    indexing_maps = [affine_map<(a,b,c) -> (a,b,c)>, affine_map<(a,b,c) -> (a,b)>, affine_map<(a,b,c) -> (a,b,c)>],
    iterator_types = ["parallel","parallel","parallel"]}
    ins(%x, %y : tensor<8x4x32xf32>, tensor<8x4xf32>) outs(%o : tensor<8x4x32xf32>) {
  ^bb0(%xi: f32, %yi: f32, %oi: f32):
    %s = arith.addf %xi, %yi : f32
    linalg.yield %s : f32
  } -> tensor<8x4x32xf32>
  return %r : tensor<8x4x32xf32>
}
