// Leading-axis broadcast end-to-end example.
//
// Computation:
//   out[a,b,c] = x[a,b,c] + y[b,c]      (y broadcast over the leading axis a)
//
// y's indexing map (a,b,c)->(b,c) projects away the leading iteration dim a.
// After collapse([1,2]) the iteration space is [a(=8, broadcast), bc(=128,
// parallel, block axis)]; a is an ordinary whole-dim parallel axis (no BCAST
// tunable) — the lowering replicates y on-chip via ascendc.broadcast_l2.

func.func @bcast_leading(%x: tensor<8x4x32xf32>, %y: tensor<4x32xf32>) -> tensor<8x4x32xf32> {
  %o = tensor.empty() : tensor<8x4x32xf32>
  %r = linalg.generic {
    indexing_maps = [affine_map<(a,b,c) -> (a,b,c)>, affine_map<(a,b,c) -> (b,c)>, affine_map<(a,b,c) -> (a,b,c)>],
    iterator_types = ["parallel","parallel","parallel"]}
    ins(%x, %y : tensor<8x4x32xf32>, tensor<4x32xf32>) outs(%o : tensor<8x4x32xf32>) {
  ^bb0(%xi: f32, %yi: f32, %oi: f32):
    %s = arith.addf %xi, %yi : f32
    linalg.yield %s : f32
  } -> tensor<8x4x32xf32>
  return %r : tensor<8x4x32xf32>
}
