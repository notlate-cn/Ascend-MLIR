// Middle-axis broadcast end-to-end example.
//
// Computation:
//   out[a,b,c] = x[a,b,c] + y[a,c]      (y broadcast over the middle axis b)
//
// y's indexing map (a,b,c)->(a,c) projects away the middle iteration dim b.
// Iteration space [a(=8, parallel, block axis), b(=16, broadcast), c(=32,
// parallel, whole)] — a and c can't collapse because b sits between them.
// b is an ordinary whole-dim parallel axis (no BCAST tunable) — the lowering
// replicates y on-chip via ascendc.broadcast_l2.

func.func @bcast_middle(%x: tensor<8x16x32xf32>, %y: tensor<8x32xf32>) -> tensor<8x16x32xf32> {
  %o = tensor.empty() : tensor<8x16x32xf32>
  %r = linalg.generic {
    indexing_maps = [affine_map<(a,b,c) -> (a,b,c)>, affine_map<(a,b,c) -> (a,c)>, affine_map<(a,b,c) -> (a,b,c)>],
    iterator_types = ["parallel","parallel","parallel"]}
    ins(%x, %y : tensor<8x16x32xf32>, tensor<8x32xf32>) outs(%o : tensor<8x16x32xf32>) {
  ^bb0(%xi: f32, %yi: f32, %oi: f32):
    %s = arith.addf %xi, %yi : f32
    linalg.yield %s : f32
  } -> tensor<8x16x32xf32>
  return %r : tensor<8x16x32xf32>
}
