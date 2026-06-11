// 3D → 1D reduce with adjacent multi-reduce-axes — end-to-end example.
//
// Computation:
//   out[a] = sum_{r1, r2}( x[a, r1, r2] )
//
// Input  shape: x[A, R1, R2]   f32
// Output shape: out[A]          f32
//
// Iterator types: ["parallel", "reduction", "reduction"].  Both reduction
// iters are adjacent (no parallel axis between them), so the Collapse pass
// merges them into a single R axis of extent R1*R2.  After that the kernel
// is identical to a 2D AR-pattern reduce: same FullLoad template, same
// reduce_sum_2d_l2 (AR layout) emission.
//
// %init is the DPS accumulator; the kernel zero-initializes its per-tile
// VECCALC accumulator internally so the runtime-allocated output buffer
// does not need to be pre-zeroed.
//
// Note: the **non-collapsible** form (parallel axis BETWEEN two reduction
// iters, e.g. `out[a] = sum_{r1, r2} x[r1, a, r2]`) is NOT yet supported —
// ComputeConversion assumes operand layout = [parallel++, reduction++]
// and the interleaved memory order doesn't satisfy that.  Tracked as
// future work in project_af_scheduler_port memory.

module {
  func.func @multi_r_adj(%x: tensor<8x16x32xf32>,
                          %init: tensor<8xf32>) -> tensor<8xf32> {
    %r = linalg.generic {
      indexing_maps = [affine_map<(a, r1, r2) -> (a, r1, r2)>,
                       affine_map<(a, r1, r2) -> (a)>],
      iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%x : tensor<8x16x32xf32>)
      outs(%init : tensor<8xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %t = arith.addf %v, %acc : f32
      linalg.yield %t : f32
    } -> tensor<8xf32>
    return %r : tensor<8xf32>
  }
}
