// Non-contiguous (a.k.a. "displaced") multi-reduce-axes end-to-end example.
//
// Computation:
//   out[a] = sum_{r1, r2}( x[r1, a, r2] )
//
// Input  shape: x[R1, A, R2]   f32
// Output shape: out[A]          f32
//
// Iterator types: ["reduction", "parallel", "reduction"].  The parallel axis
// between the two reduction iters means Collapse cannot merge them — a flat
// (parallel ++ reduction) operand layout does not exist.  FullLoad sims wrong
// (ComputeConversion's `redDim = first reduction` heuristic mis-picks RA).
// This kernel exercises the new peel-outer-R emit path: the outermost displaced
// R is peeled into a step=1 scf.for around a 2-D rank-reduced inner
// linalg.generic (≈ AutoFuse `IsNeedMultiReduce`, reduce_api_call.cpp:96).

module {
  func.func @multi_r_noncontig(%x: tensor<8x16x32xf32>,
                                %init: tensor<16xf32>) -> tensor<16xf32> {
    %r = linalg.generic {
      indexing_maps = [affine_map<(r1, a, r2) -> (r1, a, r2)>,
                       affine_map<(r1, a, r2) -> (a)>],
      iterator_types = ["reduction", "parallel", "reduction"]}
      ins(%x : tensor<8x16x32xf32>)
      outs(%init : tensor<16xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %t = arith.addf %v, %acc : f32
      linalg.yield %t : f32
    } -> tensor<16xf32>
    return %r : tensor<16xf32>
  }
}
