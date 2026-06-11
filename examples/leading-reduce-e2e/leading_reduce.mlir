// 2D reduce-sum on the LEADING axis — end-to-end example for FullLoad + RA.
//
// Computation:
//   out[d1] = sum_{d0}( x[d0, d1] )
//
// Input  shape: x[D0, D1]   f32
// Output shape: out[D1]     f32   (axis=0 reduce)
//
// Iterator types: ["reduction", "parallel"].  The reduction iter sits BEFORE
// the parallel iter, and the only operand's physical layout is [d0, d1] →
// ReduceLayout::RA (result[a] = sum_r src[r*A + a], R = D0, A = D1).
//
// D0 is small (kept whole on-chip), D1 is the block axis.  Three things this
// example pins:
//   1. costEstimate picks FullLoad (R fits, no row-loop degrade).
//   2. ComputeConversion emits reduce_sum_2d_l2 with the RA layout.
//   3. The compiled kernel is numerically correct on the simulator.
//
// %init is the DPS accumulator; in the CANN signature it becomes the output
// slot (cann.num_inputs = 1).  The kernel zeros its per-tile VECCALC
// accumulator internally so the runtime-allocated output does not need to be
// pre-zeroed.

#map_full   = affine_map<(d0, d1) -> (d0, d1)>
#map_reduce = affine_map<(d0, d1) -> (d1)>

module {
  func.func @leading_reduce(%x: tensor<?x?xf32>,
                            %init: tensor<?xf32>) -> tensor<?xf32> {
    %out = linalg.generic {
        indexing_maps = [#map_full, #map_reduce],
        iterator_types = ["reduction", "parallel"]}
        ins(%x : tensor<?x?xf32>)
        outs(%init : tensor<?xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %v = arith.addf %acc, %in : f32
      linalg.yield %v : f32
    } -> tensor<?xf32>

    return %out : tensor<?xf32>
  }
}
