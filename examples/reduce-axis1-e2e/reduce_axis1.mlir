// 3D reduce-sum on the MIDDLE axis — end-to-end example.
//
// Computation:
//   out[d0, d2] = sum_{d1}( x[d0, d1, d2] )
//
// Input  shape: x[D0, D1, D2]  f32
// Output shape: out[D0, D2]    f32   (axis=1 reduce)
//
// Iterator types: ["parallel", "reduction", "parallel"].  The reduction axis
// sits *between* two parallel axes, so the collapse pass cannot merge them and
// TilePlanGen routes this through the "split-parallel" policy: d0 is block-
// dispatched and walked one row at a time (inner step = 1), d1 (reduction) and
// d2 stay full, so each tile body is `x[row, :, :] -> out[row, :]` (contiguous
// DataCopy) and the d1-over-d2 reduction lowers to reduce_sum_2d_l2 with the
// RA layout (result[a] = sum_r src[r*A + a], R = D1, A = D2).
//
// The %init operand is the DPS accumulator; in the CANN signature it becomes
// the output slot (cann.num_inputs = 1).  The kernel zeros its per-tile
// VECCALC accumulator internally (Duplicate 0), so the runtime-allocated
// output buffer does not need to be pre-zeroed.

#map_full   = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_reduce = affine_map<(d0, d1, d2) -> (d0, d2)>

module {
  func.func @reduce_axis1(%x: tensor<?x?x?xf32>,
                           %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %out = linalg.generic {
        indexing_maps = [#map_full, #map_reduce],
        iterator_types = ["parallel", "reduction", "parallel"]}
        ins(%x : tensor<?x?x?xf32>)
        outs(%init : tensor<?x?xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %v = arith.addf %acc, %in : f32
      linalg.yield %v : f32
    } -> tensor<?x?xf32>

    return %out : tensor<?x?xf32>
  }
}
