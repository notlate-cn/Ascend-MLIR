// 3D reduce-sum end-to-end example.
//
// Computation:
//   out[d0,d1] = sum_{d2}( x[d0,d1,d2] )
//
// Input  shape: x[D0, D1, D2]  f32
// Output shape: out[D0, D1]    f32
//
// Iterator types: ["parallel", "parallel", "reduction"]
// The %init operand is the DPS accumulator; in the CANN signature it becomes
// the output slot (cann.num_inputs = 1).  The generated kernel zeros its
// per-tile VECCALC accumulator internally (Duplicate 0), so the runtime-
// allocated output buffer does not need to be pre-zeroed.  Taking %init as a
// function arg (rather than a linalg.fill) also avoids forming a 2-op
// fill+reduce group, which the auto-fuse reduction-split path asserts
// against.

#map_full   = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_reduce = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @reduce_sum_3d(%x: tensor<?x?x?xf32>,
                            %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %out = linalg.generic {
        indexing_maps = [#map_full, #map_reduce],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%x : tensor<?x?x?xf32>)
        outs(%init : tensor<?x?xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %v = arith.addf %acc, %in : f32
      linalg.yield %v : f32
    } -> tensor<?x?xf32>

    return %out : tensor<?x?xf32>
  }
}
