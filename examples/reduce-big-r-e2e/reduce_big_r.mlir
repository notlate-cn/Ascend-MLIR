// 2D reduce-sum with a too-large reduction axis — end-to-end example.
//
// Computation:
//   out[a] = sum_{r}( x[a, r] )
//
// Input  shape: x[A, R]  f32   (A = 8, R = 65536)
// Output shape: out[A]   f32   (axis=1 reduce)
//
// Iterator types: ["parallel", "reduction"].  R · elem_bytes = 65536·4 = 256 KB
// far exceeds the ~32 KB per-tile budget, so TilePlanGen auto-enables the
// RBLOCK reduction-split policy: A is block-dispatched (XBLOCK rows/core, walked
// in XBLOCK_SUB chunks) and R is tile-split into RBLOCK_0 chunks.  Each tile
// body carries a VECCALC accumulator across the RBLOCK loop —
//   acc = 0;  for r-chunk: tmp = reduce(x[rows, r-chunk]);  acc += tmp
// — and after the loop stores `acc` to GM (out[rows]).
//
// The %init operand is the DPS accumulator; in the CANN signature it becomes
// the output slot (cann.num_inputs = 1).  The kernel zeros its per-tile VECCALC
// accumulator internally (Duplicate 0), so the runtime-allocated output buffer
// does not need to be pre-zeroed.

#map_full   = affine_map<(a, r) -> (a, r)>
#map_reduce = affine_map<(a, r) -> (a)>

module {
  func.func @reduce_big_r(%x: tensor<8x65536xf32>,
                          %init: tensor<8xf32>) -> tensor<8xf32> {
    %out = linalg.generic {
        indexing_maps = [#map_full, #map_reduce],
        iterator_types = ["parallel", "reduction"]}
        ins(%x : tensor<8x65536xf32>)
        outs(%init : tensor<8xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %v = arith.addf %acc, %in : f32
      linalg.yield %v : f32
    } -> tensor<8xf32>

    return %out : tensor<8xf32>
  }
}
