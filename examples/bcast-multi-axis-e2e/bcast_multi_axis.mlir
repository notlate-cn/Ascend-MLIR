// Multi-axis broadcast end-to-end example.
//
// Computation:
//   out[d0,d1,d2] = b[d0,d1,d2] + a[d1]      (a broadcast on d0 AND d2)
//
// `a` is rank-1 [D1]; its indexing map projects away two iteration dims, so
// the lowered ascendc.broadcast_l2 has srcShape = [1, D1, 1] (two broadcast
// axes).  Exercises the ascendc-decompose-multi-axis-broadcast pass.

#map_full = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_d1   = affine_map<(d0, d1, d2) -> (d1)>

module {
  func.func @bcast_multi_axis(%a: tensor<?xf32>,
                                %b: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = tensor.dim %b, %c0 : tensor<?x?x?xf32>
    %d1 = tensor.dim %b, %c1 : tensor<?x?x?xf32>
    %d2 = tensor.dim %b, %c2 : tensor<?x?x?xf32>
    %init = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
    %out = linalg.generic {
        indexing_maps = [#map_d1, #map_full, #map_full],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%a, %b : tensor<?xf32>, tensor<?x?x?xf32>)
        outs(%init : tensor<?x?x?xf32>) {
    ^bb0(%ina: f32, %inb: f32, %o: f32):
      %v = arith.addf %inb, %ina : f32
      linalg.yield %v : f32
    } -> tensor<?x?x?xf32>
    return %out : tensor<?x?x?xf32>
  }
}
