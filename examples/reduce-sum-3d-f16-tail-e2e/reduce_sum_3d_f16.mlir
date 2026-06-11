// 3D reduce-sum end-to-end example — f16 variant.
//   out[d0,d1] = sum_{d2}( x[d0,d1,d2] )
//
// Same iteration structure as reduce-sum-3d-e2e; element type is f16.

#map_full   = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_reduce = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @reduce_sum_3d_f16(%x: tensor<?x?x?xf16>,
                                %init: tensor<?x?xf16>) -> tensor<?x?xf16> {
    %out = linalg.generic {
        indexing_maps = [#map_full, #map_reduce],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%x : tensor<?x?x?xf16>)
        outs(%init : tensor<?x?xf16>) {
    ^bb0(%in: f16, %acc: f16):
      %v = arith.addf %acc, %in : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    return %out : tensor<?x?xf16>
  }
}
