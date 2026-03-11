#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @ewop_broadcast_gather(%arg0: tensor<?x?xf16>, %arg1: tensor<?xi32>, %arg2: tensor<?xf16>) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_0 = tensor.dim %arg1, %c0 : tensor<?xi32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"], library_call = "broadcast_add_gathered"} ins(%arg0, %arg2 : tensor<?x?xf16>, tensor<?xf16>) outs(%0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %2 = arith.addf %in, %in_1 : f16
      linalg.yield %2 : f16
    } -> tensor<?x?xf16>
    return %1 : tensor<?x?xf16>
  }
}

