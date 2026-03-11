#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @ewop_broadcast_gather(%arg0: tensor<?x?xf16>, %arg1: tensor<?xi32>, %arg2: tensor<?xf16>) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %dim_1 = tensor.dim %arg1, %c0 : tensor<?xi32>
    %0 = tensor.empty(%dim, %dim_1) : tensor<?x?xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"], library_call = "gather_by_index"} ins(%arg1, %arg0 : tensor<?xi32>, tensor<?x?xf16>) outs(%0 : tensor<?x?xf16>) {
    ^bb0(%in: i32, %in_2: f16, %out: f16):
      linalg.yield %in_2 : f16
    } -> tensor<?x?xf16>
    %2 = tensor.empty(%dim, %dim_1) : tensor<?x?xf16>
    %3 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"], library_call = "broadcast_add_gathered"} ins(%1, %arg2 : tensor<?x?xf16>, tensor<?xf16>) outs(%2 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %4 = arith.addf %in, %in_2 : f16
      linalg.yield %4 : f16
    } -> tensor<?x?xf16>
    return %3 : tensor<?x?xf16>
  }
}

