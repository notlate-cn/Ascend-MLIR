#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @ewop_broadcast_concat(%arg0: tensor<?xf16>, %arg1: tensor<?x?xf16>, %arg2: tensor<?xf16>, %arg3: tensor<?x?xf16>) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf16>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf16>
    %0 = arith.muli %dim, %c2 : index
    %1 = tensor.empty(%0, %dim_0) : tensor<?x?xf16>
    %2 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"], library_call = "broadcast_add"} ins(%arg0, %arg1 : tensor<?xf16>, tensor<?x?xf16>) outs(%2 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %5 = arith.addf %in, %in_2 : f16
      linalg.yield %5 : f16
    } -> tensor<?x?xf16>
    %inserted_slice = tensor.insert_slice %3 into %1[0, 0] [%dim, %dim_0] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"], library_call = "broadcast_mul"} ins(%arg2, %arg3 : tensor<?xf16>, tensor<?x?xf16>) outs(%2 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %5 = arith.mulf %in, %in_2 : f16
      linalg.yield %5 : f16
    } -> tensor<?x?xf16>
    %inserted_slice_1 = tensor.insert_slice %4 into %inserted_slice[%dim, 0] [%dim, %dim_0] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
    return %inserted_slice_1 : tensor<?x?xf16>
  }
}

