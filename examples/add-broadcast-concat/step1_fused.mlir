#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @ewop_broadcast_concat(%arg0: tensor<?xf16>, %arg1: tensor<?x?xf16>, %arg2: tensor<?xf16>, %arg3: tensor<?x?xf16>) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf16>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf16>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?xf16>, tensor<?x?xf16>) outs(%0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %3 = arith.addf %in, %in_1 : f16
      linalg.yield %3 : f16
    } -> tensor<?x?xf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2, %arg3 : tensor<?xf16>, tensor<?x?xf16>) outs(%0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %3 = arith.mulf %in, %in_1 : f16
      linalg.yield %3 : f16
    } -> tensor<?x?xf16>
    %concat = tensor.concat dim(0) %1, %2 : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
    return %concat : tensor<?x?xf16>
  }
}

