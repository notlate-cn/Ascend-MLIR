#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @ewop_broadcast_transpose(%arg0: tensor<?x?xf16>, %arg1: tensor<?xf16>, %arg2: tensor<?xf16>) -> tensor<?x?xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %0 = tensor.empty(%dim_0, %dim) : tensor<?x?xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f16, %in_2: f16, %out: f16):
      %2 = arith.maximumf %in, %cst : f16
      %3 = arith.addf %2, %in_1 : f16
      %4 = arith.mulf %3, %in_2 : f16
      linalg.yield %4 : f16
    } -> tensor<?x?xf16>
    return %1 : tensor<?x?xf16>
  }
}

