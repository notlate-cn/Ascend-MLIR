#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @ewop_broadcast_transpose(%arg0: tensor<?x?xf16>, %arg1: tensor<?xf16>, %arg2: tensor<?xf16>) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"], library_call = "relu_bias_add"} ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?xf16>) outs(%0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %6 = arith.maximumf %in, %cst : f16
      %7 = arith.addf %6, %in_1 : f16
      linalg.yield %7 : f16
    } -> tensor<?x?xf16>
    %2 = tensor.empty(%dim_0, %dim) : tensor<?x?xf16>
    %3 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"], library_call = "transpose"} ins(%1 : tensor<?x?xf16>) outs(%2 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?xf16>
    %4 = tensor.empty(%dim_0, %dim) : tensor<?x?xf16>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"], library_call = "scale_mul"} ins(%3, %arg2 : tensor<?x?xf16>, tensor<?xf16>) outs(%4 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %6 = arith.mulf %in, %in_1 : f16
      linalg.yield %6 : f16
    } -> tensor<?x?xf16>
    return %5 : tensor<?x?xf16>
  }
}

