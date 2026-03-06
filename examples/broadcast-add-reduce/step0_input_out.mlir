#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @broadcast_add_reducesum(%arg0: tensor<?xf16>, %arg1: tensor<?x?xf16>) -> tensor<?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf16>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf16>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?xf16>) outs(%0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?xf16>
    %2 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %arg1 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%2 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %7 = arith.addf %in, %in_1 : f16
      linalg.yield %7 : f16
    } -> tensor<?x?xf16>
    %cst = arith.constant 0.000000e+00 : f16
    %4 = tensor.empty(%dim) : tensor<?xf16>
    %5 = linalg.fill ins(%cst : f16) outs(%4 : tensor<?xf16>) -> tensor<?xf16>
    %6 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<?x?xf16>) outs(%5 : tensor<?xf16>) {
    ^bb0(%in: f16, %out: f16):
      %7 = arith.addf %out, %in : f16
      linalg.yield %7 : f16
    } -> tensor<?xf16>
    return %6 : tensor<?xf16>
  }
}

