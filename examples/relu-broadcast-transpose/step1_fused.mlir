#map = affine_map<(d0, d1) -> (d1, 0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @relu_transpose_broadcast_add(%arg0: tensor<?x1xf16>, %arg1: tensor<?x?xf16>) -> tensor<?x?xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x1xf16>
    %dim_0 = tensor.dim %arg1, %c0 : tensor<?x?xf16>
    %0 = tensor.empty(%dim_0, %dim) : tensor<?x?xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x1xf16>, tensor<?x?xf16>) outs(%0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %2 = arith.maximumf %in, %cst : f16
      %3 = arith.addf %2, %in_1 : f16
      linalg.yield %3 : f16
    } -> tensor<?x?xf16>
    return %1 : tensor<?x?xf16>
  }
}

