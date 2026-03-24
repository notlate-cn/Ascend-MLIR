#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @broadcast_add_reducesum(%arg0: tensor<?xf16>, %arg1: tensor<?x?xf16>) -> tensor<?xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf16>
    %0 = tensor.empty(%dim) : tensor<?xf16>
    %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<?xf16>) -> tensor<?xf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?xf16>, tensor<?x?xf16>) outs(%1 : tensor<?xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %3 = arith.addf %in, %in_0 : f16
      %4 = arith.addf %out, %3 : f16
      linalg.yield %4 : f16
    } -> tensor<?xf16>
    return %2 : tensor<?xf16>
  }
}

