#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2)>
module {
  func.func @kernel(%arg0: tensor<1x?x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1x?x3072xf32> {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 1.41421354 : f32
    %dim = tensor.dim %arg0, %c1 : tensor<1x?x3072xf32>
    %0 = tensor.empty(%dim) : tensor<1x?x3072xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x?x3072xf32>, tensor<3072xf32>) outs(%0 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %3 = arith.addf %in, %in_2 : f32
      linalg.yield %3 : f32
    } -> tensor<1x?x3072xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x?x3072xf32>) outs(%0 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.divf %in, %cst_1 : f32
      %4 = math.erf %3 : f32
      %5 = arith.addf %4, %cst : f32
      %6 = arith.mulf %5, %cst_0 : f32
      %7 = arith.mulf %in, %6 : f32
      linalg.yield %7 : f32
    } -> tensor<1x?x3072xf32>
    return %2 : tensor<1x?x3072xf32>
  }
}
