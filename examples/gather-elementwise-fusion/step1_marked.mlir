#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @relu_index_select_add(%arg0: tensor<?x?xf16>, %arg1: tensor<?xi64>, %arg2: tensor<?xf16>) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %dim_1 = tensor.dim %arg1, %c0 : tensor<?xi64>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x?xf16>) outs(%0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      %6 = arith.maximumf %in, %cst : f16
      linalg.yield %6 : f16
    } -> tensor<?x?xf16>
    %2 = tensor.empty(%dim, %dim_1) : tensor<?x?xf16>
    %3 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<?xi64>) outs(%2 : tensor<?x?xf16>) attrs =  {gather_dim = 1 : i64} {
    ^bb0(%in: i64, %out: f16):
      %6 = linalg.index 0 : index
      %7 = linalg.index 1 : index
      %8 = arith.index_cast %in : i64 to index
      %extracted = tensor.extract %1[%6, %8] : tensor<?x?xf16>
      linalg.yield %extracted : f16
    } -> tensor<?x?xf16>
    %4 = tensor.empty(%dim, %dim_1) : tensor<?x?xf16>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %arg2 : tensor<?x?xf16>, tensor<?xf16>) outs(%4 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %6 = arith.addf %in, %in_2 : f16
      linalg.yield %6 : f16
    } -> tensor<?x?xf16>
    return %5 : tensor<?x?xf16>
  }
}

