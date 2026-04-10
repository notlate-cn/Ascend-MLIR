#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @ewop_broadcast_split(%arg0: tensor<?x?xf16>, %arg1: tensor<?xf16>, %arg2: tensor<?xf16>, %arg3: tensor<?xf16>, %arg4: tensor<?xf16>) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %dim = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %dim_0 = tensor.dim %arg1, %c0 : tensor<?xf16>
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [%dim_0, %dim] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
    %extracted_slice_1 = tensor.extract_slice %arg0[%dim_0, 0] [%dim_0, %dim] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
    %0 = tensor.empty(%dim_0, %dim) : tensor<?x?xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %arg1, %arg3 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
      %3 = arith.maximumf %in, %cst : f16
      %4 = arith.addf %3, %in_2 : f16
      %5 = arith.mulf %4, %in_3 : f16
      linalg.yield %5 : f16
    } -> tensor<?x?xf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_1, %arg2, %arg4 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
      %3 = arith.maximumf %in, %cst : f16
      %4 = arith.addf %3, %in_2 : f16
      %5 = arith.mulf %4, %in_3 : f16
      linalg.yield %5 : f16
    } -> tensor<?x?xf16>
    %concat = tensor.concat dim(0) %1, %2 : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
    return %concat : tensor<?x?xf16>
  }
}

