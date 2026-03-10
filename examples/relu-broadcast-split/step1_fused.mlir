#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @ewop_broadcast_split(%arg0: tensor<?x?xf16>, %arg1: tensor<?xf16>, %arg2: tensor<?xf16>, %arg3: tensor<?xf16>) -> (tensor<?x?xf16>, tensor<?x?xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %dim_1 = tensor.dim %arg2, %c0 : tensor<?xf16>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?xf16>) outs(%0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_3: f16, %out: f16):
      %5 = arith.maximumf %in, %cst : f16
      %6 = arith.addf %5, %in_3 : f16
      linalg.yield %6 : f16
    } -> tensor<?x?xf16>
    %extracted_slice = tensor.extract_slice %1[0, 0] [%dim, %dim_1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
    %2 = tensor.empty(%dim, %dim_1) : tensor<?x?xf16>
    %3 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"], library_call = "split_scale0"} ins(%extracted_slice, %arg2 : tensor<?x?xf16>, tensor<?xf16>) outs(%2 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_3: f16, %out: f16):
      %5 = arith.mulf %in, %in_3 : f16
      linalg.yield %5 : f16
    } -> tensor<?x?xf16>
    %extracted_slice_2 = tensor.extract_slice %1[0, %dim_1] [%dim, %dim_1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
    %4 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"], library_call = "split_scale1"} ins(%extracted_slice_2, %arg3 : tensor<?x?xf16>, tensor<?xf16>) outs(%2 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_3: f16, %out: f16):
      %5 = arith.mulf %in, %in_3 : f16
      linalg.yield %5 : f16
    } -> tensor<?x?xf16>
    return %3, %4 : tensor<?x?xf16>, tensor<?x?xf16>
  }
}

