// -----// IR Dump After LinalgFoldUnitExtentDimsPass (linalg-fold-unit-extent-dims) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @bucketed_dyn(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>, %arg3: tensor<?x?x?xf32>, %arg4: tensor<?x?x?xf32>, %arg5: tensor<?x?x?xf32>, %arg6: tensor<?x?x?xf32>, %arg7: tensor<?x?xf32>, %arg8: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%arg7 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      %3 = arith.mulf %2, %in_1 : f32
      %4 = arith.addf %3, %in_2 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg4, %arg5, %arg6 : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%arg8 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      %3 = arith.mulf %2, %in_1 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<?x?xf32>
    return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
  }
}


