// -----// IR Dump After LinalgElementwiseOpFusionPass (linalg-fuse-elementwise-ops) //----- //
func.func private @kernel_group0(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>, %arg3: tensor<?x?x?xf32>, %arg4: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%arg4 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
    %1 = arith.addf %in, %in_0 : f32
    %2 = arith.mulf %1, %in_1 : f32
    %3 = arith.addf %2, %in_2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

