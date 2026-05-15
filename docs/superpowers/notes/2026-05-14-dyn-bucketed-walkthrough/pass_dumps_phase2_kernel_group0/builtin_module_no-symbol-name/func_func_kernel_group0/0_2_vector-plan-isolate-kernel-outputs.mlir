// -----// IR Dump After VectorPlanIsolateKernelOutputs (vector-plan-isolate-kernel-outputs) //----- //
func.func private @kernel_group0(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>, %arg3: tensor<?x?x?xf32>, %arg4: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = bufferization.alloc_tensor() copy(%arg4) : tensor<?x?xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
    %2 = arith.addf %in, %in_0 : f32
    %3 = arith.mulf %2, %in_1 : f32
    %4 = arith.addf %3, %in_2 : f32
    %5 = arith.addf %out, %4 : f32
    linalg.yield %5 : f32
  } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

