// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower | FileCheck %s --implicit-check-not=linalg.generic

// Softmax: tests arith.subf, math.exp, arith.maximumf reduction, arith.addf reduction, arith.divf

func.func @softmax(%input: tensor<4x32xf32>) -> tensor<4x32xf32> {
  // Step 1: row max reduction
  %max_init = tensor.empty() : tensor<4xf32>
  %row_max = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%input : tensor<4x32xf32>)
    outs(%max_init : tensor<4xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %m = arith.maximumf %x, %acc : f32
    linalg.yield %m : f32
  } -> tensor<4xf32>

  // Step 2: subtract max (broadcast), then exponentiate
  %sub_init = tensor.empty() : tensor<4x32xf32>
  %shifted = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%input, %row_max : tensor<4x32xf32>, tensor<4xf32>)
    outs(%sub_init : tensor<4x32xf32>) {
  ^bb0(%x: f32, %m: f32, %out: f32):
    %s = arith.subf %x, %m : f32
    linalg.yield %s : f32
  } -> tensor<4x32xf32>

  %exp_init = tensor.empty() : tensor<4x32xf32>
  %exps = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%shifted : tensor<4x32xf32>)
    outs(%exp_init : tensor<4x32xf32>) {
  ^bb0(%x: f32, %out: f32):
    %e = math.exp %x : f32
    linalg.yield %e : f32
  } -> tensor<4x32xf32>

  // Step 3: row sum reduction
  %sum_init = tensor.empty() : tensor<4xf32>
  %row_sum = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%exps : tensor<4x32xf32>)
    outs(%sum_init : tensor<4xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %a = arith.addf %x, %acc : f32
    linalg.yield %a : f32
  } -> tensor<4xf32>

  // Step 4: divide by sum (broadcast)
  %out_init = tensor.empty() : tensor<4x32xf32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%exps, %row_sum : tensor<4x32xf32>, tensor<4xf32>)
    outs(%out_init : tensor<4x32xf32>) {
  ^bb0(%x: f32, %s: f32, %out: f32):
    %d = arith.divf %x, %s : f32
    linalg.yield %d : f32
  } -> tensor<4x32xf32>

  return %result : tensor<4x32xf32>
}

// CHECK: func.func @softmax
// CHECK: arith.maximumf
// CHECK: math.exp
// CHECK: ascendc.div_l2
// CHECK: return
