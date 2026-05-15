// RUN: not afir-opt %s --ascend-normalize --ascend-kernelize 2>&1 | FileCheck %s

func.func @unsupported_linalg_result_rank_mismatch(
    %arg0: tensor<4xf32>, %arg1: tensor<4x4xf32>)
    -> (tensor<4xf32>, tensor<4x4xf32>) {
  %0:2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4xf32>, tensor<4x4xf32>)
    outs(%arg0, %arg1 : tensor<4xf32>, tensor<4x4xf32>) {
  ^bb0(%x: f32, %y: f32, %out0: f32, %out1: f32):
    %sum0 = arith.addf %x, %out0 : f32
    %sum1 = arith.addf %y, %out1 : f32
    linalg.yield %sum0, %sum1 : f32, f32
  } -> (tensor<4xf32>, tensor<4x4xf32>)
  return %0#0, %0#1 : tensor<4xf32>, tensor<4x4xf32>
}

// CHECK: error: unsupported Kernelize op semantics: linalg op has inconsistent ranked result ranks
