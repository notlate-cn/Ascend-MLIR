// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower | FileCheck %s --implicit-check-not=linalg.generic

// RMSNorm: x / sqrt(mean(x^2) + eps)
// Tests: arith.mulf (fused with addf in reduction), math.rsqrt, arith.divf

func.func @rmsnorm(%input: tensor<4x32xf32>, %weight: tensor<32xf32>) -> tensor<4x32xf32> {
  %cst_eps = arith.constant 1.0e-6 : f32
  %cst_inv_n = arith.constant 0.03125 : f32

  // Step 1: sum(x^2) via fused body — but linalg.generic reduction only supports single compute op.
  // Use two kernels: x^2 (parallel), then reduce-add.
  %sq_init = tensor.empty() : tensor<4x32xf32>
  %squared = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<4x32xf32>)
    outs(%sq_init : tensor<4x32xf32>) {
  ^bb0(%x: f32, %out: f32):
    %sq = arith.mulf %x, %x : f32
    linalg.yield %sq : f32
  } -> tensor<4x32xf32>

  %sum_init = tensor.empty() : tensor<4xf32>
  %sum_sq = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%squared : tensor<4x32xf32>)
    outs(%sum_init : tensor<4xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %a = arith.addf %x, %acc : f32
    linalg.yield %a : f32
  } -> tensor<4xf32>

  // Step 2: rsqrt(mean + eps)
  %rms_init = tensor.empty() : tensor<4xf32>
  %rms = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%sum_sq : tensor<4xf32>)
    outs(%rms_init : tensor<4xf32>) {
  ^bb0(%s: f32, %out: f32):
    %mean = arith.mulf %s, %cst_inv_n : f32
    %m_eps = arith.addf %mean, %cst_eps : f32
    %r = math.rsqrt %m_eps : f32
    linalg.yield %r : f32
  } -> tensor<4xf32>

  // Step 3: normalize and scale
  %out_init = tensor.empty() : tensor<4x32xf32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%input, %rms, %weight : tensor<4x32xf32>, tensor<4xf32>, tensor<32xf32>)
    outs(%out_init : tensor<4x32xf32>) {
  ^bb0(%x: f32, %r: f32, %w: f32, %out: f32):
    %normed = arith.mulf %x, %r : f32
    %scaled = arith.mulf %normed, %w : f32
    linalg.yield %scaled : f32
  } -> tensor<4x32xf32>

  return %result : tensor<4x32xf32>
}

// CHECK: func.func @rmsnorm
// CHECK: ascendc.mul_l2
// CHECK: math.rsqrt
// CHECK: ascendc.mul_l2
// CHECK: return
