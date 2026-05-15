// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @sdpa_like(%q: tensor<2x4x8xf32>,
                     %k: tensor<2x8x4xf32>,
                     %v: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %score_empty = tensor.empty() : tensor<2x4x4xf32>
  %score = linalg.batch_matmul
      ins(%q, %k : tensor<2x4x8xf32>, tensor<2x8x4xf32>)
      outs(%score_empty : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

  %max_empty = tensor.empty() : tensor<2x4xf32>
  %row_max = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%score : tensor<2x4x4xf32>)
    outs(%max_empty : tensor<2x4xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %m = arith.maximumf %x, %acc : f32
    linalg.yield %m : f32
  } -> tensor<2x4xf32>

  %soft_empty = tensor.empty() : tensor<2x4x4xf32>
  %soft = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    ],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%score, %row_max : tensor<2x4x4xf32>, tensor<2x4xf32>)
    outs(%soft_empty : tensor<2x4x4xf32>) {
  ^bb0(%x: f32, %m: f32, %out: f32):
    %centered = arith.subf %x, %m : f32
    linalg.yield %centered : f32
  } -> tensor<2x4x4xf32>

  %out_empty = tensor.empty() : tensor<2x4x8xf32>
  %out = linalg.batch_matmul
      ins(%soft, %v : tensor<2x4x4xf32>, tensor<2x4x8xf32>)
      outs(%out_empty : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %out : tensor<2x4x8xf32>
}

// CHECK: FusionCandidateAnalysis
// CHECK: kind = "HandwrittenPattern" primitive = "HandwrittenPattern"
// CHECK-SAME: internal_ops = [0, 1, 2, 3]
// CHECK-SAME: families = ["cube"]
// CHECK: KernelPatternGraph
// CHECK: source = "HandwrittenPattern"
// CHECK-SAME: internal_ops = [0, 1, 2, 3]
// CHECK: KernelPartition
// CHECK: kernel_pattern = "kernel_0"
// CHECK-SAME: internal_ops = [0, 1, 2, 3]
