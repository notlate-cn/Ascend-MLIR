// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower | FileCheck %s --implicit-check-not=linalg.

func.func @sdpa_like_supported_body(%q: tensor<2x4x8xf32>,
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
    %centered = arith.addf %x, %m : f32
    linalg.yield %centered : f32
  } -> tensor<2x4x4xf32>

  %out_empty = tensor.empty() : tensor<2x4x8xf32>
  %out = linalg.batch_matmul
      ins(%soft, %v : tensor<2x4x4xf32>, tensor<2x4x8xf32>)
      outs(%out_empty : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %out : tensor<2x4x8xf32>
}

// CHECK: ascend.schedule.tuning_cache = ["attention_sdpa|grouped_tile_per_block|2x4x8|2x4x4x8"]
// CHECK: ascendc.add_l2
// CHECK: return
