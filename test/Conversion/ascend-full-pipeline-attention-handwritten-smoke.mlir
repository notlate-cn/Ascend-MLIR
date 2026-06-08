// RUN: not ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower 2>&1 | FileCheck %s

func.func @sdpa_like_supported_body(%q: tensor<2x4x8xf16>,
                                    %k: tensor<2x8x4xf16>,
                                    %v: tensor<2x4x8xf16>) -> tensor<2x4x8xf16> {
  %score_empty = tensor.empty() : tensor<2x4x4xf16>
  %score = linalg.batch_matmul
      ins(%q, %k : tensor<2x4x8xf16>, tensor<2x8x4xf16>)
      outs(%score_empty : tensor<2x4x4xf16>) -> tensor<2x4x4xf16>

  %max_empty = tensor.empty() : tensor<2x4xf16>
  %row_max = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%score : tensor<2x4x4xf16>)
    outs(%max_empty : tensor<2x4xf16>) {
  ^bb0(%x: f16, %acc: f16):
    %m = arith.maximumf %x, %acc : f16
    linalg.yield %m : f16
  } -> tensor<2x4xf16>

  %soft_empty = tensor.empty() : tensor<2x4x4xf16>
  %soft = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    ],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%score, %row_max : tensor<2x4x4xf16>, tensor<2x4xf16>)
    outs(%soft_empty : tensor<2x4x4xf16>) {
  ^bb0(%x: f16, %m: f16, %out: f16):
    %centered = arith.addf %x, %m : f16
    linalg.yield %centered : f16
  } -> tensor<2x4x4xf16>

  %out_empty = tensor.empty() : tensor<2x4x8xf16>
  %out = linalg.batch_matmul
      ins(%soft, %v : tensor<2x4x4xf16>, tensor<2x4x8xf16>)
      outs(%out_empty : tensor<2x4x8xf16>) -> tensor<2x4x8xf16>
  return %out : tensor<2x4x8xf16>
}

// CHECK: unsupported GM-output batch_matmul lowering: materialize GM tensors through cube/local buffers before lowering; scalar loop fallback is disabled
