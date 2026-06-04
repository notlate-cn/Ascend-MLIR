// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @sdpa_like(%q: tensor<2x4x8xf16>,
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
    %centered = arith.subf %x, %m : f16
    linalg.yield %centered : f16
  } -> tensor<2x4x4xf16>

  %out_empty = tensor.empty() : tensor<2x4x8xf16>
  %out = linalg.batch_matmul
      ins(%soft, %v : tensor<2x4x4xf16>, tensor<2x4x8xf16>)
      outs(%out_empty : tensor<2x4x8xf16>) -> tensor<2x4x8xf16>
  return %out : tensor<2x4x8xf16>
}

// CHECK: SchedulePatternView:
// CHECK:   kernel = kernel_0
// CHECK:   ops = 4
// CHECK:   handwritten_kind = "attention_sdpa"
// CHECK: ScheduleProblem:
// CHECK:   template_tags = [cube, attention_sdpa]
// CHECK:   structure_constraints = [matmul_contract, branch_merge_axes_consistent, handwritten_group, attention_sdpa_chain]
// CHECK: TemplateRegistry:
// CHECK:   kernel = kernel_0
// CHECK:   template = attention_sdpa/grouped_tile_per_block
// CHECK: Schedule report
// CHECK: schedule_family = "attention_sdpa"
// CHECK: linalg.batch_matmul
// CHECK-SAME: ascend.schedule.family = "attention_sdpa"
// CHECK-SAME: ascend.schedule.template = "grouped_tile_per_block"
