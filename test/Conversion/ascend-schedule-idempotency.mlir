// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' | FileCheck %s

func.func @stale_schedule_metadata(%arg0: tensor<64xf16>,
                                   %arg1: tensor<64xf16>)
    -> tensor<64xf16>
    attributes {
      ascend.schedule.kernel_metadata = [
        {
          kernel = "kernel_0",
          guard_markers = [],
          tail_policies = [],
          tail_plan = [],
          tail_markers = [],
          target_tile_policy = "stale_policy"
        }
      ],
      ascend.schedule.tail_policies = [],
      ascend.schedule.tail_plan = [],
      ascend.schedule.target_tile_policy = "stale_policy"
    } {
  %empty = tensor.empty() : tensor<64xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty : tensor<64xf16>)
    attrs = {
      ascend.schedule.family = "stale_family",
      ascend.schedule.template = "stale_template",
      ascend.schedule.decision_id = "stale.decision",
      ascend.schedule.runtime_top_k = 99 : i64,
      ascend.schedule.schedule_contract = "stale_contract",
      ascend.schedule.tail_policies = [],
      ascend.schedule.tail_plan = [],
      ascend.schedule.target_tile_policy = "stale_policy"
    } {
  ^bb0(%x: f16, %y: f16, %old: f16):
    %sum = arith.addf %x, %y : f16
    linalg.yield %sum : f16
  } -> tensor<64xf16>
  return %out : tensor<64xf16>
}

// CHECK-LABEL: func.func @stale_schedule_metadata
// CHECK-SAME: ascend.schedule.kernel_metadata
// CHECK-SAME: kernel = "kernel_0"
// CHECK-NOT: stale_policy
// CHECK: linalg.generic
// CHECK-SAME: ascend.schedule.family = "vector_generic"
// CHECK-SAME: ascend.schedule.runtime_top_k = 1 : i64
// CHECK-SAME: ascend.schedule.target_tile_policy = "target_default_32"
// CHECK-NOT: stale_policy
