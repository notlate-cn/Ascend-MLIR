// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @large_rank2_vector(%arg0: tensor<640x15000xf16>,
                              %arg1: tensor<640x15000xf16>)
                              -> tensor<640x15000xf16> {
  %empty = tensor.empty() : tensor<640x15000xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<640x15000xf16>, tensor<640x15000xf16>)
    outs(%empty : tensor<640x15000xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<640x15000xf16>
  return %out : tensor<640x15000xf16>
}

// CHECK: TemplateRegistry:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   matches = 1
// CHECK-NEXT:   template = vector_generic/single_tile_per_block
// CHECK: ScheduleSearch:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   generated = 65
// CHECK-NEXT:   kept = 4
// CHECK-NEXT:   compile_time_top_k = 4
// CHECK: ScheduleDecisionSet:
// CHECK:   kernel = kernel_0
// CHECK:   decisions = 4
// CHECK:   runtime_top_k = 1
// CHECK:   selected = kernel_0.decision.0
// CHECK-NEXT:   candidate_guards = 2
// CHECK-NEXT:   decision_guards = 0
// CHECK-NEXT:   tile_params =
// CHECK: linalg.generic
// CHECK-SAME: ascend.schedule.family = "vector_generic"
// CHECK-SAME: ascend.schedule.tile_binding = "symbolic"
