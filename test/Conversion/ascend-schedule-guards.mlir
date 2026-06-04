// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @static_rank2(%arg0: tensor<4x8xf16>,
                        %arg1: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %empty = tensor.empty() : tensor<4x8xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>)
    outs(%empty : tensor<4x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<4x8xf16>
  return %out : tensor<4x8xf16>
}

func.func @dynamic_rank2(%arg0: tensor<?x8xf16>,
                         %arg1: tensor<?x8xf16>) -> tensor<?x8xf16> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x8xf16>
  %empty = tensor.empty(%d0) : tensor<?x8xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x8xf16>, tensor<?x8xf16>)
    outs(%empty : tensor<?x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.mulf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<?x8xf16>
  return %out : tensor<?x8xf16>
}

func.func @dynamic_rank1(%arg0: tensor<?xf16>,
                         %arg1: tensor<?xf16>) -> tensor<?xf16> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?xf16>
  %empty = tensor.empty(%d0) : tensor<?xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<?xf16>, tensor<?xf16>)
    outs(%empty : tensor<?xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<?xf16>
  return %out : tensor<?xf16>
}

func.func @reduction_prunes_full_tile(%arg0: tensor<2x2x2x2x2xf16>)
    -> tensor<2x2x2x2xf16> {
  %empty = tensor.empty() : tensor<2x2x2x2xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel",
                      "reduction"]
  } ins(%arg0 : tensor<2x2x2x2x2xf16>)
    outs(%empty : tensor<2x2x2x2xf16>) {
  ^bb0(%x: f16, %acc: f16):
    %v = arith.addf %acc, %x : f16
    linalg.yield %v : f16
  } -> tensor<2x2x2x2xf16>
  return %out : tensor<2x2x2x2xf16>
}

func.func @reduction_keeps_split_logical_axis_guard(%arg0: tensor<2x4xf16>)
    -> tensor<2xf16> {
  %empty = tensor.empty() : tensor<2xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<2x4xf16>)
    outs(%empty : tensor<2xf16>) {
  ^bb0(%x: f16, %acc: f16):
    %v = arith.addf %acc, %x : f16
    linalg.yield %v : f16
  } -> tensor<2xf16>
  return %out : tensor<2xf16>
}

// CHECK: ScheduleSearch:
// CHECK-NEXT:   kernel = kernel_0
// CHECK: ScheduleGuards:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   candidate_guards = 2
// CHECK-NEXT:   decision_guards = 0
// CHECK-NOT:   decision_guard = a0 % 64 == 0
// CHECK-NOT:   decision_guard =
// CHECK-NEXT:   guard_budget = 8
// CHECK-NEXT:   pruned_by_guard_budget = 0
// CHECK-NEXT:   candidate_guard = d0 == 4
// CHECK-NEXT:   candidate_guard = d1 == 8
// CHECK: ScheduleDecisionSet:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   decisions = 4
// CHECK-NEXT:   runtime_top_k = 1
// CHECK-NEXT:   selected = kernel_0.decision.0
// CHECK-NEXT:   candidate_guards = 2
// CHECK-NEXT:   decision_guards = 0

// CHECK: ScheduleSearch:
// CHECK-NEXT:   kernel = kernel_1
// CHECK: ScheduleGuards:
// CHECK-NEXT:   kernel = kernel_1
// CHECK-NEXT:   candidate_guards = 3
// CHECK-NEXT:   decision_guards = 0
// CHECK-NOT:   decision_guard = a0 % 64 == 0
// CHECK-NOT:   decision_guard =
// CHECK-NEXT:   guard_budget = 8
// CHECK-NEXT:   pruned_by_guard_budget = 0
// CHECK-NEXT:   candidate_guard = T_arg0_dim0 > 0
// CHECK-NEXT:   candidate_guard = T_arg0_dim0 <= arg0_dim0
// CHECK-NEXT:   candidate_guard = d1 == 8
// CHECK: ScheduleDecisionSet:
// CHECK-NEXT:   kernel = kernel_1
// CHECK-NEXT:   decisions = 4
// CHECK-NEXT:   runtime_top_k = 1
// CHECK-NEXT:   selected = kernel_1.decision.0
// CHECK-NEXT:   candidate_guards = 3
// CHECK-NEXT:   decision_guards = 0

// CHECK: ScheduleSearch:
// CHECK-NEXT:   kernel = kernel_2
// CHECK: ScheduleGuards:
// CHECK-NEXT:   kernel = kernel_2
// CHECK-NEXT:   candidate_guards = 2
// CHECK-NEXT:   decision_guards = 0
// CHECK-NEXT:   guard_budget = 8
// CHECK-NEXT:   pruned_by_guard_budget = 0
// CHECK-NEXT:   candidate_guard = T_arg0_dim0 > 0
// CHECK-NEXT:   candidate_guard = T_arg0_dim0 <= arg0_dim0
// CHECK: ScheduleDecisionSet:
// CHECK-NEXT:   kernel = kernel_2
// CHECK-NEXT:   decisions = 4
// CHECK-NEXT:   runtime_top_k = 1
// CHECK-NEXT:   selected = kernel_2.decision.0
// CHECK-NEXT:   candidate_guards = 2
// CHECK-NEXT:   decision_guards = 0

// CHECK: ScheduleSearch:
// CHECK-NEXT:   kernel = kernel_3
// CHECK-NEXT:   generated = 17
// CHECK-NEXT:   kept = 4
// CHECK-NEXT:   compile_time_top_k = 4
// CHECK-NEXT:   instance = kernel_3.reduction_static.0
// CHECK-NEXT:   instance = kernel_3.reduction_static.1
// CHECK-NEXT:   instance = kernel_3.reduction_static.2
// CHECK-NEXT:   instance = kernel_3.reduction_static.3
// CHECK: ScheduleGuards:
// CHECK-NEXT:   kernel = kernel_3
// CHECK-NEXT:   candidate_guards = 4
// CHECK-NEXT:   decision_guards = 0
// CHECK-NOT:   decision_guard =
// CHECK-NEXT:   guard_budget = 8
// CHECK-NEXT:   pruned_by_guard_budget = 0
// CHECK-NEXT:   candidate_guard = d0 == 2
// CHECK-NEXT:   candidate_guard = d1 == 2
// CHECK-NEXT:   candidate_guard = d2 == 2
// CHECK-NEXT:   candidate_guard = d3 == 2
// CHECK-NEXT:   kept_instance = kernel_3.reduction_static.1
// CHECK-NEXT:   candidate_guard = d0 == 2
// CHECK-NEXT:   candidate_guard = d1 == 2
// CHECK-NEXT:   candidate_guard = d2 == 2
// CHECK-NEXT:   candidate_guard = d3 == 2
// CHECK-NEXT:   kept_instance = kernel_3.reduction_static.2
// CHECK-NEXT:   candidate_guard = d0 == 2
// CHECK-NEXT:   candidate_guard = d1 == 2
// CHECK-NEXT:   candidate_guard = d2 == 2
// CHECK-NEXT:   candidate_guard = d3 == 2
// CHECK-NEXT:   kept_instance = kernel_3.reduction_static.3
// CHECK-NEXT:   candidate_guard = d0 == 2
// CHECK-NEXT:   candidate_guard = d1 == 2
// CHECK-NEXT:   candidate_guard = d2 == 2
// CHECK-NEXT:   candidate_guard = d3 == 2
// CHECK: ScheduleDecisionSet:
// CHECK-NEXT:   kernel = kernel_3
// CHECK-NEXT:   decisions = 4
// CHECK-NEXT:   runtime_top_k = 1
// CHECK-NEXT:   selected = kernel_3.decision.0
// CHECK-NEXT:   candidate_guards = 4
// CHECK-NEXT:   decision_guards = 0

// CHECK: ScheduleSearch:
// CHECK-NEXT:   kernel = kernel_4
// CHECK-NEXT:   generated = 3
// CHECK-NEXT:   kept = 3
// CHECK-NEXT:   compile_time_top_k = 4
// CHECK-NEXT:   instance = kernel_4.reduction_static.0
// CHECK-NEXT:   instance = kernel_4.reduction_static.1
// CHECK-NEXT:   instance = kernel_4.reduction_static.2
// CHECK: ScheduleGuards:
// CHECK-NEXT:   kernel = kernel_4
// CHECK-NEXT:   candidate_guards = 1
// CHECK-NEXT:   decision_guards = 0
// CHECK-NOT:   decision_guard =
// CHECK-NEXT:   guard_budget = 8
// CHECK-NEXT:   pruned_by_guard_budget = 0
// CHECK-NEXT:   candidate_guard = d0 == 2
// CHECK-NEXT:   kept_instance = kernel_4.reduction_static.1
// CHECK-NEXT:   candidate_guard = d0 == 2
// CHECK-NEXT:   kept_instance = kernel_4.reduction_static.2
// CHECK-NEXT:   candidate_guard = d0 == 2
// CHECK: ScheduleDecisionSet:
// CHECK-NEXT:   kernel = kernel_4
// CHECK-NEXT:   decisions = 3
// CHECK-NEXT:   runtime_top_k = 1
// CHECK-NEXT:   selected = kernel_4.decision.0
// CHECK-NEXT:   candidate_guards = 1
// CHECK-NEXT:   decision_guards = 0
