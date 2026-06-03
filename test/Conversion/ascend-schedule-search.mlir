// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @vector_rank2_a(%arg0: tensor<4x8xf16>,
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

func.func @vector_rank2_b(%arg0: tensor<4x8xf16>,
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
    %v = arith.mulf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<4x8xf16>
  return %out : tensor<4x8xf16>
}

func.func @reduction_split(%arg0: tensor<4x8xf16>) -> tensor<4xf16> {
  %empty = tensor.empty() : tensor<4xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<4x8xf16>)
    outs(%empty : tensor<4xf16>) {
  ^bb0(%x: f16, %acc: f16):
    %v = arith.addf %acc, %x : f16
    linalg.yield %v : f16
  } -> tensor<4xf16>
  return %out : tensor<4xf16>
}

func.func @dynamic_vector(%arg0: tensor<?x8xf16>,
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
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<?x8xf16>
  return %out : tensor<?x8xf16>
}

func.func @reduction_large_m(%arg0: tensor<640x15000xf16>) -> tensor<640xf16> {
  %empty = tensor.empty() : tensor<640xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<640x15000xf16>)
    outs(%empty : tensor<640xf16>) {
  ^bb0(%x: f16, %acc: f16):
    %v = arith.addf %acc, %x : f16
    linalg.yield %v : f16
  } -> tensor<640xf16>
  return %out : tensor<640xf16>
}

func.func @reduction_odd_extent(%arg0: tensor<4x5xf16>) -> tensor<4xf16> {
  %empty = tensor.empty() : tensor<4xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<4x5xf16>)
    outs(%empty : tensor<4xf16>) {
  ^bb0(%x: f16, %acc: f16):
    %v = arith.addf %acc, %x : f16
    linalg.yield %v : f16
  } -> tensor<4xf16>
  return %out : tensor<4xf16>
}

func.func @matmul_large_k_axis(%lhs: tensor<640x256xf16>,
                               %rhs: tensor<256x128xf16>)
    -> tensor<640x128xf16> {
  %empty = tensor.empty() : tensor<640x128xf16>
  %out = linalg.matmul
      ins(%lhs, %rhs : tensor<640x256xf16>, tensor<256x128xf16>)
      outs(%empty : tensor<640x128xf16>) -> tensor<640x128xf16>
  return %out : tensor<640x128xf16>
}

func.func @batch_matmul_logical_axes(%lhs: tensor<2x4x8xf16>,
                                     %rhs: tensor<2x8x16xf16>)
    -> tensor<2x4x16xf16> {
  %empty = tensor.empty() : tensor<2x4x16xf16>
  %out = linalg.batch_matmul
      ins(%lhs, %rhs : tensor<2x4x8xf16>, tensor<2x8x16xf16>)
      outs(%empty : tensor<2x4x16xf16>) -> tensor<2x4x16xf16>
  return %out : tensor<2x4x16xf16>
}

// CHECK: ScheduleSearch:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   generated = 12
// CHECK-NEXT:   kept = 4
// CHECK-NEXT:   compile_time_top_k = 4
// CHECK-NEXT:   instance = kernel_0.vector_generic.0
// CHECK-NEXT:   instance = kernel_0.vector_generic.1
// CHECK-NEXT:   instance = kernel_0.vector_generic.2
// CHECK-NEXT:   instance = kernel_0.vector_generic.3
// CHECK: ScheduleSearch:
// CHECK-NEXT:   kernel = kernel_1
// CHECK-NEXT:   generated = 12
// CHECK-NEXT:   kept = 4
// CHECK-NEXT:   compile_time_top_k = 4
// CHECK-NEXT:   instance = kernel_1.vector_generic.0
// CHECK-NEXT:   instance = kernel_1.vector_generic.1
// CHECK-NEXT:   instance = kernel_1.vector_generic.2
// CHECK-NEXT:   instance = kernel_1.vector_generic.3
// CHECK: ScheduleSearch:
// CHECK-NEXT:   kernel = kernel_2
// CHECK-NEXT:   generated = 4
// CHECK-NEXT:   kept = 4
// CHECK-NEXT:   compile_time_top_k = 4
// CHECK-NEXT:   instance = kernel_2.reduction_static.0
// CHECK-NEXT:   instance = kernel_2.reduction_static.1
// CHECK-NEXT:   instance = kernel_2.reduction_static.2
// CHECK-NEXT:   instance = kernel_2.reduction_static.3
// CHECK: ScheduleSearch:
// CHECK-NEXT:   kernel = kernel_3
// CHECK-NEXT:   generated = 24
// CHECK-NEXT:   kept = 4
// CHECK-NEXT:   compile_time_top_k = 4
// CHECK-NEXT:   instance = kernel_3.vector_generic.0
// CHECK-NEXT:   instance = kernel_3.vector_generic.1
// CHECK-NEXT:   instance = kernel_3.vector_generic.2
// CHECK-NEXT:   instance = kernel_3.vector_generic.3
// CHECK: ScheduleDecisionSet:
// CHECK:   kernel = kernel_3
// CHECK:   decisions = 4
// CHECK:   runtime_top_k = 1
// CHECK:   selected = kernel_3.decision.0
// CHECK-NEXT:   candidate_guards = 2
// CHECK-NEXT:   decision_guards = 0
// CHECK-NEXT:   tile_params = [name=TB_M axis=0 binding=runtime {{.*}}] [name=TB_N axis=1 binding=runtime {{.*}}]
// CHECK-NEXT:   tail_plans = [axis=0 selected=masked_tail affected=[data_copy,vector_compute,write_back] align=0 buffering=separate_tail_buffer guard=false extent=? tile=? main=? tail=?] [axis=1 selected=masked_tail affected=[data_copy,vector_compute,write_back] align=0 buffering=separate_tail_buffer guard=false extent=8 tile=? main=? tail=?]
// CHECK: ScheduleSearch:
// CHECK-NEXT:   kernel = kernel_4
// CHECK-NEXT:   generated = 10
// CHECK-NEXT:   kept = 4
// CHECK-NEXT:   compile_time_top_k = 4
// CHECK-NEXT:   instance = kernel_4.reduction_static.0
// CHECK-NEXT:   instance = kernel_4.reduction_static.1
// CHECK-NEXT:   instance = kernel_4.reduction_static.2
// CHECK-NEXT:   instance = kernel_4.reduction_static.3
// CHECK: ScheduleDecisionSet:
// CHECK:   kernel = kernel_4
// CHECK:   decisions = 4
// CHECK:   runtime_top_k = 1
// CHECK:   selected = kernel_4.decision.0
// CHECK-NEXT:   candidate_guards = 1
// CHECK-NEXT:   decision_guards = 0
// CHECK: ScheduleSearch:
// CHECK-NEXT:   kernel = kernel_5
// CHECK-NEXT:   generated = 3
// CHECK-NEXT:   kept = 3
// CHECK-NEXT:   compile_time_top_k = 4
// CHECK-NEXT:   instance = kernel_5.reduction_static.0
// CHECK-NEXT:   instance = kernel_5.reduction_static.1
// CHECK-NEXT:   instance = kernel_5.reduction_static.2
// CHECK: ScheduleSearch:
// CHECK-NEXT:   kernel = kernel_6
// CHECK-NEXT:   generated = 54
// CHECK-NEXT:   kept = 4
// CHECK-NEXT:   compile_time_top_k = 4
// CHECK-NEXT:   instance = kernel_6.cube_static_matmul.0
// CHECK-NEXT:   instance = kernel_6.cube_static_matmul.1
// CHECK-NEXT:   instance = kernel_6.cube_static_matmul.2
// CHECK-NEXT:   instance = kernel_6.cube_static_matmul.3
// CHECK: ScheduleDecisionSet:
// CHECK:   kernel = kernel_6
// CHECK:   selected = kernel_6.decision.0
// CHECK-NEXT:   candidate_guards = 2
// CHECK-NEXT:   decision_guards = 0
// CHECK-NEXT:   tile_params = {{.*}}[name=t_K axis=2 binding=extent {{.*}}]
// CHECK-NEXT:   tail_plans = {{.*}}[axis=2 selected=full_extent affected=[reduction] align=0 buffering=separate_tail_buffer guard=false extent=256 tile=256 main=256 tail=0]
// CHECK: ScheduleSearch:
// CHECK-NEXT:   kernel = kernel_7
// CHECK-NEXT:   generated = 24
// CHECK-NEXT:   kept = 4
// CHECK-NEXT:   compile_time_top_k = 4
// CHECK-NEXT:   instance = kernel_7.cube_static_matmul.0
// CHECK-NEXT:   instance = kernel_7.cube_static_matmul.1
// CHECK-NEXT:   instance = kernel_7.cube_static_matmul.2
// CHECK-NEXT:   instance = kernel_7.cube_static_matmul.3
// CHECK: ScheduleDecisionSet:
// CHECK:   kernel = kernel_7
// CHECK:   selected = kernel_7.decision.0
// CHECK-NEXT:   candidate_guards = 3
// CHECK-NEXT:   decision_guards = 0
// CHECK-NEXT:   tile_params = {{.*}}[name=Tb_N axis=3 binding=extent {{.*}}]
// CHECK-NEXT:   tail_plans = {{.*}}[axis=3 selected=full_extent affected=[reduction] align=0 buffering=separate_tail_buffer guard=false extent=8 tile=8 main=8 tail=0]
// CHECK: schedule_family = "vector_generic"
// CHECK: schedule_template = "single_tile_per_block"
// CHECK: ascend.schedule.family = "vector_generic"
// CHECK-SAME: ascend.schedule.target_tile_policy = "target_default_32"
