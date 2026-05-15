// RUN: afir-opt %s --ascend-realize='placement-mode=target-aware cann-root=%S/Inputs/ascend-target-aware-placement-cann soc=SyntheticSoC dump-report=true debug-stage=realize' 2>&1 | FileCheck %s

func.func @data_movement_plan_vector_temporary(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> attributes {ascend.normalized = true} {
  %empty0 = tensor.empty() : tensor<64xf16>
  %mid = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty0 : tensor<64xf16>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_0.decision.0",
      ascend.schedule.structured_lowering = "loop_skeleton_v0"
    } {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>

  %empty1 = tensor.empty() : tensor<64xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%mid : tensor<64xf16>)
    outs(%empty1 : tensor<64xf16>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_0.decision.0",
      ascend.schedule.structured_lowering = "loop_skeleton_v0"
    } {
  ^bb0(%x: f16, %o: f16):
    %v = arith.negf %x : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>

  return %out : tensor<64xf16>
}

// CHECK-LABEL: Realize report
// CHECK: StaticMemoryPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "workspace_layout"
// CHECK-NEXT:   tracked_places = 4
// CHECK-NEXT:   local_buffers = 1
// CHECK-NEXT:   live_intervals = 1
// CHECK-NEXT:   workspace_slots = 1
// CHECK-NEXT:   peak_usage_known = true
// CHECK-NEXT:   peak_usage_units = 1
// CHECK-NEXT:   peak_usage_bytes_known = true
// CHECK-NEXT:   local_buffer_bytes = 128
// CHECK-NEXT:   workspace_bytes = 128
// CHECK-NEXT:   peak_usage_bytes = 128
// CHECK-NEXT:   capacity_check_deferred = true
// CHECK: MovementPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "movement_planning"
// CHECK-NEXT:   cross_place_edges = 1
// CHECK-NEXT:   movements = 0
// CHECK-NEXT:   redundant_movements = 0
// CHECK-NEXT:   movement_demands = 1
// CHECK-NEXT:   selected_paths = 0
// CHECK-NEXT:   path_selection_deferred = 1
// CHECK-NEXT:   workspace_reuse_candidates = 1
// CHECK-NEXT:   materialization_deferred = true
// CHECK: MemoryRealizationPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "read_only_freeze"
// CHECK-NEXT:   frozen = true
// CHECK-NEXT:   verification_scope = "plan_identity_only"
// CHECK-NEXT:   plan_ids_verified = true
// CHECK-NEXT:   memory_space_annotations = 0
// CHECK-NEXT:   materialized_allocs = 0
// CHECK-NEXT:   materialized_copies = 0
// CHECK-NOT: memref.copy
// CHECK-NOT: memory_space
