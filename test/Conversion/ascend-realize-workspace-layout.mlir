// RUN: afir-opt %s --ascend-realize='placement-mode=target-aware cann-root=%S/Inputs/ascend-target-aware-placement-cann soc=SyntheticSoC dump-report=true debug-stage=realize' 2>&1 | FileCheck %s
// RUN: afir-opt %s --ascend-realize='placement-mode=target-aware cann-root=%S/Inputs/ascend-target-aware-placement-cann soc=SyntheticSoC materialization-mode=memory-space-annotate dump-report=true debug-stage=realize' 2>&1 | FileCheck %s --check-prefix=MATERIALIZE

func.func @workspace_layout_vector_temporary(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> attributes {ascend.normalized = true} {
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
// CHECK: PlacementPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "target_aware"
// CHECK-NEXT:   selected_places = 4
// CHECK-NEXT:   gm_places = 3
// CHECK-NEXT:   on_chip_places = 1
// CHECK-NEXT:   deferred_local_places = 0
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
// CHECK-NEXT:   capacity_check_deferred = false
// CHECK: MovementPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "movement_planning"
// CHECK-NEXT:   cross_place_edges = 1
// CHECK-NEXT:   movements = 0
// CHECK-NEXT:   redundant_movements = 0
// CHECK-NEXT:   movement_demands = 1
// CHECK-NEXT:   selected_paths = 1
// CHECK-NEXT:   path_selection_deferred = 0
// CHECK-NEXT:   workspace_reuse_candidates = 1
// CHECK-NEXT:   dynamic_view_chain_rewrites = 0
// CHECK-NEXT:   deferred_view_chain_rewrites = 0
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

// MATERIALIZE-LABEL: Realize report
// MATERIALIZE: MovementPlan:
// MATERIALIZE-NEXT:   kernel = kernel_0
// MATERIALIZE-NEXT:   mode = "movement_planning"
// MATERIALIZE-NEXT:   cross_place_edges = 1
// MATERIALIZE-NEXT:   movements = 0
// MATERIALIZE-NEXT:   redundant_movements = 0
// MATERIALIZE-NEXT:   movement_demands = 1
// MATERIALIZE-NEXT:   selected_paths = 1
// MATERIALIZE-NEXT:   path_selection_deferred = 0
// MATERIALIZE-NEXT:   workspace_reuse_candidates = 1
// MATERIALIZE-NEXT:   dynamic_view_chain_rewrites = 0
// MATERIALIZE-NEXT:   deferred_view_chain_rewrites = 0
// MATERIALIZE-NEXT:   materialization_deferred = true
// MATERIALIZE: MemoryRealizationPlan:
// MATERIALIZE-NEXT:   kernel = kernel_0
// MATERIALIZE-NEXT:   mode = "memory_space_materialize"
// MATERIALIZE-NEXT:   frozen = true
// MATERIALIZE-NEXT:   verification_scope = "memory_space_materialization"
// MATERIALIZE-NEXT:   plan_ids_verified = true
// MATERIALIZE-NEXT:   memory_space_annotations = 0
// MATERIALIZE-NEXT:   materialized_allocs = 2
// MATERIALIZE-NEXT:   materialized_copies = 2
// MATERIALIZE: memref.alloc() : memref<64xf16, 9 : i32>
// MATERIALIZE: memref.copy {{.*}} : memref<64xf16> to memref<64xf16, 9 : i32>
// MATERIALIZE: memref.alloc() {{.*}} : memref<64xf16, 10 : i32>
// MATERIALIZE: memref.copy {{.*}} : memref<64xf16, 10 : i32> to memref<64xf16>
