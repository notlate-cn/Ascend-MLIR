// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='dump-report=true debug-stage=realize' 2>&1 | FileCheck %s

func.func @elementwise(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> {
  %empty = tensor.empty() : tensor<64xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty : tensor<64xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>
  return %out : tensor<64xf16>
}

// CHECK: Ascend realize report (ascend-realize)
// CHECK: DebugStepCatalog:
// CHECK:   step = "realize.plan-memory"
// CHECK:   title = "Build memory realization plan"
// CHECK:   step = "realize.bufferize"
// CHECK:   outputs = "memref IR with explicit buffers and view chains"
// CHECK:   step = "realize.annotate-memory-space"
// CHECK:   inspect_hint = "Check alloc/copy nodes, memory_space attrs, workspace slots, and deferred movement counts."
// CHECK: Realize report
// CHECK-NEXT:   kernels = 1
// CHECK: BufferizedKernelIR:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "tensor_facts"
// CHECK-NEXT:   buffer_values = 3
// CHECK-NEXT:   input_values = 2
// CHECK-NEXT:   output_values = 1
// CHECK-NEXT:   temporary_values = 0
// CHECK: PlacementPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "gm_default"
// CHECK-NEXT:   selected_places = 3
// CHECK-NEXT:   gm_places = 3
// CHECK-NEXT:   on_chip_places = 0
// CHECK-NEXT:   deferred_local_places = 0
// CHECK: StaticMemoryPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "empty_workspace"
// CHECK-NEXT:   tracked_places = 3
// CHECK-NEXT:   local_buffers = 0
// CHECK-NEXT:   live_intervals = 0
// CHECK-NEXT:   workspace_slots = 0
// CHECK-NEXT:   peak_usage_known = false
// CHECK-NEXT:   peak_usage_units = 0
// CHECK-NEXT:   peak_usage_bytes_known = false
// CHECK-NEXT:   local_buffer_bytes = 0
// CHECK-NEXT:   workspace_bytes = 0
// CHECK-NEXT:   peak_usage_bytes = 0
// CHECK-NEXT:   capacity_check_deferred = false
// CHECK: MovementPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "gm_noop"
// CHECK-NEXT:   cross_place_edges = 0
// CHECK-NEXT:   movements = 0
// CHECK-NEXT:   redundant_movements = 0
// CHECK-NEXT:   movement_demands = 0
// CHECK-NEXT:   selected_paths = 0
// CHECK-NEXT:   path_selection_deferred = 0
// CHECK-NEXT:   workspace_reuse_candidates = 0
// CHECK-NEXT:   dynamic_view_chain_rewrites = 0
// CHECK-NEXT:   deferred_view_chain_rewrites = 0
// CHECK-NEXT:   materialization_deferred = false
// CHECK: MemoryRealizationPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "read_only_freeze"
// CHECK-NEXT:   frozen = true
// CHECK-NEXT:   verification_scope = "plan_identity_only"
// CHECK-NEXT:   plan_ids_verified = true
// CHECK-NEXT:   memory_space_annotations = 0
// CHECK-NEXT:   materialized_allocs = 0
// CHECK-NEXT:   materialized_copies = 0
// CHECK: linalg.generic
// CHECK-SAME: ascend.schedule.decision_id = "kernel_0.decision.0"
// CHECK-SAME: ascend.schedule.schedule_contract = "generic_tiled_loop"
