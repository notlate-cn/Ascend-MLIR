// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=plan-only dump-report=true debug-stage=realize' 2>&1 | FileCheck %s --check-prefix=PLAN
// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='dump-report=true debug-stage=realize' 2>&1 | FileCheck %s --check-prefix=DEFAULT

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

// PLAN: Ascend realize report (ascend-realize)
// PLAN: DebugStepCatalog:
// PLAN:   step = "realize.plan-memory"
// PLAN:   title = "Build memory realization plan"
// PLAN:   step = "realize.bufferize"
// PLAN:   outputs = "memref IR with explicit buffers and view chains"
// PLAN:   step = "realize.annotate-memory-space"
// PLAN:   inspect_hint = "Check alloc/copy nodes, memory_space attrs, workspace slots, and deferred movement counts."
// PLAN: Realize report
// PLAN-NEXT:   kernels = 1
// PLAN: BufferizedKernelIR:
// PLAN-NEXT:   kernel = kernel_0
// PLAN-NEXT:   mode = "tensor_facts"
// PLAN-NEXT:   buffer_values = 3
// PLAN-NEXT:   input_values = 2
// PLAN-NEXT:   output_values = 1
// PLAN-NEXT:   temporary_values = 0
// PLAN: PlacementPlan:
// PLAN-NEXT:   kernel = kernel_0
// PLAN-NEXT:   mode = "gm_default"
// PLAN-NEXT:   selected_places = 3
// PLAN-NEXT:   gm_places = 3
// PLAN-NEXT:   on_chip_places = 0
// PLAN-NEXT:   deferred_local_places = 0
// PLAN: StaticMemoryPlan:
// PLAN-NEXT:   kernel = kernel_0
// PLAN-NEXT:   mode = "empty_workspace"
// PLAN-NEXT:   tracked_places = 3
// PLAN-NEXT:   local_buffers = 0
// PLAN-NEXT:   live_intervals = 0
// PLAN-NEXT:   workspace_slots = 0
// PLAN-NEXT:   peak_usage_known = false
// PLAN-NEXT:   peak_usage_units = 0
// PLAN-NEXT:   peak_usage_bytes_known = false
// PLAN-NEXT:   local_buffer_bytes = 0
// PLAN-NEXT:   workspace_bytes = 0
// PLAN-NEXT:   peak_usage_bytes = 0
// PLAN-NEXT:   capacity_check_deferred = false
// PLAN: MovementPlan:
// PLAN-NEXT:   kernel = kernel_0
// PLAN-NEXT:   mode = "gm_noop"
// PLAN-NEXT:   cross_place_edges = 0
// PLAN-NEXT:   movements = 0
// PLAN-NEXT:   redundant_movements = 0
// PLAN-NEXT:   movement_demands = 0
// PLAN-NEXT:   selected_paths = 0
// PLAN-NEXT:   path_selection_deferred = 0
// PLAN-NEXT:   workspace_reuse_candidates = 0
// PLAN-NEXT:   dynamic_view_chain_rewrites = 0
// PLAN-NEXT:   deferred_view_chain_rewrites = 0
// PLAN-NEXT:   materialization_deferred = false
// PLAN: MemoryRealizationPlan:
// PLAN-NEXT:   kernel = kernel_0
// PLAN-NEXT:   mode = "read_only_freeze"
// PLAN-NEXT:   frozen = true
// PLAN-NEXT:   verification_scope = "plan_identity_only"
// PLAN-NEXT:   plan_ids_verified = true
// PLAN-NEXT:   memory_space_annotations = 0
// PLAN-NEXT:   materialized_allocs = 0
// PLAN-NEXT:   materialized_copies = 0
// PLAN: linalg.generic
// PLAN-SAME: ascend.schedule.decision_id = "kernel_0.decision.0"
// PLAN-SAME: ascend.schedule.schedule_contract = "generic_tiled_loop"

// DEFAULT: MemoryRealizationPlan:
// DEFAULT-NEXT:   kernel = kernel_0
// DEFAULT-NEXT:   mode = "memory_space_materialize"
// DEFAULT-NEXT:   frozen = true
// DEFAULT-NEXT:   verification_scope = "memory_space_materialization"
// DEFAULT-NEXT:   plan_ids_verified = true
// DEFAULT-NEXT:   memory_space_annotations = 0
// DEFAULT-NEXT:   materialized_allocs = 1
// DEFAULT-NEXT:   materialized_copies = 1
