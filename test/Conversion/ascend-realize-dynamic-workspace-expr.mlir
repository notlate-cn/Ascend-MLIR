// RUN: ascend-mlir-opt %s --ascend-realize='placement-mode=target-aware cann-root=%S/Inputs/ascend-target-aware-placement-cann soc=SyntheticSoC dump-report=true debug-stage=realize' 2>&1 | FileCheck %s

func.func @dynamic_workspace_vector_temporary(
    %arg0: tensor<?x128xf16>, %arg1: tensor<?x128xf16>)
    -> tensor<?x128xf16> attributes {ascend.normalized = true} {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %arg0, %c0 : tensor<?x128xf16>
  %empty0 = tensor.empty(%m) : tensor<?x128xf16>
  %mid = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x128xf16>, tensor<?x128xf16>)
    outs(%empty0 : tensor<?x128xf16>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_0.decision.0",
      ascend.schedule.schedule_contract = "generic_tiled_loop"
    } {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<?x128xf16>

  %empty1 = tensor.empty(%m) : tensor<?x128xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%mid : tensor<?x128xf16>)
    outs(%empty1 : tensor<?x128xf16>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_0.decision.0",
      ascend.schedule.schedule_contract = "generic_tiled_loop"
    } {
  ^bb0(%x: f16, %o: f16):
    %v = arith.negf %x : f16
    linalg.yield %v : f16
  } -> tensor<?x128xf16>
  return %out : tensor<?x128xf16>
}

// CHECK: StaticMemoryPlan:
// CHECK:   peak_usage_bytes_known = false
// CHECK:   workspace_size_expr_known = true
// CHECK-NEXT:   workspace_size_expr = "dim_arg0_0 * 128 * 2"
// CHECK: func.func @dynamic_workspace_vector_temporary
// CHECK-SAME: cann.workspace_size_expr = "dim_arg0_0 * 128 * 2"
// CHECK-NOT: cann.workspace_size_bytes
