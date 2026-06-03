// RUN: afir-opt %s --split-input-file --ascend-realize='dump-report=true debug-stage=realize' 2>&1 | FileCheck %s

func.func @scheduled_two_op_kernel(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> attributes {ascend.normalized = true} {
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
      ascend.schedule.schedule_contract = "generic_tiled_loop"
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
      ascend.schedule.schedule_contract = "generic_tiled_loop"
    } {
  ^bb0(%x: f16, %o: f16):
    %v = arith.negf %x : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>

  return %out : tensor<64xf16>
}

// CHECK-LABEL: Realize report
// CHECK-NEXT:   kernels = 1
// CHECK: BufferizedKernelIR:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "tensor_facts"
// CHECK-NEXT:   buffer_values = 4
// CHECK-NEXT:   input_values = 2
// CHECK-NEXT:   output_values = 1
// CHECK-NEXT:   temporary_values = 1
// CHECK: PlacementPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "gm_default"
// CHECK-NEXT:   selected_places = 4
// CHECK-NEXT:   gm_places = 4
// CHECK-NEXT:   on_chip_places = 0
// CHECK-NEXT:   deferred_local_places = 1

// -----

func.func @scheduled_dead_result_kernel(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) attributes {ascend.normalized = true} {
  %empty0 = tensor.empty() : tensor<64xf16>
  %dead = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty0 : tensor<64xf16>)
    attrs = {
      ascend.kernel = "kernel_1",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_1.decision.0",
      ascend.schedule.schedule_contract = "generic_tiled_loop"
    } {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>
  return
}

// CHECK-LABEL: Realize report
// CHECK-NEXT:   kernels = 1
// CHECK: BufferizedKernelIR:
// CHECK-NEXT:   kernel = kernel_1
// CHECK-NEXT:   mode = "tensor_facts"
// CHECK-NEXT:   buffer_values = 2
// CHECK-NEXT:   input_values = 2
// CHECK-NEXT:   output_values = 0
// CHECK-NEXT:   temporary_values = 0
// CHECK: PlacementPlan:
// CHECK-NEXT:   kernel = kernel_1
// CHECK-NEXT:   mode = "gm_default"
// CHECK-NEXT:   selected_places = 2
// CHECK-NEXT:   gm_places = 2
// CHECK-NEXT:   on_chip_places = 0
// CHECK-NEXT:   deferred_local_places = 0
