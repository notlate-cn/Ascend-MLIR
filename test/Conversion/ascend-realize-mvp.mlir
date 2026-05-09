// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule --ascend-realize='dump-report=true debug-stage=realize' 2>&1 | FileCheck %s

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
// CHECK-NEXT:   selected_places = 0
// CHECK: StaticMemoryPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   workspace_slots = 0
// CHECK: MovementPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   movements = 0
// CHECK: MemoryRealizationPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   frozen = true
// CHECK: linalg.generic
// CHECK-SAME: ascend.schedule.decision_id = "kernel_0.decision.0"
// CHECK-SAME: ascend.schedule.structured_lowering = "loop_skeleton_v0"
