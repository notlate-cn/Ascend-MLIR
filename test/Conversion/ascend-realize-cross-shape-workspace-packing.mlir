// RUN: afir-opt %s --ascend-realize='placement-mode=target-aware cann-root=%S/Inputs/ascend-target-aware-placement-cann soc=SyntheticSoC materialization-mode=memory-space-annotate dump-report=true debug-stage=realize' 2>&1 | FileCheck %s

func.func @cross_shape_workspace_packing(%arg0: tensor<64xf16>,
                                         %arg1: tensor<64xf16>,
                                         %arg2: tensor<32xf16>,
                                         %arg3: tensor<32xf16>)
    -> (tensor<64xf16>, tensor<32xf16>) attributes {ascend.normalized = true} {
  %empty0 = tensor.empty() : tensor<64xf16>
  %mid0 = linalg.generic {
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
  %out0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%mid0 : tensor<64xf16>)
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

  %empty2 = tensor.empty() : tensor<32xf16>
  %mid1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg2, %arg3 : tensor<32xf16>, tensor<32xf16>)
    outs(%empty2 : tensor<32xf16>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_0.decision.0",
      ascend.schedule.structured_lowering = "loop_skeleton_v0"
    } {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<32xf16>

  %empty3 = tensor.empty() : tensor<32xf16>
  %out1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%mid1 : tensor<32xf16>)
    outs(%empty3 : tensor<32xf16>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_0.decision.0",
      ascend.schedule.structured_lowering = "loop_skeleton_v0"
    } {
  ^bb0(%x: f16, %o: f16):
    %v = arith.negf %x : f16
    linalg.yield %v : f16
  } -> tensor<32xf16>

  return %out0, %out1 : tensor<64xf16>, tensor<32xf16>
}

// CHECK: Ascend realize report (ascend-realize)
// CHECK: Realize report
// CHECK: StaticMemoryPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "workspace_layout"
// CHECK-NEXT:   tracked_places = 8
// CHECK-NEXT:   local_buffers = 2
// CHECK-NEXT:   live_intervals = 2
// CHECK-NEXT:   workspace_slots = 2
// CHECK-NEXT:   peak_usage_known = true
// CHECK-NEXT:   peak_usage_units = 1
// CHECK-NEXT:   peak_usage_bytes_known = true
// CHECK-NEXT:   local_buffer_bytes = 192
// CHECK-NEXT:   workspace_bytes = 128
// CHECK-NEXT:   peak_usage_bytes = 128
// CHECK: MemoryRealizationPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "memory_space_materialize"
// CHECK-NEXT:   frozen = true
// CHECK-NEXT:   verification_scope = "memory_space_materialization"
// CHECK-NEXT:   plan_ids_verified = true
// CHECK-NEXT:   memory_space_annotations = 0
// CHECK-NEXT:   materialized_allocs = 1
// CHECK-NEXT:   materialized_copies = 2
// CHECK: %[[WORKSPACE:.*]] = memref.alloc() : memref<64xf16, 9 : i32>
// CHECK: %[[VIEW0:.*]] = memref.reinterpret_cast %[[WORKSPACE]] to offset: [0], sizes: [64], strides: [1] : memref<64xf16, 9 : i32> to memref<64xf16, strided<[1]>, 9 : i32>
// CHECK: memref.copy {{.*}}, %[[VIEW0]] : memref<64xf16> to memref<64xf16, strided<[1]>, 9 : i32>
// CHECK: %[[VIEW1:.*]] = memref.reinterpret_cast %[[WORKSPACE]] to offset: [0], sizes: [32], strides: [1] : memref<64xf16, 9 : i32> to memref<32xf16, strided<[1]>, 9 : i32>
// CHECK: memref.copy {{.*}}, %[[VIEW1]] : memref<32xf16> to memref<32xf16, strided<[1]>, 9 : i32>
