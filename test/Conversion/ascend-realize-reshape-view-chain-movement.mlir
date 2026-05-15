// RUN: afir-opt %s --ascend-realize='placement-mode=target-aware cann-root=%S/Inputs/ascend-target-aware-placement-cann soc=SyntheticSoC materialization-mode=memory-space-annotate dump-report=true debug-stage=realize' 2>&1 | FileCheck %s

func.func @collapse_chain_movement(%arg0: tensor<4x16xf16>,
                                   %arg1: tensor<4x16xf16>) -> tensor<64xf16>
    attributes {ascend.normalized = true} {
  %empty0 = tensor.empty() : tensor<4x16xf16>
  %mid = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x16xf16>, tensor<4x16xf16>)
    outs(%empty0 : tensor<4x16xf16>)
    attrs = {
      ascend.kernel = "collapse_kernel",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "collapse_kernel.decision.0",
      ascend.schedule.structured_lowering = "loop_skeleton_v0"
    } {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<4x16xf16>

  %flat = tensor.collapse_shape %mid [[0, 1]]
      : tensor<4x16xf16> into tensor<64xf16>

  %empty1 = tensor.empty() : tensor<64xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%flat : tensor<64xf16>)
    outs(%empty1 : tensor<64xf16>)
    attrs = {
      ascend.kernel = "collapse_kernel",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "collapse_kernel.decision.0",
      ascend.schedule.structured_lowering = "loop_skeleton_v0"
    } {
  ^bb0(%x: f16, %o: f16):
    %v = arith.negf %x : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>

  return %out : tensor<64xf16>
}

func.func @expand_chain_movement(%arg0: tensor<64xf16>,
                                 %arg1: tensor<64xf16>) -> tensor<4x16xf16>
    attributes {ascend.normalized = true} {
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
      ascend.kernel = "expand_kernel",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "expand_kernel.decision.0",
      ascend.schedule.structured_lowering = "loop_skeleton_v0"
    } {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>

  %expanded = tensor.expand_shape %mid [[0, 1]] output_shape [4, 16]
      : tensor<64xf16> into tensor<4x16xf16>

  %empty1 = tensor.empty() : tensor<4x16xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%expanded : tensor<4x16xf16>)
    outs(%empty1 : tensor<4x16xf16>)
    attrs = {
      ascend.kernel = "expand_kernel",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "expand_kernel.decision.0",
      ascend.schedule.structured_lowering = "loop_skeleton_v0"
    } {
  ^bb0(%x: f16, %o: f16):
    %v = arith.negf %x : f16
    linalg.yield %v : f16
  } -> tensor<4x16xf16>

  return %out : tensor<4x16xf16>
}

// CHECK: MemoryRealizationPlan:
// CHECK-NEXT:   kernel = collapse_kernel
// CHECK-NEXT:   mode = "memory_space_materialize"
// CHECK:   materialized_allocs = 1
// CHECK-NEXT:   materialized_copies = 1
// CHECK: MemoryRealizationPlan:
// CHECK-NEXT:   kernel = expand_kernel
// CHECK-NEXT:   mode = "memory_space_materialize"
// CHECK:   materialized_allocs = 1
// CHECK-NEXT:   materialized_copies = 1
// CHECK-LABEL: func.func @collapse_chain_movement
// CHECK: %[[COLLAPSE_LOCAL:.*]] = memref.alloc() : memref<4x16xf16, 9 : i32>
// CHECK: memref.copy {{.*}}, %[[COLLAPSE_LOCAL]] : memref<4x16xf16> to memref<4x16xf16, 9 : i32>
// CHECK: %[[COLLAPSED:.*]] = memref.collapse_shape %[[COLLAPSE_LOCAL]] {{\[\[}}0, 1]] : memref<4x16xf16, 9 : i32> into memref<64xf16, 9 : i32>
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[COLLAPSED]]
// CHECK-LABEL: func.func @expand_chain_movement
// CHECK: %[[EXPAND_LOCAL:.*]] = memref.alloc() : memref<64xf16, 9 : i32>
// CHECK: memref.copy {{.*}}, %[[EXPAND_LOCAL]] : memref<64xf16> to memref<64xf16, 9 : i32>
// CHECK: %[[EXPANDED:.*]] = memref.expand_shape %[[EXPAND_LOCAL]] {{\[\[}}0, 1]] output_shape [4, 16] : memref<64xf16, 9 : i32> into memref<4x16xf16, 9 : i32>
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[EXPANDED]]
