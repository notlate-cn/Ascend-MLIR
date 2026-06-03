// RUN: afir-opt %s --ascend-realize='placement-mode=target-aware cann-root=%S/Inputs/ascend-target-aware-placement-cann soc=SyntheticSoC materialization-mode=memory-space-annotate dump-report=true debug-stage=realize' 2>&1 | FileCheck %s

func.func @view_chain_movement(%arg0: tensor<64xf16>,
                               %arg1: tensor<64xf16>) -> tensor<32xf16>
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
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_0.decision.0",
      ascend.schedule.schedule_contract = "generic_tiled_loop"
    } {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>

  %slice = tensor.extract_slice %mid[0] [32] [1]
      : tensor<64xf16> to tensor<32xf16>

  %empty1 = tensor.empty() : tensor<32xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%slice : tensor<32xf16>)
    outs(%empty1 : tensor<32xf16>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_0.decision.0",
      ascend.schedule.schedule_contract = "generic_tiled_loop"
    } {
  ^bb0(%x: f16, %o: f16):
    %v = arith.negf %x : f16
    linalg.yield %v : f16
  } -> tensor<32xf16>

  return %out : tensor<32xf16>
}

// CHECK: Ascend realize report (ascend-realize)
// CHECK: Realize report
// CHECK: MemoryRealizationPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "memory_space_materialize"
// CHECK-NEXT:   frozen = true
// CHECK-NEXT:   verification_scope = "memory_space_materialization"
// CHECK-NEXT:   plan_ids_verified = true
// CHECK-NEXT:   memory_space_annotations = 0
// CHECK-NEXT:   materialized_allocs = 2
// CHECK-NEXT:   materialized_copies = 2
// CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<64xf16, 9 : i32>
// CHECK: memref.copy {{.*}}, %[[LOCAL]] : memref<64xf16> to memref<64xf16, 9 : i32>
// CHECK: %[[SUBVIEW:.*]] = memref.subview %[[LOCAL]][0] [32] [1] : memref<64xf16, 9 : i32> to memref<32xf16, strided<[1]>, 9 : i32>
// CHECK: %[[VECOUT:.*]] = memref.alloc() {{.*}} : memref<32xf16, 10 : i32>
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[SUBVIEW]]
// CHECK-SAME: outs(%[[VECOUT]]
// CHECK: memref.copy %[[VECOUT]]
