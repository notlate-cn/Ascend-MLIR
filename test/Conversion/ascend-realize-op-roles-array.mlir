// RUN: ascend-mlir-opt %s --ascend-realize='materialization-mode=memory-space-annotate dump-report=true debug-stage=realize' 2>&1 | FileCheck %s

func.func @memory_space_annotate_from_roles_array(%arg0: tensor<64xf16>,
                                                  %arg1: tensor<64xf16>)
    -> tensor<64xf16> attributes {ascend.normalized = true} {
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
      ascend.op_roles = ["Primary", "Vector", "Injective"],
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
      ascend.op_roles = ["Primary", "Vector", "Injective"],
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
// CHECK: MemoryRealizationPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "memory_space_annotate"
// CHECK-NEXT:   frozen = true
// CHECK-NEXT:   verification_scope = "memory_space_annotation"
// CHECK-NEXT:   plan_ids_verified = true
// CHECK-NEXT:   memory_space_annotations = 1
// CHECK: memref.alloc() {{.*}} : memref<64xf16, 11 : i32>
