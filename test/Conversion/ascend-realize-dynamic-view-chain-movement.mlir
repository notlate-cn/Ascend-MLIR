// RUN: afir-opt %s --ascend-realize='placement-mode=target-aware cann-root=%S/Inputs/ascend-target-aware-placement-cann soc=SyntheticSoC materialization-mode=memory-space-annotate dump-report=true debug-stage=realize' 2>&1 | FileCheck %s

func.func @dynamic_extract_slice_consumer(
    %arg0: tensor<?x?xf32>, %out: tensor<?x?xf32>,
    %m: index, %n: index) -> tensor<?x?xf32>
    attributes {ascend.normalized = true} {
  %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
  %producer = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<?x?xf32>)
    outs(%empty : tensor<?x?xf32>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_0.decision.0",
      ascend.schedule.structured_lowering = "loop_skeleton_v0"
    } {
  ^bb0(%a: f32, %o: f32):
    linalg.yield %a : f32
  } -> tensor<?x?xf32>

  %c0 = arith.constant 0 : index
  %slice0 = tensor.extract_slice %producer[%c0, %c0][%m, %n][1, 1]
      : tensor<?x?xf32> to tensor<?x?xf32>
  %slice1 = tensor.extract_slice %producer[%c0, %c0][%m, %n][1, 1]
      : tensor<?x?xf32> to tensor<?x?xf32>

  %tmp = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%slice0 : tensor<?x?xf32>)
    outs(%out : tensor<?x?xf32>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_0.decision.0",
      ascend.schedule.structured_lowering = "loop_skeleton_v0"
    } {
  ^bb0(%a: f32, %o: f32):
    %neg = arith.negf %a : f32
    linalg.yield %neg : f32
  } -> tensor<?x?xf32>

  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%slice1 : tensor<?x?xf32>)
    outs(%out : tensor<?x?xf32>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_0.decision.0",
      ascend.schedule.structured_lowering = "loop_skeleton_v0"
    } {
  ^bb0(%a: f32, %o: f32):
    %neg = arith.negf %a : f32
    linalg.yield %neg : f32
  } -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// CHECK: MovementPlan:
// CHECK:   selected_paths = 1
// CHECK:   dynamic_view_chain_rewrites = 2
// CHECK:   deferred_view_chain_rewrites = 0
// CHECK: MemoryRealizationPlan:
// CHECK:   materialized_allocs = 1
// CHECK-NEXT:   materialized_copies = 1
// CHECK-LABEL: func.func @dynamic_extract_slice_consumer
// CHECK: memref.alloc(%{{.*}}, %{{.*}}) : memref<?x?xf32, 9 : i32>
// CHECK: memref.copy {{.*}} : memref<?x?xf32> to memref<?x?xf32, 9 : i32>
// CHECK: memref.subview {{.*}}[%{{.*}}, %{{.*}}] [%{{.*}}, %{{.*}}] [1, 1] : memref<?x?xf32, 9 : i32> to memref<?x?xf32, strided<[?, 1], offset: ?>, 9 : i32>
