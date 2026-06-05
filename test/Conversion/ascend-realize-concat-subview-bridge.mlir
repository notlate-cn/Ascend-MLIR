// RUN: ascend-mlir-opt %s --split-input-file --ascend-realize='materialization-mode=memory-space-annotate' | FileCheck %s

#identity = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @concat_copy_after_writer
// CHECK: %[[VEC:.*]] = memref.alloc() {{.*}} : memref<4x8xf16, 10 : i32>
// CHECK: linalg.generic
// CHECK-SAME: outs(%[[VEC]] : memref<4x8xf16, 10 : i32>)
// CHECK: %[[SUBVIEW:.*]] = memref.subview
// CHECK: memref.copy %[[VEC]], %[[SUBVIEW]] : memref<4x8xf16, 10 : i32> to memref<4x8xf16, strided<[8, 1]>>
// CHECK-NOT: memref.copy %{{.*}} : memref<4x8xf16> to
func.func @concat_copy_after_writer(%arg0: memref<4x8xf16>,
                                    %arg1: memref<4x8xf16>)
    -> memref<8x8xf16> attributes {ascend.normalized = true} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x8xf16>
  linalg.generic {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : memref<4x8xf16>, memref<4x8xf16>)
      outs(%alloc : memref<4x8xf16>)
      attrs = {ascend.kernel = "kernel_0",
               ascend.op_role = "vector",
               ascend.schedule.decision_id = "kernel_0.decision.0",
               ascend.schedule.schedule_contract = "generic_tiled_loop"} {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  }
  %out = memref.alloc() {alignment = 64 : i64} : memref<8x8xf16>
  %subview = memref.subview %out[0, 0] [4, 8] [1, 1]
      : memref<8x8xf16> to memref<4x8xf16, strided<[8, 1]>>
  memref.copy %alloc, %subview
      : memref<4x8xf16> to memref<4x8xf16, strided<[8, 1]>>
  return %out : memref<8x8xf16>
}

// -----

#identity = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @concat_copy_before_writer
// CHECK: %[[GM:.*]] = memref.alloc() {{.*}} : memref<4x8xf16>
// CHECK: memref.copy %[[GM]], %{{.*}} : memref<4x8xf16> to memref<4x8xf16, strided<[8, 1]>>
// CHECK: linalg.generic
// CHECK-SAME: outs(%[[GM]] : memref<4x8xf16>)
// CHECK-NOT: memref<4x8xf16, 10 : i32>
func.func @concat_copy_before_writer(%arg0: memref<4x8xf16>,
                                     %arg1: memref<4x8xf16>)
    -> memref<8x8xf16> attributes {ascend.normalized = true} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x8xf16>
  %out = memref.alloc() {alignment = 64 : i64} : memref<8x8xf16>
  %subview = memref.subview %out[0, 0] [4, 8] [1, 1]
      : memref<8x8xf16> to memref<4x8xf16, strided<[8, 1]>>
  memref.copy %alloc, %subview
      : memref<4x8xf16> to memref<4x8xf16, strided<[8, 1]>>
  linalg.generic {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : memref<4x8xf16>, memref<4x8xf16>)
      outs(%alloc : memref<4x8xf16>)
      attrs = {ascend.kernel = "kernel_0",
               ascend.op_role = "vector",
               ascend.schedule.decision_id = "kernel_0.decision.0",
               ascend.schedule.schedule_contract = "generic_tiled_loop"} {
  ^bb0(%a: f16, %b: f16, %out_elem: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  }
  return %out : memref<8x8xf16>
}
