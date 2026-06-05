// RUN: ascend-mlir-opt %s --ascend-compute-lower | FileCheck %s

#project_suffix = affine_map<(d0, d1, d2) -> (d1, d2)>
#identity2 = affine_map<(d0, d1) -> (d0, d1)>
#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @projected_suffix_copy_uses_segment_datacopy
// CHECK: ascendc.global_tensor
// CHECK: ascendc.queue : <vecin, 1>
// CHECK: ascendc.que_bind.alloc_tensor
// CHECK: ascendc.data_copy_l2
// CHECK: ascendc.que_bind.enque_tensor
// CHECK: ascendc.que_bind.deque_tensor
// CHECK: scf.for
// CHECK: ascendc.global_tensor.set_global_buffer
// CHECK: ascendc.data_copy_l2
// CHECK: ascendc.que_bind.free_tensor
// CHECK-NOT: memref.load
// CHECK-NOT: memref.store
// CHECK-NOT: linalg.generic
func.func @projected_suffix_copy_uses_segment_datacopy(
    %src: memref<128x384xf32>,
    %dst: memref<?x128x384xf32>) {
  linalg.generic {
      indexing_maps = [#project_suffix, #identity3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%src : memref<128x384xf32>)
      outs(%dst : memref<?x128x384xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return
}

// CHECK-LABEL: func.func @exact_copy_uses_plain_gm_datacopy
// CHECK: ascendc.global_tensor
// CHECK: ascendc.global_tensor
// CHECK: ascendc.data_copy_l2
// CHECK-NOT: ascendc.queue
// CHECK-NOT: ascendc.que_bind.alloc_tensor
// CHECK-NOT: memref.load
// CHECK-NOT: memref.store
// CHECK-NOT: linalg.generic
func.func @exact_copy_uses_plain_gm_datacopy(%src: memref<?x48xf32>,
                                             %dst: memref<?x48xf32>) {
  linalg.generic {
      indexing_maps = [#identity2, #identity2],
      iterator_types = ["parallel", "parallel"]}
      ins(%src : memref<?x48xf32>)
      outs(%dst : memref<?x48xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return
}
