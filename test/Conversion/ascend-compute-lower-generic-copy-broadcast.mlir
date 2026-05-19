// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

#project = affine_map<(d0, d1, d2) -> (d1, d2)>
#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @rank3_projected_copy_gm
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
func.func @rank3_projected_copy_gm(%src: memref<128x384xf32>,
                                   %dst: memref<?x128x384xf32>) {
  linalg.generic {
      indexing_maps = [#project, #identity3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%src : memref<128x384xf32>)
      outs(%dst : memref<?x128x384xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return
}
