// RUN: ascend-mlir-opt %s --ascend-compute-lower | FileCheck %s

#input = affine_map<(d0, d1) -> (d0, d1)>
#output = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @gm_sum_reduction
// CHECK: ascendc.reduce_sum_2d_l2
// CHECK: ascendc.global_tensor
// CHECK: ascendc.data_copy_l2 {{.*}} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
// CHECK-NOT: memref.store
// CHECK-NOT: linalg.generic
func.func @gm_sum_reduction(%src: memref<?x?xf32>,
                            %out: memref<?xf32>) {
  linalg.generic {
      indexing_maps = [#input, #output],
      iterator_types = ["parallel", "reduction"]}
      ins(%src : memref<?x?xf32>)
      outs(%out : memref<?xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %next = arith.addf %acc, %value : f32
      linalg.yield %next : f32
  }
  return
}
