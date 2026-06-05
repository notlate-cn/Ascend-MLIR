// RUN: ascend-mlir-opt %s --ascend-compute-lower | FileCheck %s

// CHECK-LABEL: func.func @copy_gm_to_gm
// CHECK: ascendc.global_tensor
// CHECK: ascendc.global_tensor
// CHECK: ascendc.data_copy_l2
// CHECK-NOT: memref.copy
func.func @copy_gm_to_gm(%src: memref<?x?xf32>, %dst: memref<?x?xf32>) {
  memref.copy %src, %dst : memref<?x?xf32> to memref<?x?xf32>
  return
}
