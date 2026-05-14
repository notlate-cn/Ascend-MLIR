// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

// CHECK-LABEL: func.func @named_rank2_leading_swap_gm
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: linalg.transpose
func.func @named_rank2_leading_swap_gm(%src: memref<384x128xf32>,
                                       %dst: memref<128x384xf32>) {
  linalg.transpose ins(%src : memref<384x128xf32>)
      outs(%dst : memref<128x384xf32>)
      permutation = [1, 0]
  return
}

// CHECK-LABEL: func.func @named_rank3_leading_swap_gm
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: linalg.transpose
func.func @named_rank3_leading_swap_gm(%src: memref<?x?x128xf32>,
                                       %dst: memref<?x?x128xf32>) {
  linalg.transpose ins(%src : memref<?x?x128xf32>)
      outs(%dst : memref<?x?x128xf32>)
      permutation = [1, 0, 2]
  return
}

// CHECK-LABEL: func.func @named_rank5_permutation_gm
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: linalg.transpose
func.func @named_rank5_permutation_gm(%src: memref<1x2x3x4x5xf32>,
                                      %dst: memref<4x2x3x1x5xf32>) {
  linalg.transpose ins(%src : memref<1x2x3x4x5xf32>)
      outs(%dst : memref<4x2x3x1x5xf32>)
      permutation = [3, 1, 2, 0, 4]
  return
}
