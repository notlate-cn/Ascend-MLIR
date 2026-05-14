// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

// CHECK-LABEL: func.func @batch_matmul_gm
// CHECK: scf.for
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: memref.store
// CHECK-NOT: linalg.batch_matmul
func.func @batch_matmul_gm(%lhs: memref<?x?x128xf32>,
                           %rhs: memref<?x128x384xf32>,
                           %out: memref<?x?x384xf32>) {
  linalg.batch_matmul ins(%lhs, %rhs : memref<?x?x128xf32>,
                                       memref<?x128x384xf32>)
      outs(%out : memref<?x?x384xf32>)
  return
}
