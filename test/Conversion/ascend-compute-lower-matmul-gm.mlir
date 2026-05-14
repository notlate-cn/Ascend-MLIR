// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

// CHECK-LABEL: func.func @matmul_gm
// CHECK: scf.for
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: memref.store
// CHECK-NOT: linalg.matmul
func.func @matmul_gm(%lhs: memref<?x128xf32>,
                     %rhs: memref<128x384xf32>,
                     %out: memref<?x384xf32>) {
  linalg.matmul ins(%lhs, %rhs : memref<?x128xf32>,
                                 memref<128x384xf32>)
      outs(%out : memref<?x384xf32>)
  return
}
