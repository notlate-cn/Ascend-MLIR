// RUN: not ascend-mlir-opt %s --ascend-compute-lower 2>&1 | FileCheck %s

// CHECK: error: unsupported GM-output batch_matmul lowering: materialize GM tensors through cube/local buffers before lowering; scalar loop fallback is disabled
func.func @batch_matmul_gm(%lhs: memref<?x?x128xf32>,
                           %rhs: memref<?x128x384xf32>,
                           %out: memref<?x?x384xf32>) {
  linalg.batch_matmul ins(%lhs, %rhs : memref<?x?x128xf32>,
                                       memref<?x128x384xf32>)
      outs(%out : memref<?x?x384xf32>)
  return
}
