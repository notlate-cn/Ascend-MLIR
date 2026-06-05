// RUN: not ascend-mlir-opt %s --ascend-compute-lower 2>&1 | FileCheck %s

// CHECK: unsupported movement path VECCALC -> GM
func.func @unsupported_copy(%dst: memref<?x?xf32>) {
  %src = memref.alloc() : memref<16x16xf32, 11 : i32>
  memref.copy %src, %dst : memref<16x16xf32, 11 : i32> to memref<?x?xf32>
  return
}
