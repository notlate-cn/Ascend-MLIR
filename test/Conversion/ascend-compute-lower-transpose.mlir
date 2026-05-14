// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

// CHECK-LABEL: func.func @named_rank2_transpose
// CHECK: ascendc.transpose
// CHECK-NOT: linalg.transpose
func.func @named_rank2_transpose() {
  %src = memref.alloc() : memref<16x32xf16, 9 : i32>
  %dst = memref.alloc() : memref<32x16xf16, 10 : i32>
  %gm = memref.alloc() : memref<32x16xf16>
  linalg.transpose ins(%src : memref<16x32xf16, 9 : i32>)
      outs(%dst : memref<32x16xf16, 10 : i32>)
      permutation = [1, 0]
  memref.copy %dst, %gm : memref<32x16xf16, 10 : i32> to memref<32x16xf16>
  return
}

// CHECK-LABEL: func.func @generic_rank2_transpose
// CHECK: ascendc.transpose
// CHECK-NOT: linalg.generic
#transpose = affine_map<(d0, d1) -> (d1, d0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
func.func @generic_rank2_transpose() {
  %src = memref.alloc() : memref<16x32xf16, 9 : i32>
  %dst = memref.alloc() : memref<32x16xf16, 10 : i32>
  %gm = memref.alloc() : memref<32x16xf16>
  linalg.generic {
      indexing_maps = [#transpose, #identity],
      iterator_types = ["parallel", "parallel"]}
      ins(%src : memref<16x32xf16, 9 : i32>)
      outs(%dst : memref<32x16xf16, 10 : i32>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
  }
  memref.copy %dst, %gm : memref<32x16xf16, 10 : i32> to memref<32x16xf16>
  return
}
