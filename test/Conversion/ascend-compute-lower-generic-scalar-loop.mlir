// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

#identity4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @rank4_trunc_mul_gm
// CHECK: scf.for
// CHECK: arith.truncf
// CHECK: arith.mulf
// CHECK: memref.store
// CHECK-NOT: linalg.generic
func.func @rank4_trunc_mul_gm(%src: memref<?x4x?x?xf32>,
                              %dst: memref<?x4x?x?xf32>) {
  %scale = arith.constant 1.7677669529663687E-1 : f64
  linalg.generic {
      indexing_maps = [#identity4, #identity4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%src : memref<?x4x?x?xf32>)
      outs(%dst : memref<?x4x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %scale_f32 = arith.truncf %scale : f64 to f32
      %scaled = arith.mulf %in, %scale_f32 : f32
      linalg.yield %scaled : f32
  }
  return
}
