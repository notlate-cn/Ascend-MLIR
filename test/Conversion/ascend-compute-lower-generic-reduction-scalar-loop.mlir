// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

#input = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#output = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @rank4_max_argmax_reduction_gm
// CHECK: scf.for
// CHECK: arith.index_cast
// CHECK: arith.maximumf
// CHECK: arith.select
// CHECK: memref.store
// CHECK-NOT: linalg.index
// CHECK-NOT: linalg.generic
func.func @rank4_max_argmax_reduction_gm(%src: memref<?x4x?x?xf32>,
                                         %max: memref<?x4x?xf32>,
                                         %argmax: memref<?x4x?xi64>) {
  linalg.generic {
      indexing_maps = [#input, #output, #output],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%src : memref<?x4x?x?xf32>)
      outs(%max, %argmax : memref<?x4x?xf32>, memref<?x4x?xi64>) {
    ^bb0(%in: f32, %old_max: f32, %old_idx: i64):
      %k = linalg.index 3 : index
      %k_i64 = arith.index_cast %k : index to i64
      %next_max = arith.maximumf %in, %old_max : f32
      %is_greater = arith.cmpf ogt, %in, %old_max : f32
      %next_idx = arith.select %is_greater, %k_i64, %old_idx : i64
      linalg.yield %next_max, %next_idx : f32, i64
  }
  return
}
