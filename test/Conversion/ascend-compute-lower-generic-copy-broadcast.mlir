// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

#project = affine_map<(d0, d1, d2) -> (d1, d2)>
#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @rank3_projected_copy_gm
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: linalg.generic
func.func @rank3_projected_copy_gm(%src: memref<128x384xf32>,
                                   %dst: memref<?x128x384xf32>) {
  linalg.generic {
      indexing_maps = [#project, #identity3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%src : memref<128x384xf32>)
      outs(%dst : memref<?x128x384xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return
}
