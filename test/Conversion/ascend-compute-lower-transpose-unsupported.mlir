// RUN: sed -n '/\/\/ NAMED-RANK3-BEGIN/,/\/\/ NAMED-RANK3-END/p' %s | not afir-opt --ascend-compute-lower 2>&1 | FileCheck %s --check-prefix=NAMED-RANK3
// RUN: sed -n '/\/\/ GENERIC-RANK3-BEGIN/,/\/\/ GENERIC-RANK3-END/p' %s | not afir-opt --ascend-compute-lower 2>&1 | FileCheck %s --check-prefix=GENERIC-RANK3

// NAMED-RANK3: unsupported compute kind unknown
// NAMED-RANK3: linalg.transpose
// GENERIC-RANK3: unsupported compute kind unknown
// GENERIC-RANK3: linalg.generic

// NAMED-RANK3-BEGIN
func.func @named_rank3_transpose() {
  %src = memref.alloc() : memref<2x4x8xf16, 9 : i32>
  %dst = memref.alloc() : memref<4x2x8xf16, 10 : i32>
  linalg.transpose ins(%src : memref<2x4x8xf16, 9 : i32>)
      outs(%dst : memref<4x2x8xf16, 10 : i32>)
      permutation = [1, 0, 2]
  return
}
// NAMED-RANK3-END

// GENERIC-RANK3-BEGIN
#rank3_transpose = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#rank3_identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @generic_rank3_transpose() {
  %src = memref.alloc() : memref<2x4x8xf16, 9 : i32>
  %dst = memref.alloc() : memref<4x2x8xf16, 10 : i32>
  linalg.generic {
      indexing_maps = [#rank3_transpose, #rank3_identity],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%src : memref<2x4x8xf16, 9 : i32>)
      outs(%dst : memref<4x2x8xf16, 10 : i32>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
  }
  return
}
// GENERIC-RANK3-END
