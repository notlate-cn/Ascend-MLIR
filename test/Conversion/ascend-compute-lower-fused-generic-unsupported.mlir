// RUN: not afir-opt %s --ascend-compute-lower 2>&1 | FileCheck %s

#map = affine_map<(d0) -> (d0)>

// CHECK: unsupported compute kind unknown
func.func @unsupported_fused_branch() {
  %lhs = memref.alloc() : memref<64xf16, 9 : i32>
  %rhs = memref.alloc() : memref<64xf16, 9 : i32>
  %out = memref.alloc() : memref<64xf16, 10 : i32>
  linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel"]}
      ins(%lhs, %rhs : memref<64xf16, 9 : i32>, memref<64xf16, 9 : i32>)
      outs(%out : memref<64xf16, 10 : i32>) {
    ^bb0(%in0: f16, %in1: f16, %out0: f16):
      %sum = arith.addf %in0, %in1 : f16
      %product = arith.mulf %in0, %in1 : f16
      %combined = arith.addf %sum, %product : f16
      linalg.yield %combined : f16
    }
  return
}
