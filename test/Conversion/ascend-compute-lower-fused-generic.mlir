// RUN: ascend-mlir-opt %s --ascend-compute-lower | FileCheck %s

#map_broadcast = affine_map<(d0, d1) -> (d1, 0)>
#map_identity = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @fused_max_add_generic
// CHECK: ascendc.max_l2
// CHECK: ascendc.add_l2
// CHECK-NOT: linalg.generic
func.func @fused_max_add_generic() {
  %cst = arith.constant 0.000000e+00 : f16
  %lhs = memref.alloc() : memref<64x1xf16, 9 : i32>
  %rhs = memref.alloc() : memref<64x64xf16, 9 : i32>
  %out = memref.alloc() : memref<64x64xf16, 10 : i32>
  linalg.generic {
      indexing_maps = [#map_broadcast, #map_identity, #map_identity],
      iterator_types = ["parallel", "parallel"]}
      ins(%lhs, %rhs : memref<64x1xf16, 9 : i32>, memref<64x64xf16, 9 : i32>)
      outs(%out : memref<64x64xf16, 10 : i32>) {
    ^bb0(%in0: f16, %in1: f16, %out0: f16):
      %relu = arith.maximumf %in0, %cst : f16
      %sum = arith.addf %relu, %in1 : f16
      linalg.yield %sum : f16
    }
  return
}
